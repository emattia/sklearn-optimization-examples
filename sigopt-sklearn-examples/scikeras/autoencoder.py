import sigopt.sklearn
import scikeras
import warnings
from tensorflow import get_logger
get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", message="Setting the random state for TF")
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow import keras
from typing import Dict, Any
from sklearn.base import TransformerMixin
from sklearn.metrics import mean_squared_error
from scikeras.wrappers import BaseWrapper
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# https://www.adriangb.com/scikeras/stable/notebooks/AutoEncoders.html
class AutoEncoder(BaseWrapper, TransformerMixin):
    encoder_model_: BaseWrapper
    decoder_model_: BaseWrapper
    def _keras_build_fn(self, encoding_dim: int, meta: Dict[str, Any]):
        n_features_in = meta["n_features_in_"]
        encoder_input = keras.Input(shape=(n_features_in,))
        encoder_output = keras.layers.Dense(encoding_dim, activation='relu')(encoder_input)
        encoder_model = keras.Model(encoder_input, encoder_output)
        decoder_input = keras.Input(shape=(encoding_dim,))
        decoder_output = keras.layers.Dense(n_features_in, activation='sigmoid', name="decoder")(decoder_input)
        decoder_model = keras.Model(decoder_input, decoder_output)
        autoencoder_input = keras.Input(shape=(n_features_in,))
        encoded_img = encoder_model(autoencoder_input)
        reconstructed_img = decoder_model(encoded_img)
        autoencoder_model = keras.Model(autoencoder_input, reconstructed_img)
        self.encoder_model_ = BaseWrapper(encoder_model, verbose=self.verbose)
        self.decoder_model_ = BaseWrapper(decoder_model, verbose=self.verbose)
        return autoencoder_model

    def _initialize(self, X, y=None):
        X, _ = super()._initialize(X=X, y=y)
        X_tf = self.encoder_model_.initialize(X).predict(X)
        self.decoder_model_.initialize(X_tf)
        return X, X

    def initialize(self, X):
        self._initialize(X=X, y=X)
        return self

    def fit(self, X, y=None, *, sample_weight=None) -> "AutoEncoder":
        super().fit(X=X, y=X, sample_weight=sample_weight)
        return self

    def score(self, X) -> float:
        return 1 - mean_squared_error(self.predict(X), X)

    def transform(self, X) -> np.ndarray:
        X: np.ndarray = self.feature_encoder_.transform(X)
        return self.encoder_model_.predict(X)

    def inverse_transform(self, X_tf: np.ndarray):
        X: np.ndarray = self.decoder_model_.predict(X_tf)
        return self.feature_encoder_.inverse_transform(X)

### LOAD DATA ### 
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

### ESTIMATOR CONFIG ### 
params = dict(
    loss="binary_crossentropy",
    encoding_dim=32,
    random_state=0,
    epochs=5,
    verbose=False,
    optimizer="adam",
)
sigopt.set_project('random')
### FIT AND TRACK ### 
run_context = sigopt.sklearn.run(
  AutoEncoder(**params), 
  x_train, 
  params=params,
  run_options={
    'name':'autoencoder', 
    'autolog_metrics':False, 
    'learning_task':'unsupervised',
    'params_as_metadata': ['verbose', 'random_state', 'validation_split', 'loss']
  },
)
autoencoder = run_context.model
roundtrip_imgs = autoencoder.inverse_transform(autoencoder.transform(x_test))

n = 10  # How many digits we will display
f = plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(roundtrip_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

run_context.run.log_image(f, name="autoencoder generated images vs. originals")
run_context.run.end()