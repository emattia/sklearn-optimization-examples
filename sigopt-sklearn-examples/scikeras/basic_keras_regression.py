import numpy as np
import sigopt.sklearn
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow import keras
from typing import Dict, Iterable, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, make_scorer

class MLPRegressor(KerasRegressor):

    def __init__(
        self,
        hidden_layer_sizes=(100, ),
        optimizer="adam",
        optimizer__learning_rate=0.001,
        epochs=200,
        verbose=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.optimizer__learning_rate = optimizer__learning_rate
        self.epochs = epochs
        self.verbose = verbose

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        model = keras.Sequential()
        inp = keras.layers.Input(shape=(self.n_features_in_))
        model.add(inp)
        for hidden_layer_size in self.hidden_layer_sizes:
            layer = keras.layers.Dense(hidden_layer_size, activation="relu")
            model.add(layer)
        out = keras.layers.Dense(1)
        model.add(out)
        model.compile(loss="mse", optimizer=compile_kwargs["optimizer"])
        return model

y = np.arange(1000)
X = (y/2).reshape(-1, 1)
train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=.05)

sigopt.set_project('random')
params = dict(epochs=20)
run_context = sigopt.sklearn.run(
  MLPRegressor(**params), 
  train_X, 
  train_y, 
  params=params,
  validation_sets=[(test_X, test_y, "test")],
  scoring = {"mae": make_scorer(mean_absolute_error)},
  run_options={'name': 'Custom MLP Test Test'}
)
run_context.run.end()