import numpy as np
import sigopt.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

X, y = make_classification(10000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)
train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=.05)

def get_model(hidden_layer_dim, meta):
    # note that meta is a special argument that will be
    # handed a dict containing input metadata
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_features_in_, input_shape=X_shape_[1:]))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(hidden_layer_dim))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(n_classes_))
    model.add(keras.layers.Activation("softmax"))
    return model

params = dict(loss="sparse_categorical_crossentropy",hidden_layer_dim=100)
clf = KerasClassifier(model=get_model, **params)
sigopt.set_project('random')
run_context = sigopt.sklearn.run(
  clf, 
  train_X, 
  train_y, 
  params=params, 
  validation_sets=[(test_X, test_y, "test")],
  scoring = {"accuracy": make_scorer(accuracy_score)},
  run_options={'autolog_metrics': False}
)