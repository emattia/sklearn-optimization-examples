import numpy
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sigopt.sklearn
data=load_wine()
X,y=data.data,data.target
train_x,test_x,train_y,test_y = train_test_split(X,y)
sigopt.set_project('random')
experiment = sigopt.sklearn.experiment(
  SVC(), 
  train_x, 
  train_y, 
  fixed_params={'kernel': 'rbf'},
  validation_sets=[(test_x, test_y, "test")], 
  scoring={'my-acc': lambda estimator, X, y_true: numpy.mean(estimator.predict(X) == y_true)},
  experiment_design={
    'parameters': [{'name': 'C', 'type': 'double', 'bounds': {'min':0.1, 'max':5}}],
    'metrics': [{'name': 'test - my-acc'}],
    'budget': 1
  }
)