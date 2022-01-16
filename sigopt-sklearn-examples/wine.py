import sigopt.sklearn
from sigopt.sklearn.run import run
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
data=load_wine()
X,y=data.data,data.target
train_x,test_x,train_y,test_y = train_test_split(X,y)
validation_sets = [(test_x, test_y, "test")]
def custom_acc(estimator, X, y_true):
  import numpy
  return numpy.mean(estimator.predict(X) == y_true)
sigopt.set_project('random')
run_context = sigopt.sklearn.run(
  train_x, 
  train_y, 
  SVC(), 
  validation_sets=validation_sets, 
  scoring={"my-acc":custom_acc, "my-f1":make_scorer(f1_score, average='micro')},
  run_options={"autolog_metrics":False}
)
run_context.run.end()