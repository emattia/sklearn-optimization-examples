import sigopt.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
data=load_wine()
X,y=data.data,data.target
train_x,test_x,train_y,test_y = train_test_split(X,y)
sigopt.set_project('random')
sklearn_run_context = sigopt.sklearn.run(
  SVC(), 
  train_x, 
  train_y, 
  validation_sets = [(test_x, test_y, "test")],
  params={'C': 1.5}
)
# sklearn_run_context.run.end()