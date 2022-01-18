import sigopt.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
data=load_wine()
x,y=data.data,data.target
train_x,test_x,train_y,test_y = train_test_split(x,y)
test1_x,test2_x,test1_y,test2_y = train_test_split(test_x, test_y, test_size=.5)
clfs = [
  SVC(), 
  DecisionTreeClassifier(), 
  LogisticRegression(), 
  RandomForestClassifier(), 
  GradientBoostingClassifier(), 
  ExtraTreesClassifier(), 
  GaussianProcessClassifier(),
]
validation_sets = [(test1_x, test1_y, "test1"), (test2_x, test2_y, "test2")]
sigopt.set_project('random')
for clf in clfs:
  run_context = sigopt.sklearn.run(train_x, train_y, clf, validation_sets=validation_sets)
  run_context.run.end()