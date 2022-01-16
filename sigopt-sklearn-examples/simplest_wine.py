import sigopt.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
data=load_wine()
X,y=data.data,data.target
train_x,test_x,train_y,test_y = train_test_split(X,y)
svc = SVC()
validation_sets = [(test_x, test_y, "test")]
sigopt.set_project('random')
run_context = sigopt.sklearn.run(train_x, train_y, SVC(), validation_sets=validation_sets)
run_context.run.end()