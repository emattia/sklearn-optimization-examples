# https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
clf = xgb.XGBClassifier(verbose=0, silent=True)
parameters = {
  "eta"              : [0.10, 0.20, 0.30] ,
  "max_depth"        : [3, 6, 12],
  #  "min_child_weight" : [ 1, 3, 5, 7 ],
  "gamma"            : [0.0, 0.2],
  "colsample_bytree" : [0.5 , 0.7]
} 
grid = GridSearchCV(clf,parameters, n_jobs=4,scoring="neg_log_loss",cv=3)
grid.fit(X_train, Y_train)