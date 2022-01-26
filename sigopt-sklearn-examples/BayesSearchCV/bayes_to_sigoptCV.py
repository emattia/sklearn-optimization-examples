from skopt import BayesSearchCV
from sigopt.sklearn import SigOptSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
space = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'degree': Integer(1,8),
    'kernel': Categorical(['linear', 'poly', 'rbf']),
}
opt = BayesSearchCV(SVC(),space,n_iter=10,random_state=0)
#opt = SigOptSearchCV(SVC(),space,n_iter=2,random_state=0)
_ = opt.fit(X_train, y_train)
print(opt.score(X_test, y_test))