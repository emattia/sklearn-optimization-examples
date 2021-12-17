import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt

from skopt.searchcv import SigOptSearchCV, BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, fbeta_score, f1_score

import sigopt

# sigopt.set_project('random')

X, y = load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

# log-uniform: understand as search over p = exp(x) by varying x
def f(clf, x, y):
  return -1 #{'always scores -1': -1}

my_config = {
    'name': 'My Exp',
    'metrics': [{'name': 'always scores -1'}],
    'budget': 5
}

opt = SigOptSearchCV(
    SVC(), # swap for an sklearn.base.Estimator
    {
        'C': (1e-6, 1e+6, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8),  # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    },
    n_iter=2,
    cv=3,
    scoring={ 
        'acc': 'accuracy',
        'f2': make_scorer(fbeta_score, beta=2, average="weighted"), 
        'f1': make_scorer(f1_score, average='weighted'),
        'always scores -1': f
    },
    # scoring = f, #refit='score'
    # scoring=['accuracy', 'f1'],
    project_id = 'random',
    experiment_config = my_config
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
# print("test score: %s" % opt.score(X_test, y_test))