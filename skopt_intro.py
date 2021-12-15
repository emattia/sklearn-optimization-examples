import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt

from skopt.searchcv import SigOptSearchCV, BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, fbeta_score

import sigopt

# sigopt.set_project('random')

X, y = load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

# log-uniform: understand as search over p = exp(x) by varying x
def f(clf, x, y):
  return {'score': -1}

my_config = {
    'name': 'My Exp',
    # 'metrics': [{'name': 'accuracy'}],
    'budget': 5
}

opt = SigOptSearchCV(
    SVC(),
    {
        'C': (1e-6, 1e+6, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8),  # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    },
    n_iter=2,
    cv=2,
    # scoring={'f1': 'f1', 'acc': 'accuracy', 'acc2': make_scorer(accuracy_score), 'f2': make_scorer(fbeta_score, beta=2)},
    #scoring = f, #refit='score'
    #scoring=['accuracy', 'f1'],
    project_id = 'random',
    experiment_config = my_config
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
# print("test score: %s" % opt.score(X_test, y_test))