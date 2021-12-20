import numpy as np
from numpy.lib.function_base import average
np.random.seed(123)
import matplotlib.pyplot as plt

from skopt.searchcv import SigOptSearchCV, BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, fbeta_score, f1_score

import sigopt

# sigopt.set_project('random')

X, y = load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

# log-uniform: understand as search over p = exp(x) by varying x
def f(clf, x, y):
  return -1 #{'always scores -1': -1}

parameters = [    ### SIGOPT PARAM CONFIG ### 
  {'name': 'C', 'type': 'double', 'bounds': {'min': 1e-6, 'max':1e6}},
  {'name': 'gamma', 'type': 'double', 'bounds': {'min': 1e-6, 'max':1e1}},
  {'name': 'degree', 'type': 'int', 'bounds': {'min': 1, 'max':8}},
  {'name': 'kernel', 'type': 'categorical', 'categorical_values': ['linear', 'poly', 'rbf']},
]

my_config = {
    'name': 'My Exp',
    # 'parameters': parameters,
    'metrics': [{'name': 'always scores -1'}],
    'budget': 5
}

opt = SigOptSearchCV(
    estimator = SVC(), # swap for an sklearn.base.Estimator

    search_spaces = {   ### BAYES SEARCH CV PARAM CONFIG ###
        'C': (1e-6, 1e+6, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8),  # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    },

    # param_grid = {    ### GRID SEARCH CV PARAM CONFIG ### NOT YET IMPLEMENTED
    #     'C': [1e-6, 1e+6],
    #     'gamma': [1e-6, 1e+1],
    #     'degree': [1, 8],  # integer valued parameter
    #     'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    # },

    ### STRING 
    scoring='accuracy', 

    ### LIST OR TUPLE OF STRINGS + refit --> Doesnt work for multiclass bc of sklearn, should generate same error msg
    # TODO: Binary class example for testing this option
    # scoring = [make_scorer(accuracy_score), make_scorer(f1_score)], refit='accuracy',
    
    ### CALLABLE + refit
    # scoring = f, refit='score',

    ### DICTIONARY | key (name): value (scorer) + refit
    # scoring={ 
    #     'f1': make_scorer(f1_score, average='weighted'),
    #     'f2': make_scorer(fbeta_score, beta=2, average="weighted"), 
    #     'acc score': make_scorer(accuracy_score),
    #     'always scores -1': f
    # }, refit = 'f1', 
    # BayesSearch CV convention is to select 'score' metric as optimization target

    ### COMMON ### 
    cv=3,

    ### SIGOPT and BAYES SEARCH ### 
    n_iter=5, # if budget in exp_config, this is overridden

    ### SIGOPT SEARCH ### 
    project_id = 'random',
    experiment_config = my_config
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
# print("test score: %s" % opt.score(X_test, y_test))