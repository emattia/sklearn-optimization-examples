import numpy as np
from numpy.lib.function_base import average
np.random.seed(123)
import matplotlib.pyplot as plt

from skopt.searchcv import BayesSearchCV
from sigopt.sklearn import SigOptSearchCV

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.svm import SVC

from sklearn.metrics import make_scorer, accuracy_score, fbeta_score, f1_score
import sigopt

### Set SigOpt project ### 
# sigopt.set_project('random')

### Load data ### 
X, y = load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)


def f(estimator, X, y):
  '''flexible function that returns metric score(s) for a model training/fitting run.
     https://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object
  '''
  return -1 # can also return names and multiple metrics like {'always scores -1': -1, ...}
  # note, how does SigOptSearchCV handle custom multi-metric, how does this interact with exp config? 

 ### SigOpt param space for Support Vector Machine ### 
parameters = [   
  {'name': 'C', 'type': 'double', 'bounds': {'min': 1e-6, 'max':1e6}},
  {'name': 'gamma', 'type': 'double', 'bounds': {'min': 1e-6, 'max':1e1}},
  {'name': 'degree', 'type': 'int', 'bounds': {'min': 1, 'max':8}},
  {'name': 'kernel', 'type': 'categorical', 'categorical_values': ['linear', 'poly', 'rbf']},
]

sigopt_experiment_config = {    # you don't need to specify any of these, 
    'name': 'My Exp',           # although you can and 
    # 'parameters': parameters, # the SigOptSearchCV object should take care of necessary conversions.  
    # 'metrics': [{'name': 'always scores -1'}], # maybe you want to optimize the custom metric f defined above
    # 'budget': 3              
}

sigopt.set_project('random')

opt = SigOptSearchCV(
    estimator = SVC(), # sklearn.base.Estimator

    ### BayesSearchCV param config ###
    search_spaces = {   
        'C': (1e-6, 1e+6, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8),  # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    },

    ### GridSearchCV param config ###
    # param_grid = {    
    #     'C': [1e-6, 1e+6],
    #     'gamma': [1e-6, 1e+1], # real valued parameter dimension
    #     'degree': [1, 8],  # integer valued parameter dimension
    #     'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter dimension
    # },

    ### scoring mode 1: string ### 
    # scoring='accuracy', 

    ### scoring mode 2: list or tuple of strings + refit
      # Right now, this example oesnt work for multiclass because of sklearn, 
      # should generate same error msg, and do binary class
    # scoring = [make_scorer(accuracy_score), make_scorer(f1_score)], refit='accuracy',
    
    ### scoring mode 3: callabale ### 
		  # turn on `refit` arg when using custom function
    # scoring = make_scorer(f, greater_is_better=True), refit='score',

    ### scoring mode 4: dictionary like {'name': sklearn.metrics._scorer._PredictScorer)
        # turn on `refit` arg when using dictionary
    scoring={ 
        'f1': make_scorer(f1_score, average='weighted'),
        'f2': make_scorer(fbeta_score, beta=2, average="weighted"), 
        'my accuracy': make_scorer(accuracy_score),
        # 'always scores -1': make_scorer(f, greater_is_better=True)
    }, refit = 'f1', 
    # if no metric space is specified, `refit` value is used as optimization target.

    cv=3,
    n_iter=2,     # if budget in exp_config, this is overridden
    # n_points=2, # TODO: write about what happens here

    ### only in SigOptSearchCV ### 
    # project_id = 'random',
    # experiment_config = sigopt_experiment_config
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
# print("test score: %s" % opt.score(X_test, y_test))