# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_refit_callable.html
# #sphx-glr-auto-examples-model-selection-plot-grid-search-refit-callable-py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def lower_bound(cv_results):
    best_score_idx = np.argmax(cv_results[f"mean_test_{metric_name}"])
    return (
        cv_results[f"mean_test_{metric_name}"][best_score_idx]
        - cv_results[f"std_test_{metric_name}"][best_score_idx]
    )

def best_low_complexity(cv_results):
    threshold = lower_bound(cv_results)
    candidate_idx = np.flatnonzero(cv_results[f"mean_test_{metric_name}"] >= threshold)
    best_idx = candidate_idx[
        cv_results["param_reduce_dim__n_components"][candidate_idx].argmin()
    ]
    return best_idx

pipe = Pipeline(
    [
        ("reduce_dim", PCA(random_state=42)),
        ("classify", LinearSVC(random_state=42, C=0.01)),
    ]
)

param_grid = {"reduce_dim__n_components": [6, 8, 10, 12, 14]}
metric_name = "accuracy"
grid = GridSearchCV(
    pipe,
    cv=2,
    n_jobs=1,
    param_grid=param_grid,
    scoring=[metric_name],
    refit=best_low_complexity,
)
X, y = load_digits(return_X_y=True)
grid.fit(X, y)
n_components = grid.cv_results_["param_reduce_dim__n_components"]
test_scores = grid.cv_results_[f"mean_test_{metric_name}"]
print(n_components, test_scores)