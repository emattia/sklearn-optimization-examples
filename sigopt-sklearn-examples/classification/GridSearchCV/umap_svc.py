# https://umap-learn.readthedocs.io/en/latest/auto_examples/plot_feature_extraction_classification.html
# #sphx-glr-auto-examples-plot-feature-extraction-classification-py
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline 
from skopt.searchcv import SigOptSearchCV
from umap import UMAP
X,y = make_classification(
  n_samples=1000,
  n_features=300,
  n_informative=250,
  n_redundant=0,
  n_repeated=0,
  n_classes=2,
  random_state=10,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
svc = LinearSVC(dual=False, random_state=37)
umap = UMAP(random_state=12)
pipeline = Pipeline([("umap", umap), ("svc", svc)])
params_grid_pipeline = {
  "umap__n_neighbors": [5, 10],
  "umap__n_components": [5, 10],
  "svc__C": [10 ** k for k in range(-2, 1)],
}
clf_pipeline = GridSearchCV(pipeline, params_grid_pipeline, cv=2)
clf_pipeline.fit(X_train, y_train)
print("Test accuracy: {clf_pipeline.score(X_test, y_test)}")
