import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster._unsupervised import silhouette_score
digits = datasets.load_digits()
images = digits.images
X = np.reshape(images, (len(images), -1))
model = DBSCAN()
model.fit_predict(X)

