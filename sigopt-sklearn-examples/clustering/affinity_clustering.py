import sigopt.sklearn
from sklearn.cluster import AffinityPropagation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (homogeneity_score, 
                             completeness_score, 
                             v_measure_score, 
                             adjusted_rand_score, 
                             adjusted_mutual_info_score,
                             make_scorer)
from sklearn.datasets import make_blobs
centers = [[1, 1], [-1, -1], [1, -1]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=.2)
metrics = {
  "homogeneity_score": make_scorer(homogeneity_score), 
  "completeness_score": make_scorer(completeness_score), 
  "v_measure_score": make_scorer(v_measure_score), 
  "adjusted_rand_score": make_scorer(adjusted_rand_score), 
  "adjusted_mutual_info_score": make_scorer(adjusted_mutual_info_score)
}
sigopt.set_project('random')

run_context = sigopt.sklearn.run(
  AffinityPropagation(), 
  train_X, 
  train_y, 
  validation_sets=[(test_X, test_y, "test")], 
  params=dict(preference=-50, random_state=0),
  scoring=metrics,
  run_options={"name": "Supervised Clustering, Custom Metrics", "autolog_metrics":False}
)
run_context.run.end()

run_context = sigopt.sklearn.run(
  AffinityPropagation(), 
  train_X, 
  params=dict(preference=-50, random_state=0),
  run_options={"name": "Unsupervised Clustering"}
)
run_context.run.end()