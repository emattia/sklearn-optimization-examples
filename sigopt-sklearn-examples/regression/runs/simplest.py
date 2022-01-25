import sigopt.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
data=load_diabetes()
X, y = data.data, data.target
train_x,test_x,train_y,test_y = train_test_split(X,y)
sigopt.set_project('random')
existing_run = sigopt.create_run('My Existing Run')
run_context = sigopt.sklearn.run(
  GradientBoostingRegressor(), 
  train_x, 
  train_y, 
  params={"max_depth": 4, "min_samples_split": 5,"learning_rate": 0.01},
  validation_sets=[(test_x, test_y, "test")],
  run_options={'run': existing_run}
)
run_context.run.end()