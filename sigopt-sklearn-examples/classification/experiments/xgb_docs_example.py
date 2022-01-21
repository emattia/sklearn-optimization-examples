import sigopt.sklearn 
from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier 
data=load_wine() 
X,y=data.data,data.target 
train_X,test_X,train_y,test_y = train_test_split(X,y) 
experiment = sigopt.sklearn.experiment( 
  estimator= XGBClassifier(), 
  train_X = train_X,  
  train_y = train_y,  
  validation_sets = [(test_X, test_y, "test")],  
  fixed_params = {'num_boost_rounds': 100, 'learning_rate': .1}, 
  experiment_design = { 
    'name': 'XGBClassifier', 'budget': 2,
    'parameters': [{'name': 'max_depth', 'type': 'int', 'bounds':{'min': 3, 'max':8}}] 
  }
)
print(f'See results at https://app.sigopt.com/experiment/{experiment.id}') 