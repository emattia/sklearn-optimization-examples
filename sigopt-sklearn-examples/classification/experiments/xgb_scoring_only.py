from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from xgboost import XGBClassifier
import sigopt.sklearn
data=load_wine()
X,y=data.data,data.target
train_x,test_x,train_y,test_y = train_test_split(X,y)
sigopt.set_project('random')
experiment_config = {
  'parameters': [{'name': 'learning_rate', 'type': 'double', 'bounds': {'min':0.1, 'max':.5}}, 
                 {'name': 'max_depth'    , 'type': 'int'   , 'bounds': {'min':3  , 'max':12}}],
  # 'metrics': [{'name': 'test-accuracy', 'strategy': 'optimize', 'objective': 'maximize'}],
  'budget': 2
}
experiment = sigopt.sklearn.experiment(
  train_x, 
  train_y, 
  XGBClassifier(use_label_encoder=False), 
  validation_sets=[(test_x, test_y, "test")], 
  scoring={'accuracy': make_scorer(accuracy_score)},
  experiment_config=experiment_config
)