import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import sigopt.sklearn
import warnings
warnings.filterwarnings("ignore")

def load_and_process_data(filepath='.data/BitcoinHeistData.csv'):
  data = pandas.read_csv(filepath)
  X = data.drop(labels=['label'], axis=1)
  fit_address_encoder = LabelEncoder().fit(X['address'])
  X['address'] = fit_address_encoder.transform(X['address'])
  X = X.values
  Y = data.loc[:, 'label'].values
  class_names = ['montrealAPT', 'montrealCryptXXX', 'montrealCryptoLocker', 'montrealCryptoTorLocker2015', 
                 'montrealDMALocker', 'montrealDMALockerv3', 'montrealFlyper', 'montrealGlobe', 'montrealGlobeImposter', 
                 'montrealGlobev3', 'montrealNoobCrypt', 'montrealSamSam', 'montrealVenusLocker', 'montrealWannaCry', 
                 'montrealXLockerv5.0', 'paduaCryptoWall', 'paduaKeRanger', 'princetonCerber', 'princetonLocky', 'white']
  metric_names = []
  for n in class_names:
    metric_names.append(f"{n} precision")
    metric_names.append(f"{n} recall")
  return train_test_split(X, Y, test_size=0.1, random_state=104), (class_names, metric_names)

def my_scoring(est, X, y_test):
  clf_report = classification_report(y_test, est.predict(X), zero_division=1, output_dict=True)
  out = {}
  for class_name in CLASS_NAMES:
    out[f"{class_name}-precision"] = clf_report[class_name]['precision']
    out[f"{class_name}-recall"] = clf_report[class_name]['recall']
  return out

# https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset
(train_x,test_x,train_y,test_y), (class_names, metric_names) = load_and_process_data(filepath='./data/BitcoinHeistData.csv')
CLASS_NAMES = class_names
experiment_config = {'parameters': [
    dict(name="n_estimators", type="int", bounds=dict(min=1, max=20)),
    dict(name="max_depth", type="int", bounds=dict(min=1, max=20)),
    dict(name="learning_rate", type="double", bounds=dict(min=0.05, max=0.5), transformation="log")
]}
sigopt.sklearn.experiment(train_x, train_y, XGBClassifier(), validation_sets=[(test_x, test_y, 'test')], scoring=my_scoring)