import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sigopt.sklearn
df =  sns.load_dataset('titanic')
df['is_male'] = df['sex'].map({'male': 1, 'female': 0})
df = df.select_dtypes('number').dropna() 
x = df.drop(columns=['survived'])
y = df['survived']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=1)
model = RandomForestClassifier(random_state=42)
sigopt.set_project('random')
run_context = sigopt.sklearn.run(train_X=train_x, train_y=train_y, estimator=model, validation_sets=[(test_x, test_y, 'test')])
model = run_context.model
# https://towardsdatascience.com/explaining-scikit-learn-models-with-shap-61daff21b12a
explainer = shap.Explainer(model)
shap_test = explainer(test_x)
shap_df = pd.DataFrame(shap_test.values[:,:,1], columns=shap_test.feature_names, index=test_x.index)
shap.plots.bar(shap_test[:,:,1], show=False)
f = plt.gcf()
run_context.run.log_image(f, name='Global Shapley Values for `survived` class')
run_context.run.end()