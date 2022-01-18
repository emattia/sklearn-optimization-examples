from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import shap # v0.39.0
shap.initjs()
diabetes = load_diabetes(as_frame=True)
X = diabetes['data'].iloc[:, :4] # Select first 4 columns
y = diabetes['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
explainer = shap.Explainer(model)
shap_test = explainer(X_test)
# shap_df = pd.DataFrame(shap_test.values, columns=shap_test.feature_names, index=X_test.index)
shap.summary_plot(shap_test, cmap=plt.get_cmap("winter_r"))
shap.plots.heatmap(shap_test, cmap=plt.get_cmap("winter_r"))
shap.plots.bar(shap_test[0])
class WaterfallData():
    def __init__ (self, shap_test, index):
        self.values = shap_test[index].values
        self.base_values = shap_test[index].base_values[0]
        self.data = shap_test[index].data
        self.feature_names = shap_test.feature_names
shap.plots.waterfall(WaterfallData(shap_test, 0))