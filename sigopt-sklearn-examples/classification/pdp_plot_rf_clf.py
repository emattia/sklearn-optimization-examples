from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import sigopt.sklearn
data = load_breast_cancer()
x,y = data.data, data.target
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=1)
model = RandomForestClassifier(random_state=42)
sigopt.set_project('random')
run_context = sigopt.sklearn.run(train_X=train_x, train_y=train_y, estimator=model, validation_sets=[(test_x, test_y, 'test')])
model = run_context.model
# https://scikit-learn.org/stable/modules/partial_dependence.html#partial-dependence-plots
class_idx = 0; feat_idx = 0
PartialDependenceDisplay.from_estimator(model, test_x, features = [feat_idx], target=class_idx)
f = plt.gcf()
run_context.run.log_image(f, name=f'{data.feature_names[feat_idx]} Partial Dependence for {class_idx} class')
