import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
import shap
from collections import Counter
from shap.plots import _waterfall
from IPython.display import (display, display_html, display_png, display_svg)

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Load the data
data= pd.read_csv("data/data_train.csv")
print(data.shape, '\n', data.head(5))

y_train=data['fraud']
x_train=data.drop('fraud', axis=1)
print(Counter(y_train))

random_seed=12345

# Load the data
data_test=pd.read_csv('data/data_test.csv')
print(data_test.shape)

x_test=data_test.drop(['fraud'], axis=1)
y_test=data_test['fraud']
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


random_seed=12345
# LGBM
LGBM=LGBMClassifier(learning_rate=0.5,
                    n_estimators=40,
                    max_depth=14,
                    num_leaves=31,
                    min_child_samples=20,
                    min_child_weight=0.001,
                    colsample_bytree=1,
                    random_state=random_seed
                    )

LGBM.fit(x_train, y_train)
y_pred_LGBM=LGBM.predict(x_test)
y_proba_LGBM=LGBM.predict_proba(x_test)
cm_LGBM=confusion_matrix(y_test, y_pred_LGBM)
print(cm_LGBM)

explainer = shap.TreeExplainer(LGBM)
shap_value = explainer.shap_values(x_train)
#
# SHAP summary plot
# fig = plt.subplots(figsize=(6,4),dpi=400)   plot_type="dot",
ax=shap.summary_plot(shap_value[1], x_train,max_display=20)

# SHAP dependence plot
shap.dependence_plot("KYC", shap_value[1], x_train, interaction_index="Registration")
shap.dependence_plot("rating_overall", shap_value[1], x_train, interaction_index="KYC")
shap.dependence_plot("rating_ai", shap_value[1], x_train, interaction_index="github")

# SHAP force/waterfall/decision plot
# non-fraudent
shap.initjs()
shap.force_plot(explainer.expected_value[1],
                shap_value[1][6],
                x_train.iloc[6],
                text_rotation=20,
                matplotlib=True)

shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1],
                                       shap_value[1][6],
                                       feature_names = x_train.columns,
                                       max_display = 19)

shap.decision_plot(explainer.expected_value[1],
                   shap_value[1][6],
                   x_train.iloc[6])

# fraudent
shap.initjs()
shap.force_plot(explainer.expected_value[1],
                shap_value[1][13],
                x_train.iloc[13],
                text_rotation=20,
                matplotlib=True)

shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1],
                                       shap_value[1][13],
                                       feature_names = x_train.columns,
                                       max_display = 19)

shap.decision_plot(explainer.expected_value[1],
                   shap_value[1][13],
                   x_train.iloc[13])