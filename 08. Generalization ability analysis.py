import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB


from sklearn.model_selection import KFold
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import scikitplot as skplt

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

# Load the data
data_train = pd.read_csv("data/data_train.csv")
print(data_train.shape, '\n', data_train.head(5))

y_train=data_train['fraud']
x_train=data_train.drop('fraud', axis=1)
print(x_train.shape, y_train.shape)
print(Counter(y_train))

data_test=pd.read_csv('data/data_test.csv')
print(data_test.shape)
print(data_test.head(5), data_test.shape)
x_test=data_test.drop(['fraud'], axis=1)
y_test=data_test['fraud']

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


random_seed=12345

# 准备比较的模型
# LR
LR=LogisticRegression(random_state=random_seed)
LR.fit(x_train, y_train)
y_pred_LR=LR.predict(x_test)
y_proba_LR=LR.predict_proba(x_test)
cm_LR=confusion_matrix(y_test, y_pred_LR)

# SVM
SVM=SVC(probability=True, random_state=random_seed)
SVM.fit(x_train, y_train)
y_pred_SVM=SVM.predict(x_test)
y_proba_SVM=SVM.predict_proba(x_test)
cm_SVM=confusion_matrix(y_test, y_pred_SVM)

# MLP
MLP=MLPClassifier(random_state=random_seed)
MLP.fit(x_train, y_train)
y_pred_MLP=MLP.predict(x_test)
y_proba_MLP=MLP.predict_proba(x_test)
cm_MLP=confusion_matrix(y_test, y_pred_MLP)

# SGD
SGD=SGDClassifier(loss="log", random_state=random_seed)
SGD.fit(x_train, y_train)
y_pred_SGD=SGD.predict(x_test)
y_proba_SGD=SGD.predict_proba(x_test)
cm_SGD=confusion_matrix(y_test, y_pred_SGD)

# BNB
BNB=BernoulliNB()
BNB.fit(x_train, y_train)
y_pred_BNB=BNB.predict(x_test)
y_proba_BNB=BNB.predict_proba(x_test)
cm_BNB=confusion_matrix(y_test, y_pred_BNB)

# GNB
GNB=GaussianNB()
GNB.fit(x_train, y_train)
y_pred_GNB=GNB.predict(x_test)
y_proba_GNB=GNB.predict_proba(x_test)
cm_GNB=confusion_matrix(y_test, y_pred_GNB)

# DT
DT=DecisionTreeClassifier(random_state=random_seed)
DT.fit(x_train, y_train)
y_pred_DT=DT.predict(x_test)
y_proba_DT=DT.predict_proba(x_test)
cm_DT=confusion_matrix(y_test, y_pred_DT)

# RF
RF=RandomForestClassifier(random_state=random_seed)
RF.fit(x_train, y_train)
y_pred_RF=RF.predict(x_test)
y_proba_RF=RF.predict_proba(x_test)
cm_RF=confusion_matrix(y_test, y_pred_RF)

# Ada
Ada=AdaBoostClassifier(random_state=random_seed)
Ada.fit(x_train, y_train)
y_pred_Ada=Ada.predict(x_test)
y_proba_Ada=Ada.predict_proba(x_test)
cm_Ada=confusion_matrix(y_test, y_pred_Ada)

# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)
GBDT.fit(x_train, y_train)
y_pred_GBDT=GBDT.predict(x_test)
y_proba_GBDT=GBDT.predict_proba(x_test)
cm_GBDT=confusion_matrix(y_test, y_pred_GBDT)

# KNN
KNN=KNeighborsClassifier()
KNN.fit(x_train, y_train)
y_pred_KNN=KNN.predict(x_test)
y_proba_KNN=KNN.predict_proba(x_test)
cm_KNN=confusion_matrix(y_test, y_pred_KNN)

# XG
XG=XGBClassifier(random_state=random_seed)
XG.fit(x_train, y_train)
y_pred_XG=XG.predict(x_test)
y_proba_XG=XG.predict_proba(x_test)
cm_XG=confusion_matrix(y_test, y_pred_XG)

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

# Verify that the optimal parameters improve the effect
LGBM.fit(x_train, y_train)
y_pred_LGBM=LGBM.predict(x_test)
y_proba_LGBM=LGBM.predict_proba(x_test)
acc_LGBM=accuracy_score(y_test, y_pred_LGBM)
pre_LGBM=precision_score(y_test, y_pred_LGBM)
rec_LGBM=recall_score(y_test, y_pred_LGBM)
f1_LGBM=f1_score(y_test, y_pred_LGBM)
auc_LGBM=roc_auc_score(y_test, y_pred_LGBM)
cm_LGBM=confusion_matrix(y_test, y_pred_LGBM)
print('On the test set:', acc_LGBM, pre_LGBM, rec_LGBM, f1_LGBM, auc_LGBM, '\n', cm_LGBM)

# 比较混淆矩阵
# KNN
skplt.metrics.plot_confusion_matrix(y_test, y_pred_KNN, title=None, cmap='tab20_r', text_fontsize=15)
plt.title('(a) KNN', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

# DT
skplt.metrics.plot_confusion_matrix(y_test, y_pred_DT, title=None, cmap='tab20_r', text_fontsize=15)
plt.title('(b) DT', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

# BNB
skplt.metrics.plot_confusion_matrix(y_test, y_pred_BNB, title=None, cmap='tab20_r', text_fontsize=15)
plt.title('(c) BNB', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

# GBDT
skplt.metrics.plot_confusion_matrix(y_test, y_pred_GBDT, title=None, cmap='tab20_r', text_fontsize=15)
plt.title('(d) GBDT', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

# Ada
skplt.metrics.plot_confusion_matrix(y_test, y_pred_Ada, title=None, cmap='tab20_r', text_fontsize=15)
plt.title('(e) Ada', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

# LGBM
skplt.metrics.plot_confusion_matrix(y_test, y_pred_LGBM, title=None, cmap='tab20_r', text_fontsize=15)
plt.title('(f) FICODM', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()


# ROC curve# GBDT SVM
# SVM
skplt.metrics.plot_roc(y_test, y_proba_SVM, cmap='Set1', text_fontsize=15)
plt.title('(a) SVM', y=-0.2, fontsize=15)
plt.xlabel('False positive rate', fontsize=15)
plt.ylabel('True positive rate', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)

legend_font = {"family" : "Times New Roman"}
plt.legend(fontsize=15, frameon=True, prop=legend_font)
plt.show()

# GBDT
skplt.metrics.plot_roc(y_test, y_proba_GBDT, cmap='Set1', text_fontsize=15)
plt.title('(b) GBDT', y=-0.2, fontsize=15)
plt.xlabel('False positive rate', fontsize=15)
plt.ylabel('True positive rate', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)

legend_font = {"family" : "Times New Roman"}
plt.legend(fontsize=15, frameon=True, prop=legend_font)
plt.show()

# LGBM
skplt.metrics.plot_roc(y_test, y_proba_LGBM, cmap='Set1', text_fontsize=15)
plt.title('(c) FICODM', y=-0.2, fontsize=15)
plt.xlabel('False positive rate', fontsize=15)
plt.ylabel('True positive rate', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)

legend_font = {"family" : "Times New Roman"}
plt.legend(fontsize=15, frameon=True, prop=legend_font)
plt.show()