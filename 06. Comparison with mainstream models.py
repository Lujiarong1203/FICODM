import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, cross_validate, RandomizedSearchCV
from collections import Counter
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
# from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the data
data_train= pd.read_csv("data/data_train.csv")
data_test = pd.read_csv("data/data_test.csv")

random_seed=12345

def model_comparison(train_set, test_set, estimator):
    x_train = train_set.drop('fraud', axis=1)
    y_train = train_set['fraud']
    x_test = test_set.drop('fraud', axis=1)
    y_test = test_set['fraud']
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # # estimator
    est = estimator
    est.fit(x_train, y_train)
    y_pred = est.predict(x_test)
    score_data = []
    scoring = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
    for sco in scoring:
        score = sco(y_test, y_pred)
        score_data.append(score)
    print(score_data)


# 模型对比
KNN=KNeighborsClassifier()
model_comparison(data_train, data_test, estimator=KNN)
print('KNN')

DT=DecisionTreeClassifier()
model_comparison(data_train, data_test, estimator=DT)
print('DT')

BNB=BernoulliNB()
model_comparison(data_train, data_test, estimator=BNB)
print('BNB')

GNB=GaussianNB()
model_comparison(data_train, data_test, estimator=GNB)
print('GNB')

Ada=AdaBoostClassifier()
model_comparison(data_train, data_test, estimator=Ada)
print('Ada')

GBDT=GradientBoostingClassifier()
model_comparison(data_train, data_test, estimator=GBDT)
print('GBDT')

XG=XGBClassifier()
model_comparison(data_train, data_test, estimator=XG)
print('XG')

FICODM=LGBMClassifier(learning_rate=0.1,
                        n_estimators=40,
                        max_depth=14,
                        num_leaves=31,
                        min_child_samples=20,
                        min_child_weight=0.001,
                        colsample_bytree=1,
                        random_state=random_seed)
model_comparison(data_train, data_test, estimator=FICODM)
print('FICODM')