import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from imbalanced_ensemble.utils._plot import plot_2Dprojection_and_cardinality

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data= pd.read_csv("data/data_1.csv")
print(data.shape)

index=data.rating_overall
print(index.value_counts())

data['rating_overall']=data['rating_overall'].apply(lambda x:
                                                    float(0) if x == 'NoEntry'
                                                    else x)

# Converts partial feature types
columns=['rating_overall', 'KYC_num']
for k in columns:
    data[k]=data[k].apply(lambda x: float(x))
print('The converted data type：', '\n', data.dtypes)

data_num=data[['rating_count', 'rating_overall', 'rating_ai']]
print(data_num.shape)
print(data_num.describe().T)

print(Counter(data['fraud']))

print(data.isnull().sum())

na_ratio=data.isnull().sum()[data.isnull().sum()>=0].sort_values(ascending=False)/len(data)
na_sum=data.isnull().sum().sort_values(ascending=False)
print(na_ratio)

# 填充缺失值
data['rating_overall']=data['rating_overall'].fillna(data['rating_overall'].astype(float).mean())
data['rating_ai']=data['rating_ai'].fillna(data['rating_ai'].astype(float).mean())
data['rating_count']=data['rating_count'].fillna(data['rating_count'].astype(float).mean())
data['Category_sum']=data['Category_sum'].fillna(data['Category_sum'].astype(float).mean())

print('After filling in the missing values：', '\n', data.isnull().sum())

mm=MinMaxScaler()
Data=pd.DataFrame(mm.fit_transform(data))
Data.columns=data.columns
print(Data.head(5))

# 部分特征的分布
# Distribution of some important features
str_col= ['KYC', 'Registration']
print(str_col)

dist_cols = 2
dist_rows = len(str_col)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

i = 1
plt.rcParams.update({'font.size': 15})
for col in str_col:
    ax=plt.subplot(1, 2, i)
    ax=sns.countplot(x=Data[col], hue="fraud", data=Data, palette=['limegreen', 'indianred'])
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 17})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 17})
    # ax.set_title(col, fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper right')
    i += 1
    plt.tight_layout();
plt.show()

# Distribution of some important features
str_col= ['KYC', 'github']
print(str_col)

dist_cols = 2
dist_rows = len(str_col)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

i = 1
plt.rcParams.update({'font.size': 15})
for col in str_col:
    ax=plt.subplot(1, 2, i)
    ax=sns.countplot(x=Data[col], hue="fraud", data=Data, palette=['limegreen', 'indianred'])
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 17})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 17})
    # ax.set_title(col, fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper right')
    i += 1
    plt.tight_layout();
plt.show()

# Numerical features distribution
nem_col=['rating_overall', 'rating_ai']
print(nem_col)

dist_cols = 2
dist_rows = len(nem_col)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

i = 1
plt.rcParams.update({'font.size': 15})
for col in nem_col:
    ax = plt.subplot(1, 2, i)
    ax = sns.kdeplot(data=Data[Data.fraud==0][col], bw=0.5, label="Not fraud", color="limegreen", shade=True)
    ax = sns.kdeplot(data=Data[Data.fraud==1][col], bw=0.5, label="Fraud", color="indianred", shade=True)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 17})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 17})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    i += 1
plt.show()

# 划分特征和标签
x=data.drop('fraud', axis=1)
y=data.fraud
print(x.shape, y.shape)

# standardized
mm=MinMaxScaler()
X=pd.DataFrame(mm.fit_transform(x))
X.columns=x.columns
print(X.head(5))

# 2维投影
plot_2Dprojection_and_cardinality(X, y)
# plt.tick_params(labelsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
plt.legend(loc='upper right')
plt.show()

data.to_csv(path_or_buf=r'data/data_2.csv', index=None)