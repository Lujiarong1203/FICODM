import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import validation_curve
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the data
data_train = pd.read_csv("data/data_train.csv")
data_test = pd.read_csv("data/data_test.csv")

y=data_train['fraud']
x=data_train.drop('fraud', axis=1)
print(x.shape, y.shape)
print(Counter(y))

random_seed=12345
# K-fold
kf=KFold(n_splits=5, shuffle=True, random_state=random_seed)
cnt=1
for train_index, test_index in kf.split(x, y):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

# Parameter tuning/Each time a parameter is tuned, update the parameter corresponding to other_params to the optimal value

# 1-learning_rate
cv_params= {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
model = LGBMClassifier(random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the learning_rate validation_curve
param_range_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='learning_rate',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

print(train_scores_1, '\n', train_mean_1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label="Training score")

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10,label="Cross-validation score")

plt.fill_between(param_range_1,test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(a) learning_rate', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()



# 2-n_estimators
cv_params= {'n_estimators': range(5, 60, 5)}
model = LGBMClassifier(learning_rate=0.5, random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the n_estimators validation_curve
param_range_1=range(5, 60, 5)
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='n_estimators',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

print(train_scores_1, '\n', train_mean_1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='Training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10,label='Cross-validation score')

plt.fill_between(param_range_1,test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(b) n_estimators', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()



# 3-max_depth
cv_params= {'max_depth': range(10, 20, 1)}
model = LGBMClassifier(learning_rate=0.5, n_estimators=40, random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the max_depth validation curve
param_range_1=range(10, 20, 1)
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='max_depth',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='Training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='Cross-validation score')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(c) max_depth', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()



# 4-num_leaves
cv_params= {'num_leaves': range(26, 36, 1)}
model = LGBMClassifier(learning_rate=0.5, n_estimators=40, max_depth=14, random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the max_depth validation curve
param_range_1=range(26, 36, 1)
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='num_leaves',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='Training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='Cross-validation score')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(d) num_leaves', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()



# 5-min_child_sample
cv_params= {'min_child_samples': range(15, 25, 1)}
model = LGBMClassifier(learning_rate=0.5, n_estimators=40, max_depth=14, num_leaves=31, random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the max_depth validation curve
param_range_1=range(15, 25, 1)
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='min_child_samples',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='Training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='Cross-validation score')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(e) min_child_sample', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()


# 6-min_child_weight
cv_params= {'min_child_weight': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]}
model = LGBMClassifier(learning_rate=0.5, n_estimators=40, max_depth=14, num_leaves=31, min_child_samples=20, random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the max_depth validation curve
param_range_1 = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='min_child_weight',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='Training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='Cross-validation score')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(f) min_child_weight', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()



# 7-colsample_bytree
cv_params= {'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
model = LGBMClassifier(learning_rate=0.5, n_estimators=40, max_depth=14, num_leaves=31, min_child_samples=20, min_child_weight=0.001, random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the max_depth validation curve
param_range_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='colsample_bytree',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='Training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='Cross-validation score')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(f) colsample_bytree', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()



# 最佳参数：model = LGBMClassifier(learning_rate=0.5, n_estimators=40, max_depth=14, num_leaves=31, min_child_samples=20, min_child_weight=0.001, colsample_bytree=1, random_state=random_seed)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def Hyper_comparison(train_set, test_set, estimator):
    x_train = train_set.drop('fraud', axis=1)
    y_train = train_set['fraud']
    x_test = test_set.drop('fraud', axis=1)
    y_test = test_set['fraud']
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # # estimator
    est = estimator
    est.fit(x_train, y_train)
    y_pred = est.predict(x_test)
    score_data =[]
    scoring = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
    for sco in scoring:
        score = sco(y_test, y_pred)
        score_data.append(score)
    print(score_data)

LGBM_1 = LGBMClassifier(random_state=random_seed)
LGBM_2 = LGBMClassifier(learning_rate=0.1,
                        n_estimators=40,
                        max_depth=14,
                        num_leaves=31,
                        min_child_samples=20,
                        min_child_weight=0.001,
                        colsample_bytree=1,
                        random_state=random_seed)

Hyper_comparison(data_train, data_test, LGBM_1)
Hyper_comparison(data_train, data_test, LGBM_2)
