import pandas as pd
from collections import Counter
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

random_seed=12345

data_train = pd.read_csv("data/data_train.csv")
data_test = pd.read_csv("data/data_test.csv")
print(data_train.shape, Counter(data_train.fraud), '\n', data_test.shape, Counter(data_test.fraud))

def feature_contribution_analysis(train_set, test_set):
    x_train = train_set.drop('fraud', axis=1)
    y_train = train_set['fraud']
    x_test = test_set.drop('fraud', axis=1)
    y_test = test_set['fraud']
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # # LightGBM
    LGBM = LGBMClassifier(random_state=random_seed)
    LGBM.fit(x_train, y_train)
    y_pred = LGBM.predict(x_test)
    score_data =[]
    scoring = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
    for sco in scoring:
        score = sco(y_test, y_pred)
        score_data.append(score)
    print(score_data)

# Experiment A
col_A=['About', 'Bounty', 'Category_sum', 'Premium', 'Accepting', 'Platform', 'KYC_num', 'fraud']
feature_contribution_analysis(data_train[col_A], data_test[col_A])
print('above is the result of Experiment A')

# Experiment B
col_B=['About', 'Bounty', 'Category_sum', 'Premium', 'Accepting', 'Platform', 'KYC_num',
       'Whitepaper', 'fraud']
feature_contribution_analysis(data_train[col_B], data_test[col_B])
print('above is the result of Experiment B')

# Experiment C
col_C=['About', 'Bounty', 'Category_sum', 'Premium', 'Accepting', 'Platform', 'KYC_num',
       'Whitepaper', 'github', 'fraud']
feature_contribution_analysis(data_train[col_C], data_test[col_C])
print('above is the result of Experiment C')

# Experiment D
col_D=['About', 'Bounty', 'Category_sum', 'Premium', 'Accepting', 'Platform', 'KYC_num',
       'Whitepaper', 'github', 'Registration', 'fraud']
feature_contribution_analysis(data_train[col_D], data_test[col_D])
print('above is the result of Experiment D')

# Experiment E
col_E=['About', 'Bounty', 'Category_sum', 'Premium', 'Accepting', 'Platform',  'KYC_num',
       'Whitepaper', 'github', 'Registration', 'facebook', 'telegram', 'youtube', 'instagram',
       'linkedin', 'medium', 'reddit', 'bitcointalk', 'fraud']
feature_contribution_analysis(data_train[col_E], data_test[col_E])
print('above is the result of Experiment E')

# Experiment F
col_F=['About', 'Bounty', 'Category_sum', 'Premium', 'Accepting', 'Platform', 'KYC_num',
       'Whitepaper', 'github', 'Registration', 'facebook', 'telegram', 'youtube', 'instagram',
       'linkedin', 'medium', 'reddit', 'bitcointalk', 'rating_count', 'rating_overall', 'rating_ai',
       'fraud']
feature_contribution_analysis(data_train[col_F], data_test[col_F])
print('above is the result of Experiment F')

# Experiment G
col_G=['About', 'Bounty', 'Category_sum', 'Premium', 'Accepting', 'Platform', 'KYC_num',
       'Whitepaper', 'github', 'Registration', 'facebook', 'telegram', 'youtube', 'instagram',
       'linkedin', 'medium', 'reddit', 'bitcointalk', 'rating_count', 'rating_overall', 'rating_ai',
       'KYC', 'fraud']
feature_contribution_analysis(data_train[col_G], data_test[col_G])
print('above is the result of Experiment G')