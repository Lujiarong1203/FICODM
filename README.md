# FICODM
Code repository for paper &lt; FICODM: An interpretable model for detecting fraudulent Initial Coin Offering (ICO) based on multi-source heterogeneous fusion data >
# IDE
pycharm and jupternotebook, Compiler Environment: python 3.9
# Primary dependency libs or packages:
python3.9
numpy 1.21.5
pandas 1.4.4
seaborn 0.11.2
matplotlib 3.5.2
scikit-learn 1.0.2
LightGBM 3.3.4
scikitplot 0.3.7
shap 0.41.0
# data
The dataset includes the raw dataset (dataset_fraud_14.11_2020), the initial ICO fusion dataset constructed after data type conversion and feature filtering (data_1), the dataset after missing value filling (data_2 is), the dataset after feature selection (data_3), the training set after category imbalance processing ( data_train, 70%), and test set (data_test, 30%).
# code
Code 1-4: Data pre-processing codes.
Code 5-6: Codes for feature contribution analysis, model building, and model comparison.
Code 7-9: codes for hyperparameter tuning, learning ability and generalization ability analysis of the model.
Code 10: Model interpretability analysis
