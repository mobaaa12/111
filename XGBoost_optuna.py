from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance,XGBClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
import numpy as np
import optuna
from optuna.samplers import TPESampler
os.chdir("c:/Users/username/Desktop/Nurse Care Activity recognization/features and labels")

# name each csv
operations = ["N01T1", "N01T2", "N02T1", "N02T2", "N04T1", "N04T2",
          "N06T1", "N06T2", "N07T1", "N07T2", "N11T1", "N11T2",
          "N12T1", "S01T1", "S01T2", "S02T1", "S02T2", "S03T1", "S03T2",
          "S05T1", "S05T2", "S07T1", "S07T2", "S08T1", "S08T2",
          "S09T1", "S09T2", "S10T1", "S10T2", "S11T1", "S11T2"]

group = []
count = 1

def combine_data(operations):
    all_feature = pd.concat([pd.read_csv(f'Every second with time/{operation}_features_and_labels.csv').iloc[:, :-1]
                         for operation in operations], ignore_index=True)
    all_label = pd.concat([pd.read_csv(f'Every second with time/{operation}_features_and_labels.csv').iloc[:, -1]
                       for operation in operations], ignore_index=True)
    return all_feature,all_label


subjects = np.unique([operation[:5] for operation in operations])  
results = []  


all_feature, all_label = combine_data(operations)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(all_feature, all_label, test_size=0.2, random_state=42)

sampler = TPESampler(seed=10) 

def objective(trial): 
    param = {
        'objective': 'multi:softmax',
        'num_class': 9,
        'verbosity':3,
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'max_depth':trial.suggest_int('max_depth', 3, 9),
    }
    xgb = XGBClassifier(**param)     
    gbm = xgb.fit(X_train,y_train)
    return accuracy_score(y_test, np.round(gbm.predict(X_test)))

study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=10)  

study.best_params
print(f"The best hyperparameters are{study.best_params}")
