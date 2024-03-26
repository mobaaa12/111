import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import os
import optuna
from optuna.samplers import TPESampler

os.chdir("c:/Users/86152/Desktop/Nurse Care Activity recognization/features and labels")

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


# 定义ID列表
subjects = np.unique([operation[:5] for operation in operations])  
results = []  

all_feature, all_label = combine_data(operations)


# Define the objective function for Optuna optimization
def objective(trial, X_train, y_train, X_test, y_test):
    # Define parameters to be optimized for the LGBMClassifier
    param = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "num_class": 9,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 300, 600),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.005, 0.015),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.02, 0.06),
        "max_depth": trial.suggest_int("max_depth", 6,14),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
    }

    # Create an instance of LGBMClassifier with the suggested parameters
    lgbm_classifier = LGBMClassifier(**param)
    
    # Fit the classifier on the training data
    lgbm_classifier.fit(X_train, y_train)

    # Evaluate the classifier on the test data
    score = lgbm_classifier.score(X_test, y_test)

    return score

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(all_feature, all_label, test_size=0.2, random_state=42)

# Set up the sampler for Optuna optimization
sampler = optuna.samplers.TPESampler(seed=42)  # Using Tree-structured Parzen Estimator sampler for optimization

# Create a study object for Optuna optimization
study = optuna.create_study(direction="maximize", sampler=sampler)

# Run the optimization process
study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=25)

# Get the best parameters after optimization
best_params = study.best_params

print('='*50)
print(best_params)
