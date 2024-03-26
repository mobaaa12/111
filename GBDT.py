import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.ensemble import GradientBoostingClassifier




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



subjects = np.unique([operation[:5] for operation in operations])  
ac = []  
pr = []
rc = []
f1 = []
all_y_true = []
all_y_pred = []

best_params = {'learning_rate': 0.059203925140895175, 'n_estimators': 355, 'max_depth': 10, 'min_samples_leaf': 41, 'min_samples_split': 194, 
               'subsample': 0.8684482051282947, 'random_state': 60, 'max_features': 10}
model = GradientBoostingClassifier(**best_params)

# LOSO-CV start
for test_subject in subjects:
    print(f"Testing on subject: {test_subject}")

    train_indices = [op for i, op in enumerate(operations) if not op.startswith(test_subject)]
    print(train_indices)
    test_indices = [op for i, op in enumerate(operations) if op.startswith(test_subject)]
    print(test_indices)

    X_train, y_train = combine_data(train_indices)
    print(len(X_train), len(y_train))
    X_test, y_test = combine_data(test_indices)
    print(len(X_test), len(y_test))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {test_subject}: {accuracy}")
    precision = precision_score(y_test, y_pred,average='weighted')
    print(f"Precision: {precision}")
    recall = recall_score(y_test, y_pred,average='weighted')
    print(f"Recall: {recall}")
    fc = f1_score(y_test, y_pred,average='weighted')
    print(f"F1 Score: {fc}")
    
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    ac.append(accuracy)
    pr.append(precision)
    rc.append(recall)
    f1.append(fc)

average_accuracy = np.mean(ac)
average_precision = np.mean(pr)
average_recall = np.mean(rc)
average_f1 = np.mean(f1)

print(f"LOSO-CV Average accuracy: {average_accuracy}")
print(f"LOSO-CV Average precision: {average_precision}")
print(f"LOSO-CV Average recall: {average_recall}")
print(f"LOSO-CV Average F1 Score: {average_f1}")

cm = confusion_matrix(all_y_true, all_y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True,  cmap="YlGnBu")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
