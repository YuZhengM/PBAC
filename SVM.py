import numpy as np
import pandas as pd
from sklearn import svm, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold


def Get_data(Datafile_name,Clinicalfile_name):
    set_data=pd.read_csv(Datafile_name)
    set_data=set_data.iloc[:,1:]
    set_clinical=pd.read_csv(Clinicalfile_name)
    set_label = np.array(set_clinical.iloc[:,1])
    return set_data,set_label

if __name__ == '__main__':
    set_data, set_label = Get_data('./Immune_data/PatientFile/phs000452_after.csv',
                                   'Immune_data/CelllineFile/phs000452_clinical.csv')
    set_data = preprocessing.scale(set_data)

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    # Initialize lists to store the accuracy, AUC, and average precision scores
    acc_scores = []
    auc_scores = []
    avg_prec_scores = []
    i = 0
    for train_index, test_index in skf.split(set_data, set_label):
        print(f"正在进行第{i + 1}次训练...")
        data_train, data_test = set_data[train_index], set_data[test_index]
        label_train, label_test = set_label[train_index], set_label[test_index]

        # Initialize the SVM classifier
        clf = svm.SVC(probability=True)

        # Define the parameter grid for the SVM
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

        # Initialize the grid search
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')

        # Fit the grid search to the data
        grid_search.fit(data_train, label_train)

        # Find the best parameters
        print(f"Best parameters: {grid_search.best_params_}")

        # Use the trained classifier (with the best parameters) to make predictions on the test data
        predictions = grid_search.predict(data_test)

        # Compute the accuracy and ROC AUC score of the classifier
        acc = accuracy_score(label_test, predictions)
        auc = roc_auc_score(label_test, predictions)

        # Calculate the average precision score
        avg_prec = average_precision_score(label_test, predictions)

        acc_scores.append(acc)
        auc_scores.append(auc)
        avg_prec_scores.append(avg_prec)

        i += 1
        # Calculate the average scores over all folds
    avg_acc = sum(acc_scores) / len(acc_scores)
    avg_auc = sum(auc_scores) / len(auc_scores)
    avg_avg_prec = sum(avg_prec_scores) / len(avg_prec_scores)

    print(f"Average Accuracy over all folds: {avg_acc}")
    print(f"Average ROC AUC Score over all folds: {avg_auc}")
    print(f"Average Average Precision Score over all folds: {avg_avg_prec}")
