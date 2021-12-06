# Developed by Keshav, Nasir, Shu, December 2021
# All rights reserved
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import cross_val_score

STATION_ID = 33
CSV_PATH = f'dataset/labelled_dataset_{STATION_ID}.csv'
x0, x1, x2, x3, x4, x5, x6, x7, y = [], [], [], [], [], [], [], [], []  # Features (x) and labels (y)


def extract_csv(path):
    global x0, x1, x2, x3, x4, x5, x6, x7, y
    df = pd.read_csv(path)
    x0 = df.iloc[:, 0]  # Station ID
    x1 = df.iloc[:, 1]  # 1st feature: Total number of stands
    x2 = df.iloc[:, 2]  # 2nd feature: Week of the year
    x3 = df.iloc[:, 3]  # 3rd feature: Weekday (Int [1, ..., 7])
    x4 = df.iloc[:, 4]  # 4rd feature: Minute of the point (Int [0, ..., 24 * 60 - 5 = 1435])
    x5 = df.iloc[:, 5]  # 5th feature: Available stands
    x6 = df.iloc[:, 6]  # 6th feature: Available bikes
    x7 = df.iloc[:, 7]  # 7th feature: Ratio Bike / Stand

    y = df.iloc[:, 8]  # Third column as coorespondent labels


def dummy_classifier(features, labels):
    from sklearn.dummy import DummyClassifier

    __dummy_clf = DummyClassifier(strategy='most_frequent').fit(features, labels)
    __dummy_pred = __dummy_clf.predict(features)

    print("Most Frequent Baseline Model:")
    print("F1 score \n", f1_score(labels, __dummy_pred, average='weighted'))
    print("Confusion matrix: \n", confusion_matrix(labels, __dummy_pred))


def knn_classifier_tuning(features, labels):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.calibration import CalibratedClassifierCV

    # Since there's no need to augment the features...
    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)

    # Tuning k value
    mean_error, std_error = [], []
    k_value = np.array(range(1, 21))  # From 1 to 20
    for k in k_value:
        __knn_model = KNeighborsClassifier(n_neighbors=k).fit(xtrain, ytrain)
        __knn_cal_model = CalibratedClassifierCV(__knn_model, method='isotonic', cv=5)
        __scores = cross_val_score(__knn_cal_model, xtrain, ytrain, cv=5, n_jobs=-1, scoring='f1_weighted')
        mean_error.append(np.array(__scores).mean())
        std_error.append(np.array(__scores).std())

    # Evaluation
    fig5 = plt.figure(figsize=(8, 6))
    plt.suptitle("Tuning k-value for training kNN Classifier", fontsize=20)
    plt.errorbar(np.array(range(1, 21)), mean_error, yerr=std_error)
    # plt.ylim(0.7, 1)
    plt.xlabel('Value of K', fontsize=20)
    plt.ylabel('F1 Score', fontsize=20)


def decision_tree(features, labels):
    from sklearn import tree
    from sklearn.calibration import CalibratedClassifierCV

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)

    # Tuning depth value
    mean_error, std_error = [], []
    depth_value = np.array(range(1, 11))  # From 1 to 10

    for dep in depth_value:
        __dt_model = tree.DecisionTreeClassifier(max_depth=dep)
        __dt_cal_model = CalibratedClassifierCV(__dt_model, method='isotonic', cv=5)
        __scores = cross_val_score(__dt_cal_model, xtrain, ytrain, cv=5, n_jobs=-1, scoring='f1_weighted')
        mean_error.append(np.array(__scores).mean())
        std_error.append(np.array(__scores).std())

    # Evaluation
    fig6 = plt.figure(figsize=(8, 6))
    plt.suptitle("Tuning depth for training decision tree Classifier", fontsize=20)
    plt.errorbar(np.array(range(1, 11)), mean_error, yerr=std_error)
    plt.xlabel('Value of depth', fontsize=20)
    plt.ylabel('F1 Score', fontsize=20)


def main(args):
    extract_csv(CSV_PATH)
    print(f"Total number of dataset: {len(y)}")

    # Baseline model:
    x_base = np.column_stack((x3, x4, x5, x7))
    dummy_classifier(x_base, y)

    # Play with kNN model:
    x_knn = np.column_stack((x3, x4, x5, x6))
    knn_classifier_tuning(x_knn, y)

    # Play with decision tree model:
    x_tree = np.column_stack((x3, x4, x7))
    decision_tree(x_tree, y)

    plt.show()


if __name__ == '__main__':
    main(sys.argv)
