# Developed by Keshav, Nasir, Shu, December 2021
# All rights reserved
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

CSV_PATH = 'dataset/labelled_dataset_2.csv'
x, y = [], []  # Features (x) and labels (y)


def extract_csv(path):
    global x, y
    df = pd.read_csv(path)
    x0 = df.iloc[:, 0]  # Station ID
    x1 = df.iloc[:, 1]  # 1st feature: Total number of stands
    x2 = df.iloc[:, 2]  # 2nd feature: Weekday (Int [1, ..., 7])
    x3 = df.iloc[:, 3]  # 3rd feature: Minute of the point (Int [0, ..., 24 * 60 - 5 = 1435])
    x4 = df.iloc[:, 4]  # 4th feature: Available stands
    x5 = df.iloc[:, 5]  # 5th feature: Available bikes
    x6 = df.iloc[:, 6]  # 6th feature: Ratio Bike / Stand

    # x = np.column_stack((x1, x2, x3, x4, x5, x6))  # Combine features into one matrix --> x
    x = np.column_stack((x2, x3, x5))
    y = df.iloc[:, 7]  # Third column as coorespondent labels
    # y = df.iloc[:, 5]


def separate_data_by_label(feature, label):
    # This function (method) is to separate data into 2 groups: x1* and x2*, and returns 4 lists
    # Each group contains two features: e.g. x1* contains x11 and x12
    # x1* contains data labelled as +1 while x2* contains data labelled as -1
    __coloum_num = 0
    x11, x12, x21, x22, x31, x32 = [], [], [], [], [], []
    for day, x1, _, x2, _ in feature:
        if label[__coloum_num] == 1:      # Sort out labelled +1 features
            x11.append(x1)
            x12.append(x2)
        elif label[__coloum_num] == -1:   # Sort out labelled +2 features
            x21.append(x1)
            x22.append(x2)
        else:
            x31.append(x1)
            x32.append(x2)
        __coloum_num += 1
    return x11, x12, x21, x22, x31, x32


def data_visualization(features, labels):
    print(features.shape, type(features))

    x11, x12, x21, x22, x31, x32 = separate_data_by_label(features, labels)

    # Plotting: used multiple times
    fig1 = plt.figure('Data Visualization', figsize=(16, 10))
    plt.title('Data Visualization', fontsize=22)
    plt.rcParams.update({'font.size': 14})
    # plt.scatter(x11, x12, marker='o', c='#4363D8',
    #             label="Labelled as 1", edgecolor="white", s=60, alpha=0.5)  # Scatter labelled +1
    # plt.scatter(x21, x22, marker='o', c='#E6194B',
    #             label="Labelled as 2", edgecolor="white", s=60, alpha=0.5)  # Scatter labelled -1
    # plt.scatter(x31, x32, marker='o', c='#000000',
    #             label="Labelled as 0", edgecolor="white", s=60, alpha=0.5)  # Scatter labelled 0

    # Linear plot:
    __x = np.array(x[1154:3168, 1] + np.subtract(x[1154:3168, 0], 1) * 1440)
    __y = x[1154:3168, 3]
    __data = np.column_stack((__x, __y))

    plt.plot(__data[:, 0], __data[:, 1])
    plt.xlabel(r'First Feature $x_1$', fontsize=18)
    plt.ylabel(r'Second Feature $x_2$', fontsize=18)
    plt.legend(loc='upper right')


def dummy_classifier(features, labels):
    from sklearn.dummy import DummyClassifier
    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)
    __dummy_clf = DummyClassifier(strategy='most_frequent').fit(xtrain, ytrain)
    __dummy_pred = __dummy_clf.predict(xtest)

    print("Most Frequent Baseline Model:")
    print("F1 score \n", f1_score(ytest, __dummy_pred, average='weighted'))
    print("Confusion matrix: \n", confusion_matrix(ytest, __dummy_pred))


def knn_classifier_tuning(features, labels):
    from sklearn.neighbors import KNeighborsClassifier

    # Since there's no need to augment the features...
    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)

    # Tuning k value
    mean_error, std_error = [], []
    k_value = np.array(range(1, 11))  # From 1 to 10
    for k in k_value:
        __knn_model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        __scores = cross_val_score(__knn_model, xtrain, ytrain, cv=5, scoring='f1_weighted')
        mean_error.append(np.array(__scores).mean())
        std_error.append(np.array(__scores).std())

    # Evaluation
    fig5 = plt.figure(figsize=(8, 6))
    plt.suptitle("Tuning k-value for training kNN Classifier", fontsize=20)
    plt.errorbar(np.array(range(1, 11)), mean_error, yerr=std_error)
    # plt.ylim(0.7, 1)
    plt.xlabel('Value of K', fontsize=20)
    plt.ylabel('F1 Score', fontsize=20)

    # Confusion Matrices
    __knn_model = KNeighborsClassifier(n_neighbors=1, weights='uniform').fit(xtrain, ytrain)
    __pred = __knn_model.predict(xtest)
    print("\n\n----------------------kNN model with k = 2----------------------\n\n")
    print(classification_report(ytest, __pred))
    print("F1 score: \n", f1_score(ytest, __pred, average='weighted'))
    print("Confusion matrix:\n", confusion_matrix(ytest, __pred))


def decision_tree(features, labels):
    from sklearn import tree
    import graphviz

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.5)

    # Tuning depth value
    mean_error, std_error = [], []
    depth_value = np.array(range(1, 11))  # From 1 to 10

    for dep in depth_value:
        __dt_model = tree.DecisionTreeClassifier(max_depth=dep)
        __scores = cross_val_score(__dt_model, xtrain, ytrain, cv=5, scoring='f1_weighted')
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
    print(f"Total number of dataset: {len(x)}")
    # data_visualization(x, y)

    # Baseline model:
    dummy_classifier(x, y)

    # Play with kNN model:
    knn_classifier_tuning(x, y)

    # Play with decision tree model:
    decision_tree(x, y)

    plt.show()


if __name__ == '__main__':
    main(sys.argv)
