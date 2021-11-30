# Developed by Keshav, Nasir, Shu, December 2021
# All rights reserved
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

CSV_PATH = 'dataset/labelled_dataset_2.csv'
x, y = [], []  # Features (x) and labels (y)


def extract_csv(path):
    global x, y
    df = pd.read_csv(path)
    x0 = df.iloc[:, 0]  # Station ID
    x1 = df.iloc[:, 1]  # 1st feature: Total number of stands
    x2 = df.iloc[:, 2]  # 2nd feature: Week of the year
    x3 = df.iloc[:, 3]  # 3rd feature: Weekday (Int [1, ..., 7])
    x4 = df.iloc[:, 4]  # 4rd feature: Minute of the point (Int [0, ..., 24 * 60 - 5 = 1435])
    x5 = df.iloc[:, 5]  # 5th feature: Available stands
    x6 = df.iloc[:, 6]  # 6th feature: Available bikes
    x7 = df.iloc[:, 7]  # 7th feature: Ratio Bike / Stand

    # x = np.column_stack((x1, x2, x3, x4, x5, x6))  # Combine features into one matrix --> x
    x = np.column_stack((x3, x4, x7, x3))
    y = df.iloc[:, 8]  # Third column as coorespondent labels
    # y = df.iloc[:, 5]


def separate_data_by_label(feature, label):
    # This function (method) is to separate data into 2 groups: x1* and x2*, and returns 4 lists
    # Each group contains two features: e.g. x1* contains x11 and x12
    # x1* contains data labelled as +1 while x2* contains data labelled as -1
    __coloum_num = 0
    x11, x12, x21, x22, x31, x32 = [], [], [], [], [], []
    for day, x1, x2, _ in feature:
        if label[__coloum_num] == 1:      # Sort out labelled +1 features
            x11.append(x1)
            x12.append(x2)
        elif label[__coloum_num] == 2:   # Sort out labelled +2 features
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
    plt.scatter(x31, x32, marker='o', c='#000000',
                label="Labelled as 0", edgecolor="white", s=60, alpha=0.5)  # Scatter labelled 0
    plt.scatter(x11, x12, marker='o', c='#4363D8',
                label="Labelled as 1", edgecolor="white", s=60, alpha=0.5)  # Scatter labelled +1
    plt.scatter(x21, x22, marker='o', c='#E6194B',
                label="Labelled as 2", edgecolor="white", s=60, alpha=0.5)  # Scatter labelled -1

    plt.xlabel(r'First Feature $x_1$', fontsize=18)
    plt.ylabel(r'Second Feature $x_2$', fontsize=18)
    plt.legend(loc='upper right')

    # Linear plot:
    fig2 = plt.figure('Data Visualization: lineplot', figsize=(16, 10))
    plt.title('Data Visualization: Sample week for station 2', fontsize=22)
    day_value = features[1152:3168, 3]
    para_value = (day_value - 1) * 1440
    x_value = features[1152:3168, 1] + para_value
    y_value = features[1152:3168, 2]
    plt.xlabel('Timeline (minute)', fontsize=18)
    plt.ylabel('Bikes / Total stands', fontsize=18)
    plt.plot(x_value, y_value, linewidth=3)


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
        __knn_model = KNeighborsClassifier(n_neighbors=k)
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


def decision_tree(features, labels):
    from sklearn import tree

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)

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


def roc_plot(features, labels):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import tree
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)

    best_k = 1
    print(f"\n\n----------------------kNN model with k = {best_k}----------------------\n")
    __knn_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance').fit(xtrain, ytrain)
    __pred = __knn_model.predict(xtest)
    print(classification_report(ytest, __pred))
    print("F1 score: \n", f1_score(ytest, __pred, average='weighted'))
    print("Confusion matrix:\n", confusion_matrix(ytest, __pred))

    best_depth = 5
    print(f"\n\n-----------------Decision Tree model with depth = {best_depth}-----------------\n")
    __tree_model = tree.DecisionTreeClassifier(max_depth=best_depth).fit(xtrain, ytrain)
    __pred = __tree_model.predict(xtest)
    print(classification_report(ytest, __pred))
    print("F1 score: \n", f1_score(ytest, __pred, average='weighted'))
    print("Confusion matrix:\n", confusion_matrix(ytest, __pred))

    # ROC for knn:
    pred_prob_knn = __knn_model.predict_proba(xtest)
    fpr_knn, tpr_knn, auc_knn = {}, {}, {}
    for i in range(3):
        fpr_knn[i], tpr_knn[i], _ = roc_curve(ytest, pred_prob_knn[:, i], pos_label=i)
        auc_knn[i] = auc(fpr_knn[i], tpr_knn[i])

    fig7 = plt.figure(figsize=(8, 6))
    plt.title("ROC plot of kNN classifier", fontsize=20)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Baseline Classifier: AUC = 0.50')
    plt.plot(fpr_knn[0], tpr_knn[0], color='blue', label='kNN Class 0 Classifiers: AUC = %0.4f' % auc_knn[0])
    plt.plot(fpr_knn[1], tpr_knn[1], color='red', label='kNN Class 1 Classifiers: AUC = %0.4f' % auc_knn[1])
    plt.plot(fpr_knn[2], tpr_knn[2], color='green', label='kNN Class 2 Classifiers: AUC = %0.4f' % auc_knn[2])
    plt.xlabel('False positive rate', fontsize=16)
    plt.ylabel('True positive rate', fontsize=16)
    plt.legend(loc='lower right')

    # ROC for tree:
    pred_prob_tree = __tree_model.predict_proba(xtest)
    fpr_tree, tpr_tree, auc_tree = {}, {}, {}
    print(pred_prob_tree)
    for i in range(3):
        fpr_tree[i], tpr_tree[i], _ = roc_curve(ytest, pred_prob_tree[:, i], pos_label=i)
        auc_tree[i] = auc(fpr_tree[i], tpr_tree[i])

    fig7 = plt.figure(figsize=(8, 6))
    plt.title("ROC plot of Decision Tree classifier", fontsize=20)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Baseline Classifier: AUC = 0.50')
    plt.plot(fpr_tree[0], tpr_tree[0], color='blue', label='DT Class 0 Classifiers: AUC = %0.4f' % auc_tree[0])
    plt.plot(fpr_tree[1], tpr_tree[1], color='red', label='DT Class 1 Classifiers: AUC = %0.4f' % auc_tree[1])
    plt.plot(fpr_tree[2], tpr_tree[2], color='green', label='DT Class 2 Classifiers: AUC = %0.4f' % auc_tree[2])
    plt.xlabel('False positive rate', fontsize=16)
    plt.ylabel('True positive rate', fontsize=16)
    plt.legend(loc='lower right')


def main(args):
    extract_csv(CSV_PATH)
    # print(f"Total number of dataset: {len(x)}")
    data_visualization(x, y)
    #
    # # Baseline model:
    # dummy_classifier(x, y)
    #
    # # Play with kNN model:
    # knn_classifier_tuning(x, y)
    #
    # # Play with decision tree model:
    # decision_tree(x, y)
    # roc_plot(x, y)

    plt.show()


if __name__ == '__main__':
    main(sys.argv)
