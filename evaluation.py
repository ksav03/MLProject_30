# Developed by Keshav, Nasir, Shu, December 2021
# All rights reserved
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

STATION_ID = 33
CSV_PATH = f'dataset/labelled_dataset_{STATION_ID}.csv'
x, y, xtemp = [], [], []  # Features (x) and labels (y)


def extract_csv(path):
    global x, y, xtemp
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
    x = np.column_stack((x3, x4, x5, x6))
    xtemp = np.column_stack((x3, x4, x7, x3))
    y = df.iloc[:, 8]  # Third column as coorespondent labels
    # y = df.iloc[:, 5]


def make_binary_label_arr(arr):
    temp = []
    for i, item in enumerate(arr):
        arr_to_add = []
        if item == 0:
            arr_to_add = [1, 0, 0]
        if item == 1:
            arr_to_add = [0, 1, 0]
        if item == 2:
            arr_to_add = [0, 0, 1]
        temp.append(arr_to_add)
    return np.asarray(temp)


def separate_data_by_label(feature, label):
    # This function (method) is to separate data into 2 groups: x1* and x2*, and returns 4 lists
    # Each group contains two features: e.g. x1* contains x11 and x12
    # x1* contains data labelled as +1 while x2* contains data labelled as -1
    __coloum_num = 0
    x11, x12, x21, x22, x31, x32 = [], [], [], [], [], []
    count0, count1, count2 = 0, 0, 0
    for day, x1, _, x2 in feature:
        if label[__coloum_num] == 1:      # Sort out labelled +1 features
            x11.append(x1)
            x12.append(x2)
            count1 += 1
        elif label[__coloum_num] == 2:   # Sort out labelled +2 features
            x21.append(x1)
            x22.append(x2)
            count2 += 1
        else:
            x31.append(x1)
            x32.append(x2)
            count0 += 1
        __coloum_num += 1
    print(f"Count: {count0, count1, count2}")
    return x11, x12, x21, x22, x31, x32


def data_visualization(features, labels):
    print(features.shape, type(features))

    x11, x12, x21, x22, x31, x32 = separate_data_by_label(features, labels)

    # Plotting: used multiple times
    fig1 = plt.figure('Overall Data Visualization', figsize=(12, 9))
    plt.title('Data Visualization', fontsize=22)
    plt.rcParams.update({'font.size': 14})
    plt.scatter(x31, x32, marker='o', c='#000000',
                label="Labelled as 0", edgecolor="white", s=60, alpha=0.5)  # Scatter labelled 0
    plt.scatter(x11, x12, marker='o', c='#4363D8',
                label="Labelled as 1", edgecolor="white", s=60, alpha=0.5)  # Scatter labelled +1
    plt.scatter(x21, x22, marker='o', c='#E6194B',
                label="Labelled as 2", edgecolor="white", s=60, alpha=0.5)  # Scatter labelled -1

    plt.xlabel(r'First Feature: Ratio', fontsize=18)
    plt.ylabel(r'Second Feature: Time node', fontsize=18)
    plt.legend(loc='upper right')

    # Linear plot:
    global xtemp
    fig2 = plt.figure('Data Visualization: lineplot', figsize=(12, 9))
    plt.title(f'Data Visualization: Sample week for station {STATION_ID}', fontsize=22)
    day_value = xtemp[1152:3168, 3]
    para_value = (day_value - 1) * 1440
    x_value = xtemp[1152:3168, 1] + para_value
    y_value = xtemp[1152:3168, 2]
    plt.xlabel('Timeline (minute)', fontsize=18)
    plt.ylabel('Bikes / Total stands', fontsize=18)
    plt.plot(x_value, y_value, linewidth=3)


def roc_plot(features, labels):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import tree
    from sklearn.tree import plot_tree
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV

    best_k = 16             # Optimal parameter for kNN classifier
    best_depth = 4          # Optimal parameter for Decision Trees classifier

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.4)

    print(f"\n\n----------------------kNN model with k = {best_k}----------------------\n")
    __knn_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    __knn_model = CalibratedClassifierCV(__knn_model, method='isotonic', cv=5).fit(xtrain, ytrain)
    __pred = __knn_model.predict(xtest)
    print(classification_report(ytest, __pred))
    print("F1 score: \n", f1_score(ytest, __pred, average='weighted'))
    print("Confusion matrix:\n", confusion_matrix(ytest, __pred))

    print(f"\n\n-----------------Decision Tree model with depth = {best_depth}-----------------\n")
    __tree_model = tree.DecisionTreeClassifier(max_depth=best_depth)
    __tree_model = CalibratedClassifierCV(__tree_model, method='isotonic', cv=5).fit(xtrain, ytrain)
    __pred = __tree_model.predict(xtest)
    print(classification_report(ytest, __pred))
    print("F1 score: \n", f1_score(ytest, __pred, average='weighted'))
    print("Confusion matrix:\n", confusion_matrix(ytest, __pred))

    print(f"\n\n-----------------SVM model with c = 0.1-----------------\n")
    svm_model = LinearSVC(penalty='l2', C=0.1, dual=False, max_iter=5000, class_weight='balanced', multi_class='ovr')
    # svm_model = svm_model.fit(xtrain, ytrain)
    svm_model = CalibratedClassifierCV(svm_model, method='isotonic', cv=5).fit(xtrain, ytrain)
    __pred = svm_model.predict(xtest)
    print(classification_report(ytest, __pred))
    print("F1 score: \n", f1_score(ytest, __pred, average='weighted'))
    print("Confusion matrix:\n", confusion_matrix(ytest, __pred))

    print(f"\n\n-----------------Linear Regression model with no penalty-----------------\n")
    lr_model = LogisticRegression(penalty='none', multi_class='ovr', solver='lbfgs', max_iter=5000)
    lr_model.fit(xtrain, ytrain)
    __pred = lr_model.predict(xtest)
    print(classification_report(ytest, __pred))
    print("F1 score: \n", f1_score(ytest, __pred, average='weighted'))
    print("Confusion matrix:\n", confusion_matrix(ytest, __pred))

    # Setup ROCs for displaying
    ytest_temp = make_binary_label_arr(ytest)
    # ROC for tree:
    pred_prob_tree = __tree_model.predict_proba(xtest)
    fpr_tree, tpr_tree, auc_tree = {}, {}, {}
    print(pred_prob_tree)
    for i in range(3):
        fpr_tree[i], tpr_tree[i], _ = roc_curve(ytest, pred_prob_tree[:, i], pos_label=i)
        auc_tree[i] = auc(fpr_tree[i], tpr_tree[i])
    fpr_tree["micro"], tpr_tree["micro"], _ = roc_curve(ytest_temp.ravel(), pred_prob_tree.ravel())
    auc_tree["micro"] = auc(fpr_tree["micro"], tpr_tree["micro"])

    # ROC for svm:
    # pred_prob_svm = svm_model.decision_function(xtest)
    pred_prob_svm = svm_model.predict_proba(xtest)
    fpr_svm, tpr_svm, auc_svm = {}, {}, {}
    for i in range(3):
        fpr_svm[i], tpr_svm[i], _ = roc_curve(ytest, pred_prob_svm[:, i], pos_label=i)
        auc_svm[i] = auc(fpr_svm[i], tpr_svm[i])
    print(auc_svm)
    fpr_svm["micro"], tpr_svm["micro"], _ = roc_curve(ytest_temp.ravel(), pred_prob_svm.ravel())
    auc_svm["micro"] = auc(fpr_svm["micro"], tpr_svm["micro"])

    # ROC for knn:
    pred_prob_knn = __knn_model.predict_proba(xtest)
    fpr_knn, tpr_knn, auc_knn = {}, {}, {}
    for i in range(3):
        fpr_knn[i], tpr_knn[i], _ = roc_curve(ytest, pred_prob_knn[:, i], pos_label=i)
        auc_knn[i] = auc(fpr_knn[i], tpr_knn[i])
    print(auc_knn)
    fpr_knn["micro"], tpr_knn["micro"], _ = roc_curve(ytest_temp.ravel(), pred_prob_knn.ravel())
    auc_knn["micro"] = auc(fpr_knn["micro"], tpr_knn["micro"])

    # ROC for Logistic Regression
    pred_prob_lr = lr_model.predict_proba(xtest)
    fpr_lr, tpr_lr, auc_lr = {}, {}, {}
    for i in range(3):
        fpr_lr[i], tpr_lr[i], _ = roc_curve(ytest, pred_prob_lr[:, i], pos_label=i)
        auc_lr[i] = auc(fpr_lr[i], tpr_lr[i])
    fpr_lr["micro"], tpr_lr["micro"], _ = roc_curve(ytest_temp.ravel(), pred_prob_lr.ravel())
    auc_lr["micro"] = auc(fpr_lr["micro"], tpr_lr["micro"])
    # fig7 = plt.figure(figsize=(8, 6))
    # plt.title("ROC plots of kNN classifier", fontsize=20)
    # plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Baseline Classifier: AUC = 0.50')
    # plt.plot(fpr_knn[0], tpr_knn[0], color='blue', label='kNN Class 0 Classifiers: AUC = %0.4f' % auc_knn[0])
    # plt.plot(fpr_knn[1], tpr_knn[1], color='red', label='kNN Class 1 Classifiers: AUC = %0.4f' % auc_knn[1])
    # plt.plot(fpr_knn[2], tpr_knn[2], color='green', label='kNN Class 2 Classifiers: AUC = %0.4f' % auc_knn[2])
    # plt.xlabel('False positive rate', fontsize=16)
    # plt.ylabel('True positive rate', fontsize=16)
    # plt.legend(loc='lower right')



    # fig8 = plt.figure(figsize=(8, 6))
    # plt.title("ROC plots of Decision Tree classifier", fontsize=20)
    # plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Baseline Classifier: AUC = 0.50')
    # plt.plot(fpr_tree[0], tpr_tree[0], color='blue', label='DT Class 0 Classifiers: AUC = %0.4f' % auc_tree[0])
    # plt.plot(fpr_tree[1], tpr_tree[1], color='red', label='DT Class 1 Classifiers: AUC = %0.4f' % auc_tree[1])
    # plt.plot(fpr_tree[2], tpr_tree[2], color='green', label='DT Class 2 Classifiers: AUC = %0.4f' % auc_tree[2])
    # plt.xlabel('False positive rate', fontsize=16)
    # plt.ylabel('True positive rate', fontsize=16)
    # plt.legend(loc='lower right')

    fig9 = plt.figure(figsize=(8, 6))
    plt.title("ROC plots of Classifiers used", fontsize=20)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Baseline Classifier: AUC = 0.50')
    plt.plot(fpr_knn["micro"], tpr_knn["micro"], color='green', label='kNN Classifier: AUC = %0.4f' % auc_knn["micro"])
    plt.plot(fpr_tree["micro"], tpr_tree["micro"], color='blue', label='DT Classifier: AUC = %0.4f' % auc_tree["micro"])
    plt.plot(fpr_svm["micro"], tpr_svm["micro"], color='red', label='SVM Classifier: AUC = %0.4f' % auc_svm["micro"])
    plt.plot(fpr_lr["micro"], tpr_lr["micro"], color='purple',
             label='Logistic Regression Classifier: AUC = %0.4f' % auc_lr["micro"])
    plt.xlabel('False positive rate', fontsize=16)
    plt.ylabel('True positive rate', fontsize=16)
    plt.legend(loc='lower right')


def main(args):
    extract_csv(CSV_PATH)
    print(f"Total number of dataset: {len(x)}")
    # data_visualization(x, y)

    # ROC plots
    roc_plot(x, y)
    plt.show()


if __name__ == '__main__':
    main(sys.argv)