from numpy.lib.function_base import average
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import numpy as np
from numpy import linspace
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score,ConfusionMatrixDisplay, log_loss
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier
import sys

x, y = [], []  # Features (x) and labels (y)
CSV_PATH = 'C:\\Users\\kesha\\Desktop\\Machine_Learning\\MLProject_30\\dataset\\labelled_dataset_33.csv'


def extract_csv(path):
    global x, y, time
    df = pd.read_csv(path)
    n = len(df.iloc[:, 0])
    x0 = df.iloc[:n, 0]  # Station ID
    x1 = df.iloc[:n, 1]  # 1st feature: Total number of stands
    x2 = df.iloc[:n, 2]  # 2nd feature: Week (calender week of the year)
    x3 = df.iloc[:n, 3]  # 2nd feature: Weekday (Int [1, ..., 7])
    x4 = df.iloc[:n, 4]  # 3rd feature: Minute of the point (Int [0, ..., 24 * 60 - 5 = 1435])
    x5 = df.iloc[:n, 5]  # 4th feature: Available stands
    x6 = df.iloc[:n, 6]  # 5th feature: Available bikes
    x7 = df.iloc[:n, 7]  # 6th feature: Ratio Bike / Stand

    x = np.column_stack((x2, x3, x4, x7))  # Combine features into one matrix --> x
    #x = np.column_stack((x2, x3, x4))
    y = df.iloc[:n, 8]  # Third column as coorespondent labels
    # y = df.iloc[:, 5]


def dummy_classifier(features, labels):
    from sklearn.dummy import DummyClassifier
    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)
    __dummy_clf = DummyClassifier(strategy='most_frequent').fit(xtrain, ytrain)
    __dummy_pred = __dummy_clf.predict(xtest)

    print("Most Frequent Baseline Model:")
    print("F1 score \n", f1_score(ytest, __dummy_pred, average='weighted'))
    cm = confusion_matrix(ytest, __dummy_pred, labels = __dummy_clf.classes_)
    print("Confusion matrix: \n", cm)
    confusion_matrix_visual(__dummy_clf, cm, 'Confusion matrix - Baseline Model')

def confusion_matrix_visual(model, cm, title):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap = 'Blues')
    for labels in disp.text_.ravel():
        labels.set_fontsize(20)
    plt.title(title, fontsize = 18)
    plt.xlabel('Predicted Label', fontsize = 18)
    plt.ylabel('True Label', fontsize = 18)
    #plt.rc('xtick', labelsize=10)
    #plt.rc('ytick', labelsize=10)
    plt.show()


def logistic_regression_wo_pen(features, labels):
    from sklearn.calibration import CalibratedClassifierCV
    print("\n Logistic regression without penalty\n")

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)
    model = LogisticRegression(penalty='none',multi_class='ovr', solver='lbfgs',max_iter = 5000)
    model.fit(xtrain,ytrain)
    model = CalibratedClassifierCV(model, method="sigmoid",cv=5).fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    scores = cross_val_score(model,xtest,ytest, cv = 5, scoring='f1_weighted')
    print("F1 scores:")
    print("Mean f1 score using 5-fold cross validation", scores.mean())
    cm = confusion_matrix(ytest, ypred)
    #print("Confusion matrix: \n", cm)
    confusion_matrix_visual(model, cm, 'Confusion matrix - LR without penalty')
    print(classification_report(ytest,ypred))


def logistic_regression(features, labels):
    print("\nLogistic regression model with penalty l2\n")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import CalibratedClassifierCV

    C = [0.0001, 0.1, 0.5, 1, 5]
    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)

    mean_error = []
    std_error = []

    for c in C:
        model = LogisticRegression(penalty='l2',multi_class='ovr', solver='lbfgs',C = c, max_iter = 1000).fit(xtrain,ytrain)
        cal_model = CalibratedClassifierCV(model, method="sigmoid",cv=5).fit(xtrain,ytrain)
        scores = cross_val_score(cal_model, xtest, ytest, cv = 5, scoring='f1_weighted')
        print(scores.mean())
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    # Plot cross-validation result for C parameter
    plt.figure()
    plt.errorbar(C, mean_error, yerr=std_error, linewidth = 3)
    plt.xlabel('C', fontsize = 18)
    plt.ylabel('F1 Score', fontsize = 18)
    plt.title('Cross-validation to determine C', fontsize = 18)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    #plt.xlim((c_range[0],c_range[len(c_range)-1]))
    #plt.savefig('crossvalidation for c using F1 score.png')
    plt.show()


    # Evaluation
    model = LogisticRegression(penalty='l2',multi_class='ovr', solver='lbfgs',C = 0.1, max_iter = 1000).fit(xtrain,ytrain)
    model = CalibratedClassifierCV(model, method="sigmoid",cv=5)
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    print("\n\n----------------------Logistic Regression Evaluation (without penalty)----------------------\n\n")
    print(classification_report(ytest,ypred))
    cm = confusion_matrix(ytest, ypred)
    #print("Confusion matrix:\n", cm)
    confusion_matrix_visual(model, cm, 'Confusion Matrix - LR, penalty = none')


def roc_plot(features, labels):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import tree
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)

    # ROC for Logistic regression
    model = LogisticRegression(penalty='none',multi_class='ovr', solver='lbfgs',max_iter = 5000)
    model.fit(xtrain,ytrain)
    ypred = model.predict_proba(xtest)
    fpr, tpr, auc_lr = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(ytest, ypred[:, i], pos_label=i)
        #fpr, tpr, _ = roc_curve(ytest,model.decision_function(xtest))
        auc_lr[i] = auc(fpr[i], tpr[i])

    fig7 = plt.figure(figsize=(8, 6))
    plt.title("ROC plot of Logistic Regression classifier", fontsize=16)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Baseline Classifier: AUC = 0.50')
    plt.plot(fpr[0], tpr[0], color='blue', label='LR Class 0 Classifiers: AUC = %0.4f' % auc_lr[0])
    plt.plot(fpr[1], tpr[1], color='red', label='LR Class 1 Classifiers: AUC = %0.4f' % auc_lr[1])
    plt.plot(fpr[2], tpr[2], color='green', label='LR Class 2 Classifiers: AUC = %0.4f' % auc_lr[2])
    plt.xlabel('False positive rate', fontsize=16)
    plt.ylabel('True positive rate', fontsize=16)
    plt.legend(loc='lower right')
    plt.show()


extract_csv(CSV_PATH)


print(f"Total number of dataset: {len(x)}")

## Baseline
dummy_classifier(x, y)

## Logistic regression without penalty
logistic_regression_wo_pen(x,y)

## Logistic regression with penalty
logistic_regression(x,y)

## ROC curve for logistic regression
roc_plot(x, y)

