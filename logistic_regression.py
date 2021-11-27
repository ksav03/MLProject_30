from numpy.lib.function_base import average
from sklearn.svm import LinearSVC
import numpy as np
from numpy import linspace
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

x, y = [], []  # Features (x) and labels (y)
CSV_PATH = 'C:\\Users\\kesha\\Desktop\\Machine_Learning\\MLProject_30\\dataset\\labelled_dataset_2.csv'

def extract_csv(path):
    global x, y
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

    x = np.column_stack((x1, x2, x3, x4, x5, x6, x7))  # Combine features into one matrix --> x
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
    print("Confusion matrix: \n", confusion_matrix(ytest, __dummy_pred))


def logistic_regression_wo_pen(features, labels):
    print("\n Logistic regression without penalty\n")

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)
    model = LogisticRegression(penalty='none',multi_class='ovr', solver='lbfgs',max_iter = 5000)
    model.fit(xtest, ytest)
    scores = cross_val_score(model,xtest,ytest, cv = 5, scoring='f1_weighted')
    print(scores)


def logistic_regression(features, labels):
    print("\nLogistic regression model with penalty l2\n")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    C = [0.001, 0.1, 10, 50]
    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)

    mean_error = []
    std_error = []

    for c in C:
        model = LogisticRegression(penalty='l2',multi_class='ovr', solver='lbfgs',C = c, max_iter = 5000)
        model.fit(xtrain,ytrain)
        scores = cross_val_score(model, xtest, ytest, cv = 5, scoring='f1_weighted')
        print(scores)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    # Plot cross-validation result for C parameter
    plt.figure(figsize=(9,9))
    plt.errorbar(C, mean_error, yerr=std_error, linewidth = 3)
    plt.xlabel('c')
    plt.ylabel('F1 Score')
    #plt.xlim((c_range[0],c_range[len(c_range)-1]))
    #plt.savefig('crossvalidation for c using F1 score.png')
    plt.show()


    # Evaluation
    model = LogisticRegression(penalty='none',multi_class='ovr', solver='lbfgs',max_iter = 5000)
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    print("\n\n----------------------Logistic Regression Evaluation (without penalty)----------------------\n\n")
    print(classification_report(ytest,ypred))
    print("Confusion matrix:\n", confusion_matrix(ytest, ypred))


    # ROC curve - ERROR "multiclass format is not supported"
'''    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(ytest,model.decision_function(xtest))
    plt.plot(fpr,tpr, 'blue', label = 'Logistic regression')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')'''


extract_csv(CSV_PATH)
print(f"Total number of dataset: {len(x)}")
dummy_classifier(x, y)
# Logistic regression without penalty
logistic_regression_wo_pen(x,y)
# with penalty
logistic_regression(x,y)


