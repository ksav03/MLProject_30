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


def time_to_minutes(data):
    """
    This function will return the total minutes passed during 24 time
    input data: HH:MM:SS
    return: Minutes
    """
    minute_value = []
    month_value = []
    day_value = []
    for dates in data:
        # | Get the time values, date is also stored if needed
        date, separation, time = dates.partition(' ')
        hour, minute, second = time.split(':')
        # | Convert into minutes and store in array
        total_time_minutes = int(hour) * 60 + int(minute)
        minute_value.append(total_time_minutes)
        # | Format the dates
        date_info = datetime.datetime.strptime(date, '%Y-%m-%d')
        month_value.append(date_info.month)
        day_value.append(date_info.day)

    return minute_value, month_value, day_value


def split_data(x, x2, y):
    # | Initialise variables for the pos and neg values
    x_pos, x_neg, x2_pos, x2_neg = [], [], [], []

    #| sort data based on classes
    for i, num in enumerate(y):
        if num == 1:
            x_pos.append(x[i])
            x2_pos.append(x2[i])
        else:
            x_neg.append(x[i])
            x2_neg.append(x2[i])
    return x_pos, x_neg, x2_pos, x2_neg


def svm(X, y):
    print("[INFO] Running question b(i)...")
    #| setup our C values
    penalty = [0.001, 1, 40, 100, 200, 400]
    #| Initialise our graphs
    fig, axes = plt.subplots(2, 3)
    j = 0
    for C in penalty:
        model = LinearSVC(C=C, dual=True).fit(X, y)
        #| Print model info
        print("For C = " + str(C))
        # print("intercept %f, slope %f" % (model.intercept_, model.coef_))
        # print("model coefs are: ")
        # print(model.coef_)

        predict = model.predict(X)
        print(confusion_matrix(y, predict))
        # j = predict == X
        # print(f'{j=}')
    print("[INFO] Question b(i) completed")


def find_best_c_f1(X, y):
    """
    This function uses k_fold cross validation to plot the F1 score
    against a range of C values to find the optimum C value.
    """
    print("[INFO] Finding best C for svm...")
    std, f1 = [], []
    # c_weights = np.linspace(0.01, 15, 15)
    c_weights = [0.01, 0.1, 2, 4, 7, 10, 12, 15]
    for c in c_weights:
        model = LinearSVC(dual=True, C=c, max_iter=1000)
        scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        #| Add the mean of the f1 and std for each iteration
        f1.append(np.array(scores).mean())
        std.append(np.array(scores).std())
        model = model.fit(X, y)
        predict = model.predict(X)
        print(confusion_matrix(y, predict))
    #| plot the F1 vs c_weights
    plt.errorbar(c_weights, f1, yerr=std, linewidth=3)
    plt.xlabel('C', fontsize=20)
    plt.ylabel('F1 score', fontsize=20)
    plt.show()
    print("[INFO] Done")


def svm_date_only_cross_val(X, y):
    """
        This function uses k_fold cross validation to plot the F1 score
        against a range of C values to find the optimum C value.
        """
    print("[INFO] Finding best C for svm...")
    std, f1 = [], []
    # c_weights = np.linspace(0.01, 15, 15)
    c_weights = [0.01, 0.1, 2, 4, 7, 10, 12, 15]
    for c in c_weights:
        model = LinearSVC(dual=True, C=c, max_iter=3000)
        scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        # | Add the mean of the f1 and std for each iteration
        f1.append(np.array(scores).mean())
        std.append(np.array(scores).std())
        model = model.fit(X, y)
        predict = model.predict(X)
        print(f'{predict=}')
        # print(confusion_matrix(y, predict))
    # | plot the F1 vs c_weights
    plt.errorbar(c_weights, f1, yerr=std, linewidth=3)
    plt.xlabel('C', fontsize=20)
    plt.ylabel('F1 score', fontsize=20)
    plt.show()
    print("[INFO] Done")


csv_path = "dataset\labelled_dataset_2.csv"
num_data = 3000
df = pd.read_csv(csv_path)
minutes, month, day = time_to_minutes(df.iloc[:num_data, 2])
x1 = df.iloc[:num_data, 1]
x2 = df.iloc[:num_data, 3]
x3 = df.iloc[:num_data, 4]
x4 = minutes
x5 = month
x6 = day

y = df.iloc[:num_data, 6]

X = np.column_stack((x1, x2, x3, x4, x5, x6))
X_date_info = np.column_stack((x4, x5, x6))
print(f'{X=}')
svm(X, y)
# find_best_c_f1(X, y)
# svm_date_only_cross_val(X_date_info, y)


























