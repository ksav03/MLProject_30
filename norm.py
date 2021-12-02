from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.dummy import DummyClassifier

DATA_SPLIT = 0.4
CSV_PATH = 'data\dataset\labelled_dataset_2.csv'
DATAFILES = ['data\dataset\labelled_dataset_33.csv']


def extract_csv(path):
    df = pd.read_csv(path)
    n = len(df.iloc[:, 0])
    # x0 = df.iloc[:n, 0]  # Station ID
    # x1 = df.iloc[:n, 1]  # 1st feature: Total number of stands
    x2 = df.iloc[:n, 2]  # 2nd feature: Week (calender week of the year)
    x3 = df.iloc[:n, 3]  # 2nd feature: Weekday (Int [1, ..., 7])
    x4 = df.iloc[:n, 4]  # 3rd feature: Minute of the point (Int [0, ..., 24 * 60 - 5 = 1435])
    x5 = df.iloc[:n, 5]  # 4th feature: Available stands
    # x6 = df.iloc[:n, 6]  # 5th feature: Available bikes
    x7 = df.iloc[:n, 7]  # 6th feature: Ratio Bike / Stand

    # x = np.column_stack((x1, x2, x3, x4, x5, x6, x7))  # Combine features into one matrix --> x
    # x = np.column_stack((x2, x3, x5, x4, x7))  # adding the available stands
    x = np.column_stack((x2, x3, x4, x7))
    y = df.iloc[:n, 8]  # Third column as co-responding labels
    return x, y


def extract_multiple_csv_data(datafiles):
    """
    This function concatenates multiple csv files and extracts the features we want
    param datafiles: .csv file paths in a list
    return: input features x and labels y
    """
    data = []
    for file in datafiles:
        df = pd.read_csv(file, index_col=None, header=0)
        data.append(df)
    main_data_frame = pd.concat(data, axis=0, ignore_index=True)

    n = len(main_data_frame.iloc[:, 0])
    x0 = main_data_frame.iloc[:n, 0]  # Station ID
    x2 = main_data_frame.iloc[:n, 2]  # 2nd feature: Week (calender week of the year)
    x3 = main_data_frame.iloc[:n, 3]  # 2nd feature: Weekday (Int [1, ..., 7])
    x4 = main_data_frame.iloc[:n, 4]  # 3rd feature: Minute of the point (Int [0, ..., 24 * 60 - 5 = 1435])
    #| Convert the time into decimals
    # counter = 0
    arr = []
    for item in x4:
        counter = 0
        while item >= 60:
            item = item - 60
            counter += 1
        if item < 10:
            # print(f'{counter}.0{item}')
            arr.append(float(str(counter)+".0"+str(item)))
        # elif item == 10:
        #     print(f'{counter}.{item}0')
        #     arr.append(float(str(counter) + "." + str(item) + "0"))
        else:
            arr.append(float(str(counter) + "." + str(item)))
    arr = np.asarray(arr)

    # print(arr)
    x7 = main_data_frame.iloc[:n, 7]  # 6th feature: Ratio Bike / Stand
    # x = np.column_stack((x0, x2, x3, x4, x7))
    x = np.column_stack((x0, x2, x3, arr, x7))
    counter = 0
    arr = []
    y_arr = []
    y = main_data_frame.iloc[:n, 8]  # Third column as co-responding labels
    for i, item in enumerate(x):
        counter += 1
        # if counter % 3 == 0 and (y[i] == 0 or y[i] == 1):
        if counter % 3 == 0 and (y[i] == 0):
            arr.append(item)
            y_arr.append(y[i])
        # if y[i] == 2:
        if y[i] != 0:
            arr.append(item)
            y_arr.append(y[i])
        # print(item)
    arr = np.asarray(arr)
    y_arr = np.asarray(y_arr)
    print(arr.shape)
    print(y_arr.shape)
    #| Normalize the data for minutes and ratio
    #| ------------------------Normalizing doesn't help --------------------------
    # from sklearn.preprocessing import normalize
    # temp = np.column_stack((x4, x7))
    # x4 = normalize(temp, norm='l2', axis=1, copy=True, return_norm=False)
    # x = np.column_stack((x0, x2, x3, x4))
    # print(X)
    # print(f'{x=}')
    # print(f'{x.shape=}')
    # y = main_data_frame.iloc[:n, 8]  # Third column as co-responding labels
    # print(f'{y=}')
    # print(f'{y.shape=}')
    return arr, y_arr


def dummy_classifier(features, labels):
    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=DATA_SPLIT)
    __dummy_clf = DummyClassifier(strategy='most_frequent').fit(xtrain, ytrain)
    __dummy_pred = __dummy_clf.predict(xtest)

    print("Most Frequent Baseline Model:")
    print("F1 score \n", f1_score(ytest, __dummy_pred, average='weighted'))
    print("Confusion matrix: \n", confusion_matrix(ytest, __dummy_pred))


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


def svm(features, labels):
    print(f'\n[INFO] Function "{svm.__name__}" running...')
    print(f'[INFO] Splitting data...')
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=DATA_SPLIT)
    #| There doesn't seem to be a big difference between the next two lines
    # model = LinearSVC(penalty='l2', dual=True, C=0.66, max_iter=5000)
    print(f'[INFO] Creating SVM model...')
    model = LinearSVC(penalty='l2', dual=False, C=0.4, max_iter=3000, class_weight='balanced', multi_class='ovr')
    # weight = {0: 0.8, 1: 0.3, 2: 450}
    # model = LinearSVC(penalty='l2', dual=False, C=0.4, max_iter=3000, class_weight=weight, multi_class='ovr')
    print(f'[INFO] Fitting SVM model...')
    model.fit(x_train, y_train)
    print(model.coef_)
    y_pred = model.predict(x_test)
    #| Get the probability of a prediction for ROC
    y_score = model.decision_function(x_test)
    #| Setup test and pred data to be:
    #| [0, 0, 1]
    #| [1, 0, 0]
    #|     :
    #| rather than y_p and y_t = [2, 0 , ...]
    print(f'[INFO] Changing prediction and test data shape for confusion matrix...')
    y_t = make_binary_label_arr(y_test)
    y_pred_temp = y_pred.reshape((y_pred.shape[0], 1))
    y_p = make_binary_label_arr(y_pred_temp)
    print(f'[INFO] Done')
    print(f'[INFO] Printing classification report and confusion matrix')
    print(classification_report(y_test, y_pred))
    conf_mat = confusion_matrix(y_t.argmax(axis=1), y_p.argmax(axis=1))
    # print("Confusion matrix:\n", confusion_matrix(y_t.argmax(axis=1), y_p.argmax(axis=1)))
    print("Confusion matrix:\n", conf_mat)
    print(f'[INFO] Done')
    print(f'[INFO] Printing prediction accuracy')

    print(f'[INFO] Accuracy for class 0 is {((conf_mat[0][0] / sum(conf_mat[0])) * 100):.2f}')
    print(f'[INFO] Accuracy for class 1 is {((conf_mat[1][1] / sum(conf_mat[1])) * 100):.2f}')
    print(f'[INFO] Accuracy for class 2 is {((conf_mat[2][2] / sum(conf_mat[2])) * 100):.2f}')
    print(f'[INFO] Done')

    print(f'[INFO] Getting ROC and AUC...')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # for i, item in enumerate(y_score):
    #     print(f'{item} vs {y_p[i]} vs actual:{y_t[i]}')

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_t[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print(f'[INFO] Plotting ROC and AUC...')
    colors = ['red', 'blue', 'green']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.xlim([0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for classes 0, 1, 2')
    plt.legend(loc="lower right")
    plt.show()
    print(f'[INFO] Done')
    print(f'[INFO] Function call {svm.__name__} completed')


def svm_cross_val(features, labels):
    print(f'\n[INFO] Function "{svm_cross_val.__name__}" running...')
    print(f'[INFO] Running SVM model...')

    c_weights = np.linspace(0.001, 1, 15)
    print(f'[INFO] Splitting data...')
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=DATA_SPLIT)

    mean_error = []
    std_error = []
    print(f'[INFO] Running cross validation with parameters:\n\t   {c_weights=} and cv={5}')
    for c in c_weights:
        # model = LogisticRegression(penalty='l2', multi_class='ovr', solver='lbfgs', C=c, max_iter=5000)
        # model = LinearSVC(penalty='l2', dual=True, C=c, max_iter=3000)
        model = LinearSVC(penalty='l2', dual=False, C=c, max_iter=3000)
        model.fit(x_train, y_train)
        scores = cross_val_score(model, x_test, y_test, cv=5, scoring='f1_weighted')
        # print(scores)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    # Plot cross-validation result for C parameter
    plt.figure(figsize=(9, 9))
    plt.errorbar(c_weights, mean_error, yerr=std_error, linewidth = 3)
    plt.xlabel('c')
    plt.ylabel('F1 Score')
    plt.show()
    #| Get the index of the cross val with the highest f1 score
    max_value = max(mean_error)
    max_index = mean_error.index(max_value)
    print(f'[INFO] Optimum C value is C={c_weights[max_index]}')
    print(f'[INFO] Creating SVM model...')
    #| There doesn't seem to be a big difference between the next two lines
    # model = LinearSVC(penalty='l2', dual=True, C=0.66, max_iter=5000)
    model = LinearSVC(penalty='l2', dual=False, C=0.4, max_iter=3000)
    print(f'[INFO] Fitting SVM model...')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    #| Get the probability of a prediction for ROC
    y_score = model.decision_function(x_test)
    #| Setup test and pred data to be:
    #| [0, 0, 1]
    #| [1, 0, 0]
    #|     :
    #| rather than y_p and y_t = [2, 0 , ...]
    print(f'[INFO] Changing prediction and test data shape for confusion matrix...')
    y_t = make_binary_label_arr(y_test)
    y_pred_temp = y_pred.reshape((y_pred.shape[0], 1))
    y_p = make_binary_label_arr(y_pred_temp)
    print(f'[INFO] Done')
    print(f'[INFO] Printing classification report and confusion matrix')
    print(classification_report(y_test, y_pred))
    conf_mat = confusion_matrix(y_t.argmax(axis=1), y_p.argmax(axis=1))
    # print("Confusion matrix:\n", confusion_matrix(y_t.argmax(axis=1), y_p.argmax(axis=1)))
    print("Confusion matrix:\n", conf_mat)
    print(f'[INFO] Done')
    print(f'[INFO] Printing prediction accuracy')

    print(f'[INFO] Accuracy for class 0 is {((conf_mat[0][0] / sum(conf_mat[0])) * 100):.2f}')
    print(f'[INFO] Accuracy for class 1 is {((conf_mat[1][1] / sum(conf_mat[1])) * 100):.2f}')
    print(f'[INFO] Accuracy for class 2 is {((conf_mat[2][2] / sum(conf_mat[2])) * 100):.2f}')
    print(f'[INFO] Done')

    print(f'[INFO] Getting ROC and AUC...')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_t[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print(f'[INFO] Plotting ROC and AUC...')
    colors = ['red', 'blue', 'green']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.xlim([0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for classes 0, 1, 2')
    plt.legend(loc="lower right")
    plt.show()
    print(f'[INFO] Done')
    print(f'[INFO] Function call {svm_cross_val.__name__} completed')


print(f'[INFO] Running programme')
X, Y = extract_multiple_csv_data(DATAFILES)
# from sklearn.preprocessing import normalize
# X_normalized = normalize(X, norm='l2', axis=1, copy=True, return_norm=False)
print(X)
# print(X_normalized)
# X, Y = extract_csv(CSV_PATH, DATAFILES)
# print(f"Total number of dataset: {len(X)}")
dummy_classifier(X, Y)
svm(X, Y)
# svm_cross_val(X, Y)
print(f'[INFO] Done')
