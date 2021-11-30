from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.dummy import DummyClassifier

x, y = [], []  # Features (x) and labels (y)
CSV_PATH = 'data\dataset\labelled_dataset_2.csv'

def extract_csv(path):
    global x, y
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


def dummy_classifier(features, labels):
    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2)
    __dummy_clf = DummyClassifier(strategy='most_frequent').fit(xtrain, ytrain)
    __dummy_pred = __dummy_clf.predict(xtest)

    print("Most Frequent Baseline Model:")
    print("F1 score \n", f1_score(ytest, __dummy_pred, average='weighted'))
    print("Confusion matrix: \n", confusion_matrix(ytest, __dummy_pred))


def svm(features, labels):
    print(f'\n[INFO] Function "{svm.__name__}" running...')
    print(f'[INFO] Splitting data...')
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    #| There doesn't seem to be a big difference between the next two lines
    # model = LinearSVC(penalty='l2', dual=True, C=0.66, max_iter=5000)
    print(f'[INFO] Creating SVM model...')
    model = OneVsRestClassifier(LinearSVC(penalty='l2', dual=True, C=4, max_iter=3000))
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
    arr = []
    for i, item in enumerate(y_test):
        arr_to_add = []
        if item == 0:
            arr_to_add = [1, 0, 0]
        if item == 1:
            arr_to_add = [0, 1, 0]
        if item == 2:
            arr_to_add = [0, 0, 1]
        arr.append(arr_to_add)
    y_t = np.asarray(arr)
    arr = []
    y_pred_temp = y_pred.reshape((y_pred.shape[0], 1))
    for i, item in enumerate(y_pred_temp):
        arr_to_add = []
        if item == 0:
            arr_to_add = [1, 0, 0]
        if item == 1:
            arr_to_add = [0, 1, 0]
        if item == 2:
            arr_to_add = [0, 0, 1]
        arr.append(arr_to_add)
    y_p = np.asarray(arr)
    print(f'[INFO] Done')
    print(f'[INFO] Printing classification report and confusion matrix')
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_t.argmax(axis=1), y_p.argmax(axis=1)))
    print(f'[INFO] Done')
    print(f'[INFO] Printing prediction accuracy')
    #| Get all instances where predicted values match the test
    #| resulting arrays are boolean
    num_match_zero = y_t[:, 0] == y_p[:, 0]
    num_match_one = y_t[:, 1] == y_p[:, 1]
    num_match_two = y_t[:, 2] == y_p[:, 2]
    # count = np.count_nonzero(arr)
    #| \t == 4 white spaces
    print(f'\t   Number of class 0 matches out of {len(y_test)} is: {np.count_nonzero(num_match_zero)}'
          f'\n\t   Model correctly predicts class 0 {((np.count_nonzero(num_match_zero) / len(y_test)) * 100):.2f}% '
          f'of the time')
    print(f'\t   Number of class 1 matches out of {len(y_test)} is: {np.count_nonzero(num_match_one)}'
          f'\n\t   Model correctly predicts class 1 {((np.count_nonzero(num_match_one) / len(y_test)) * 100):.2f}% '
          f'of the time')

    print(f'\t   Number of class 2 matches out of {len(y_test)} is: {np.count_nonzero(num_match_two)}'
          f'\n\t   Model correctly predicts class 1 {((np.count_nonzero(num_match_two) / len(y_test)) * 100):.2f}% '
          f'of the time')
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
    print(f'[INFO] Function call {svm.__name__} completed')


def svm_cross_val(features, labels):
    print(f'\n[INFO] Function "{svm_cross_val.__name__}" running...')
    print(f'[INFO] Running SVM model...')

    c_weights = np.linspace(0.01, 10, 15)
    print(f'[INFO] Splitting data...')
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    mean_error = []
    std_error = []
    print(f'[INFO] Running cross validation with parameters:\n\t   {c_weights=} and cv={5}')
    for c in c_weights:
        # model = LogisticRegression(penalty='l2', multi_class='ovr', solver='lbfgs', C=c, max_iter=5000)
        model = LinearSVC(penalty='l2', dual=True, C=c, max_iter=3000)
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
    model = OneVsRestClassifier(LinearSVC(penalty='l2', dual=True, C=c_weights[max_index], max_iter=3000))
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
    arr = []
    for i, item in enumerate(y_test):
        arr_to_add = []
        if item == 0:
            arr_to_add = [1, 0, 0]
        if item == 1:
            arr_to_add = [0, 1, 0]
        if item == 2:
            arr_to_add = [0, 0, 1]
        arr.append(arr_to_add)
    y_t = np.asarray(arr)
    arr = []
    y_pred_temp = y_pred.reshape((y_pred.shape[0], 1))
    for i, item in enumerate(y_pred_temp):
        arr_to_add = []
        if item == 0:
            arr_to_add = [1, 0, 0]
        if item == 1:
            arr_to_add = [0, 1, 0]
        if item == 2:
            arr_to_add = [0, 0, 1]
        arr.append(arr_to_add)
    y_p = np.asarray(arr)
    print(f'[INFO] Done')
    print(f'[INFO] Printing classification report and confusion matrix')
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_t.argmax(axis=1), y_p.argmax(axis=1)))
    print(f'[INFO] Done')
    print(f'[INFO] Printing prediction accuracy')
    #| Get all instances where predicted values match the test
    #| resulting arrays are boolean
    num_match_zero = y_t[:, 0] == y_p[:, 0]
    num_match_one = y_t[:, 1] == y_p[:, 1]
    num_match_two = y_t[:, 2] == y_p[:, 2]
    # count = np.count_nonzero(arr)
    #| \t == 4 white spaces
    print(f'\t   Number of class 0 matches out of {len(y_test)} is: {np.count_nonzero(num_match_zero)}'
          f'\n\t   Model correctly predicts class 0 {((np.count_nonzero(num_match_zero) / len(y_test)) * 100):.2f}% '
          f'of the time')
    print(f'\t   Number of class 1 matches out of {len(y_test)} is: {np.count_nonzero(num_match_one)}'
          f'\n\t   Model correctly predicts class 1 {((np.count_nonzero(num_match_one) / len(y_test)) * 100):.2f}% '
          f'of the time')

    print(f'\t   Number of class 2 matches out of {len(y_test)} is: {np.count_nonzero(num_match_two)}'
          f'\n\t   Model correctly predicts class 1 {((np.count_nonzero(num_match_two) / len(y_test)) * 100):.2f}% '
          f'of the time')
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
extract_csv(CSV_PATH)
print(f"Total number of dataset: {len(x)}")
dummy_classifier(x, y)
# svm(x, y)
svm_cross_val(x, y)
print(f'[INFO] Done')