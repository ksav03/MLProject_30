from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV

DATA_SPLIT = 0.3
DATAFILES = ['dataset\labelled_dataset_33.csv']


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
    arr = []
    for item in x4:
        counter = 0
        while item >= 60:
            item = item - 60
            counter += 1
        if item < 10:
            arr.append(float(str(counter)+".0"+str(item)))
        else:
            arr.append(float(str(counter) + "." + str(item)))
    arr = np.asarray(arr)

    x7 = main_data_frame.iloc[:n, 7]  # 7th feature: Ratio Bike / Stand
    x = np.column_stack((x0, x2, x3, arr, x7))
    y = main_data_frame.iloc[:n, 8]  # Third column as co-responding labels
    return x, y


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


def confusion_matrix_visual(model, cm, title):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()


def svm(features, labels):
    print(f'\n[INFO] Function "{svm.__name__}" running...')
    print(f'[INFO] Splitting data...')
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=DATA_SPLIT)
    #| There doesn't seem to be a big difference between the next two lines
    # model = LinearSVC(penalty='l2', dual=True, C=0.66, max_iter=5000)
    print(f'[INFO] Creating SVM model...')
    svm_model = LinearSVC(penalty='l2', C=0.1, dual=False, max_iter=5000, class_weight='balanced', multi_class='ovr')
    print(f'[INFO] Fitting SVM model...')
    svm_model = CalibratedClassifierCV(svm_model, method='isotonic', cv=5).fit(x_train, y_train)
    y_pred = svm_model.predict(x_test)

    y_score = svm_model.predict_proba(x_test)
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
    confusion_matrix_visual(svm_model, conf_mat, "SVM confusion matrix")
    print("Confusion matrix:\n", conf_mat)
    print(f'[INFO] Done')

    print(f'[INFO] Getting ROC and AUC...')
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_t[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print(f'[INFO] Plotting ROC and AUC...')
    colors = ['red', 'blue', 'green']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

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

    # c_weights = np.linspace(0.0001, 100, 200)
    c_weights = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 15, 20]
    print(f'[INFO] Splitting data...')
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=DATA_SPLIT)

    mean_error = []
    std_error = []
    print(f'[INFO] Running cross validation with parameters:\n\t   {c_weights=} and cv={5}')
    for c in c_weights:
        svm_model = LinearSVC(penalty='l2', C=c, dual=False, max_iter=5000, class_weight='balanced',
                              multi_class='ovr')
        svm_model = CalibratedClassifierCV(svm_model, method='isotonic', cv=5).fit(x_train, y_train)
        # model = LinearSVC(penalty='l2', dual=False, C=c, max_iter=6000, multi_class='ovr', class_weight='balanced')
        # model = LinearSVC(penalty='l2', dual=True, C=c, max_iter=6000, multi_class='ovr', class_weight='balanced')
        svm_model.fit(x_train, y_train)
        scores = cross_val_score(svm_model, x_test, y_test, cv=5, scoring='f1_weighted')
        # print(scores)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    # Plot cross-validation result for C parameter
    plt.figure(figsize=(9, 9))
    plt.errorbar(c_weights, mean_error, yerr=std_error, linewidth=3)
    # 18
    plt.title("Cross-validation to determine C", fontsize=20)
    plt.xlabel('C', fontsize=18)
    plt.ylabel('F1 Score', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    #| Get the index of the cross val with the highest f1 score
    max_value = max(mean_error)
    max_index = mean_error.index(max_value)
    print(f'[INFO] Optimum C value is C={c_weights[max_index]}')
    print(f'[INFO] Creating SVM model...')
    #| There doesn't seem to be a big difference between the next two lines
    # model = LinearSVC(penalty='l2', dual=True, C=0.66, max_iter=5000)
    svm_model = LinearSVC(penalty='l2', C=3, dual=False, max_iter=5000, class_weight='balanced',
                          multi_class='ovr')
    svm_model = CalibratedClassifierCV(svm_model, method='isotonic', cv=5).fit(x_train, y_train)
    # model = LinearSVC(penalty='l2', dual=False, C=3,
    #                   max_iter=3000, class_weight='balanced', multi_class='ovr')
    # model = LinearSVC(penalty='l2', dual=False, C=0.4, max_iter=3000)
    # model = LinearSVC(penalty='l2', multi_class='ovr', max_iter=5000)
    print(f'[INFO] Fitting SVM model...')
    svm_model.fit(x_train, y_train)
    y_pred = svm_model.predict(x_test)
    #| Get the probability of a prediction for ROC
    y_score = svm_model.predict_proba(x_test)
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
    print("Confusion matrix:\n", conf_mat)
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
# print(X_normalized)
# X, Y = extract_csv(CSV_PATH, DATAFILES)
# print(f"Total number of dataset: {len(X)}")
dummy_classifier(X, Y)
# svm(X, Y)
svm_cross_val(X, Y)
print(f'[INFO] Done')
