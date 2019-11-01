import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from preprocessing import init_data


def evaluate(y_true, y_pred):
    accuracy_count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            accuracy_count += 1
    accuracy = accuracy_count/len(y_true)
    return accuracy


def calculate_error(y_pred, y_test):
    # convert from char to int. a=0, b=1 etc.
    y_pred_int = [ord(char) - 97 for char in y_pred]
    y_test_int = [ord(char) - 97 for char in y_test]

    # some spaghetti à la capri to find accuarcy for each character
    predictions = {}
    for i in range(26):
        predictions[i] = (0, 0)

    for pred, y in zip(y_pred_int, y_test_int):
        if pred == y:
            predictions[y] = (
                predictions[y][0] + 1, predictions[y][1] + 1)
        else:
            predictions[y] = (
                predictions[y][0] + 1, predictions[y][1])
    for k, v in predictions.items():
        print("{}: {}".format(chr(k + 97),  v[1] / v[0]))


def clf_knn():
    '''
    knn classifier
    '''
    clf_knn = KNeighborsClassifier(n_neighbors=10)
    clf_knn.fit(X_train, y_train)
    y_pred_knn = clf_knn.predict(X_test)
    print("Predicted correctly using k-nn: {}"
          .format(evaluate(y_test, y_pred_knn)))
    return y_pred_knn


def clf_svm():
    '''
    svm classifier
    '''
    clf_svm = svm.SVC(gamma='scale')
    clf_svm.fit(X_train, y_train)
    y_pred_svm = clf_svm.predict(X_test)
    print("Predicted correctly using svm: {}"
          .format(evaluate(y_test, y_pred_svm)))
    return y_pred_svm


def clf_ann():
    '''
    multi-layer perceptron classifier
    '''
    clf_ann = MLPClassifier(solver='adam', max_iter=1000,
                            hidden_layer_sizes=(100))
    clf_ann.fit(X_train, y_train)
    y_pred_ann = clf_ann.predict(X_test)
    print("Predicted correctly using ann: {}"
          .format(evaluate(y_test, y_pred_ann)))
    return y_pred_ann


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = init_data()

    # run knn
    # y_pred = clf_knn()

    # run svm
    y_pred = clf_svm()

    # run ann
    # y_pred = clf_ann()

    # plot error
    print("\nError:")
    err = calculate_error(y_pred, y_test)
