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
    # y_pred_int = [ord(char) - 97 for char in y_pred]
    # y_test_int = [ord(char) - 97 for char in y_test]

    # some spaghetti à la capri to find accuarcy for each character
    predictions = {}
    for i in range(26):
        predictions[i] = (0, 0)

    for pred, y in zip(y_pred, y_test):
        if pred == y:
            predictions[y] = (
                predictions[y][0] + 1, predictions[y][1] + 1)
        else:
            predictions[y] = (
                predictions[y][0] + 1, predictions[y][1])
    for k, v in predictions.items():
        print("{}: {}".format(chr(k + 97),  v[1] / v[0]))


def clf_knn(X_train, X_test, y_train):
    '''
    knn classifier
    '''
    clf_knn = KNeighborsClassifier(n_neighbors=10)
    clf_knn.fit(X_train, y_train)
    y_pred_knn = clf_knn.predict(X_test)
    print("Predicted correctly using k-nn: {}"
          .format(evaluate(y_test, y_pred_knn)))
    return y_pred_knn


def clf_keras(X_train, X_test, y_train, y_test):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(576,)))
    model.add(Dense(26, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=15, batch_size=64)
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print(test_loss)
    print(test_acc)


def clf_svm(X_train, X_test, y_train):
    '''
    svm classifier
    '''
    clf_svm = svm.SVC(gamma='scale')
    clf_svm.fit(X_train, y_train)
    y_pred_svm = clf_svm.predict(X_test)
    print("Predicted correctly using svm: {}"
          .format(evaluate(y_test, y_pred_svm)))
    return y_pred_svm


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = init_data()

    # run knn
    # y_pred = clf_knn(X_train, X_test, y_train)

    # run svm
    # y_pred = clf_svm(X_train, X_test, y_train)

    # run ann
    # y_pred = clf_ann(X_train, X_test, y_train)

    # run keras
    clf_keras(X_train, X_test, y_train, y_test)

    # plot error
    # print("\nError:")
    # err = calculate_error(y_pred, y_test)
    # clf_keras(X_train, X_test, y_train)
