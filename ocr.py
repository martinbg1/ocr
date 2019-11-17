import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from preprocessing import init_data, preprocess
import keras
from keras.layers import Input, Dense, Dropout
from sklearn.utils import class_weight


def evaluate(y_true, y_pred):
    accuracy_count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            accuracy_count += 1
    accuracy = accuracy_count / len(y_true)
    return accuracy


def clf_knn(X_train, X_test, Y_train, Y_test):
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_train, Y_train)
    pred_knn = clf.predict(X_test)
    accuracy = evaluate(Y_test, pred_knn)
    print("KNN trained, accuracy: {}".format(accuracy))
    return clf


def clf_keras(X_train, X_test, Y_train, Y_test):
    class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(Y_train), Y_train
    )
    print("Class weights")
    print(class_weights)
    model_in = Input(X_train[0].shape)
    dense1 = Dense(512, activation="relu")(model_in)
    dropout1 = Dropout(0.25)(dense1)
    dense2 = Dense(256, activation="relu")(dropout1)
    dropout2 = Dropout(0.25)(dense2)
    dense3 = Dense(128, activation="relu")(dropout2)
    dense4 = Dense(64, activation="relu")(dense3)
    dense5 = Dense(32, activation="relu")(dense4)

    model_out = Dense(26, activation="softmax")(dense5)

    clf = keras.Model(inputs=model_in, outputs=model_out)

    clf.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    clf.summary()

    clf.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        # epochs=50,
        # batch_size=32,
        epochs=25,
        batch_size=128,
        shuffle=True,
        class_weight=class_weights,
    )
    return clf


def clf_svm(X_train, X_test, Y_train, Y_test):
    clf = svm.SVC(gamma="scale")
    clf.fit(X_train, Y_train)
    pred_svm = clf.predict(X_test)
    accuracy = evaluate(Y_test, pred_svm)
    print("SVM trained, accuracy: {}".format(accuracy))
    return clf


def clf_init(clf_func):
    data = init_data()
    clf = clf_func(*data)
    return clf


def clf_predict(clf, image):
    X = preprocess(image)
    prediction = clf.predict(X)
    if isinstance(clf, keras.Model):
        prediction += clf.predict(preprocess(-image))
        prediction += clf.predict(preprocess(np.rot90(image, axes=(1, 2))))
        prediction += clf.predict(preprocess(np.rot90(image, axes=(2, 1))))
        prediction = np.argmax(prediction)
    else:
        prediction = prediction[0]
    return prediction
