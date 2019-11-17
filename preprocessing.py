import os
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import skimage.io as skm


def load_data(path="dataset/chars74k-lite"):
    images, labels = list(), list()
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            if "LICENSE" not in f:
                image = skm.imread(os.path.join(root, f))
                images.append(image)
                labels.append(ord(root[-1]) - 97)
    print("Loaded classification data successfully...")
    return np.array(images), np.array(labels)


def _apply_normalization(X):
    X_n = []
    for image in X:
        image = (image - np.min(image)) / (np.ptp(image) + 1e-6)
        X_n.append(image)
    return np.array(X_n)


def _apply_hog(X):
    X_h = []
    for image in X:
        hog_image = hog(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
        X_h.append(hog_image)
    return np.array(X_h)


def _apply_augmentation(X, Y):
    X_a = X.copy()
    Y_a = Y.copy()
    # Counter-clockwise
    X_a = np.concatenate((X_a, np.rot90(X, axes=(1, 2))), axis=0)
    Y_a = np.concatenate((Y_a, Y), axis=0)
    # Clockwise
    X_a = np.concatenate((X_a, np.rot90(X, axes=(2, 1))), axis=0)
    Y_a = np.concatenate((Y_a, Y), axis=0)
    # Flip black/white colors
    X_a = np.concatenate((X_a, -X))
    Y_a = np.concatenate((Y_a, Y), axis=0)
    return X_a, Y_a


def preprocess(X, Y=None):
    if Y is not None:
        X, Y = _apply_augmentation(X, Y)
    # X = _apply_normalization(X)
    X = _apply_hog(X)
    if Y is not None:
        return X, Y
    return X


def init_data():
    X, Y = load_data()
    X, Y = preprocess(X, Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=None
    )
    print("Split data successfully...")
    return X_train, X_test, Y_train, Y_test
