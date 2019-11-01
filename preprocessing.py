import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import skimage.io as skm


def load_data(root="dataset/chars74k-lite"):
    imgs, labels = list(), list()
    for root, dirs, files in os.walk(root, topdown=False):
        for f in files:
            if "LICENSE" not in f:
                # read image with shape(20, 20)
                img = skm.imread(os.path.join(root, f))
                # run image through Histogram of Oriented Gradients
                # shape (576, )
                hog_img = hog(img, pixels_per_cell=(4, 4),
                              cells_per_block=(2, 2))
                imgs.append(hog_img)
                # add image label
                labels.append(root[-1])
    print("Loaded data successfully...")
    return np.array(imgs), np.array(labels)


def feature_scaling(X, y):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X, y)
    print("Features scaled...")
    return X_s


# currently not used
def feature_selection(X, y, p=0):
    # alternative 1
    # feature selection based on k best features
    if p == 0:
        X_f = SelectKBest(chi2, k=200).fit_transform(X, y)
    # alternative 2
    # select features according to a percentile of the highest scores
    else:
        X_f = SelectPercentile(chi2, percentile=p).fit_transform(X, y)
    print("feature selection...")

    return X_f


def init_data():
    X_init, y = load_data()
    # X = feature_selection(X_init, y, 70)
    X = feature_scaling(X_init, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)
    print("data split...")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    init_data()
