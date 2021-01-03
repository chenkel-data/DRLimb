import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10, fashion_mnist, imdb, mnist
from tensorflow.keras.preprocessing.sequence import pad_sequences

from collections import Counter
from imblearn.datasets import make_imbalance


def create_data(X_train: np.ndarray, y_train: np.ndarray,  X_test: np.ndarray, y_test: np.ndarray, min_class: list,
                maj_class: list, imb_ratio: float = 0.1,
                val_size: float = 0.2, imbalance: bool = True):
    """
    Creates Training, Validataion and Test data. If 'imbalance' is True, an imbalance is created.

    Args:
        X_train: The Training data
        y_train: the Training class labels
        X_test: The Test data
        y_test: The Test class labels
        min_class: minority class
        maj_class: majority class
        imb_ratio: ration for imbalance
        val_size: size of validation data
        imbalance: boolean indicating when creating an imbalance

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """

    # create a single label for all  majority classes (eg in mnist you have maj_class = [0, 1, 3, 4, 5, 6, 7, 8, 9])
    y_train = collapse_data(y_train, maj_class)
    y_test = collapse_data(y_test, maj_class)
    maj_class = [0]  # new collapsed major class is 0
    min_class = [1]

    if imbalance:
        X_train, y_train = create_imbalance(X_train, y_train, min_class, maj_class, imb_ratio=imb_ratio)

    # stratify to ensure class balance is kept between train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def collapse_data(y, maj_class):

    for i, value in enumerate(y):
        if value in maj_class:
            y[i] = 0
        else:
            y[i] = 1

    return y


def create_imbalance(X, y, min_class, maj_class, imb_ratio, verbose=True):
    """
    Create artificially an imbalance of (balanced) data
    """
    # get samples for each class if original total number of samples is unknown (eg. 12500 for IMDB)

    X_min, X_maj = [], []
    for i, value in enumerate(y):
        if value in min_class:
            X_min.append(X[i])
        if value in maj_class:
            X_maj.append(X[i])

    maj_cardinality = len(X_maj)  # samples of majority class
    min_count = int(maj_cardinality * imb_ratio)  # desired number of samples of minority class with ratio imb_ratio

    # need to reshape for images as 'make_imbalance' expects X to be a 2d-array.
    X_orig = X
    if len(list(X.shape)) > 2:
        X = X.reshape(X.shape[0], -1)

    X_res, y_res = make_imbalance(X, y,
                                  sampling_strategy={min_class[0]: min_count, maj_class[0]: maj_cardinality},
                                  random_state=42, verbose=True)

    # reshape backwards to original shape
    if len(list(X.shape)) > 2:
        X_res = X_res.reshape(X_res.shape[0], X_orig.shape[1], X_orig.shape[2], X_orig.shape[3])

    if verbose:
        print("min_class is: ", min_class)
        print("maj_class is: ", maj_class)
        print('Distribution before imbalancing: {}'.format(Counter(y)))
        print('Distribution after imbalancing: {}'.format(Counter(y_res)))

    return X_res, y_res


def load_image(source: str):
    """
    Loads the image datasets: mnist, famnist, cifar10.
    :return:  (X_train, y_train, X_test, y_test)
    """
    reshape_shape = -1, 28, 28, 1

    if source == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif source == "famnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    elif source == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        reshape_shape = -1, 32, 32, 3

    else:
        raise ValueError("Specify a valid source.")

    X_train = X_train.reshape(reshape_shape).astype(np.float32)
    X_test = X_test.reshape(reshape_shape).astype(np.float32)

    X_train /= 255
    X_test /= 255

    y_train = y_train.reshape(y_train.shape[0], ).astype(np.int32)
    y_test = y_test.reshape(y_test.shape[0], ).astype(np.int32)

    return X_train, y_train, X_test, y_test


def load_credit(source: str = "../../data/creditcard.csv", test_size: int = 0.5):
    """
    load creditcard csv file from kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud
    Needs to be split in train and test file
    """

    df = pd.read_csv(source)

    X_train, X_test = train_test_split(df, test_size=test_size, stratify=df["Class"])

    y_train = X_train[["Class"]].values.astype(np.int32)
    y_test = X_test[["Class"]].values.astype(np.int32)

    X_train.drop(columns=["Class"], inplace=True)
    X_test.drop(columns=["Class"], inplace=True)

    X_train = X_train.values
    X_test = X_test.values

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    return X_train, y_train, X_test, y_test