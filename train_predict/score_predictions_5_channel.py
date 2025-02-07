"""
module docstring
"""

from collections import Counter, namedtuple
import numpy as np
import pandas as pd
import sklearn.metrics


def parse_table(path, predicted_col_name="predicted", actual_col_name="actual",
                img_col_name="img_name"):
    """
    read in tab-delimited table, parse into actual vs consensus prediction
    for each overall image

    Parameters:
    ----------
    path: string
    predicted_col_name: string
    actual_col_name: string
    img_col_name: string

    Returns:
    ---------
    namedtuple of two lists:
        Output([actual], [predicted])
    """
    result = namedtuple("Output", ["actual", "predicted"])
    actual, predicted = [], []
    table = pd.read_table(path)
    # as the image names are duplicated for each MoA class, need to create
    # column of unique names for each image by joining the MoA class label
    # with the image name.
    table["unique_img_name"] = table.apply(
        lambda x: "_".join([x[actual_col_name], x[img_col_name]]),
        axis=1)
    grouped = table.groupby("unique_img_name")
    for _, group in grouped:
        unique_actual_labels = group[actual_col_name].unique()
        # each parent image should only have one MoA class for child objects
        assert len(unique_actual_labels) == 1
        actual_class = unique_actual_labels[0]
        predicted_class = consensus(group[predicted_col_name])
        actual.append(actual_class)
        predicted.append(predicted_class)
    return result(actual, predicted)


def consensus(predicted_labels):
    """return the most common label"""
    return Counter(predicted_labels).most_common()[0][0]


def get_class_labels(labels):
    """
    return a list of alphabetically sorted class labels, the same order
    as the confusion matrix labels
    """
    return sorted(list(set(labels)))


def make_confusion_matrix(actual, predicted, norm=True):
    """
    create a confusion matrix and class labels

    Returns:
    ---------
    namedtuple(cm, labels, acc)
        cm: numpy array
            confusion matrix
        labels: list
            class labels in correct order
        acc: float
            classification accuracy as determined by
            sklearn.metrics.accuracy_score
    """
    confusion_matrix_container = namedtuple("ConfusionMatrix",
                                            ["cm", "labels", "acc"])
    cm = sklearn.metrics.confusion_matrix(actual, predicted)
    labels = get_class_labels(actual)
    acc = sklearn.metrics.accuracy_score(actual, predicted)
    if norm:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    return confusion_matrix_container(cm, labels, acc)

