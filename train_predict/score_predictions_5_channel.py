"""
module docstring
"""

from collections import Counter, namedtuple
import json
import numpy as np
import pandas as pd
import sklearn.metrics


def parse_table(path, predicted_col_name="predicted", actual_col_name="actual",
                img_name_col="img_name"):
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
    grouped = table.groupby(img_name_col)
    for _, group in grouped:
        unique_actual_labels = group[actual_col_name].unique()
        # each parent image should only have one MoA class for child objects
        assert len(unique_actual_labels) == 1
        actual_class = unique_actual_labels[0]
        predicted_class = consensus(group[predicted_col_name])
        actual.append(actual_class)
        predicted.append(predicted_class)
    return result(actual, predicted)


def accuracy(actual, predicted):
    """return accuracy"""
    return sklearn.metrics.accuracy_score(actual, predicted)


def consensus(labels):
    """return the most common label"""
    return Counter(labels).most_common()[0][0]


def make_confusion_matrix(actual, predicted, norm=True):
    """
    create a confusion matrix and class labels

    Returns:
    ---------
    namedtuple(cm, labels)
        cm: numpy array
            confusion matrix
        labels: list
            class labels in correct order
    """
    confusion_matrix_container = namedtuple("ConfusionMatrix", ["cm", "labels"])
    cm = sklearn.metrics.confusion_matrix(actual, predicted)
    if norm:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    return confusion_matrix_container(cm labels)


