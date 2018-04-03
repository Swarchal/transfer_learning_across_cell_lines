"""
module docstring
"""

import os
from collections import Counter, namedtuple
import numpy as np
import pandas as pd
import sklearn.metrics


def get_cell_line(name):
    """get cell-line name from csv filename"""
    return name.split("_")[0]


def create_data_dict(directory):
    """create dictionary of {cell_line: csv}"""
    all_csvs = [i for i in os.listdir(directory) if i.endswith(".csv")]
    return {get_cell_line(i): pd.read_csv(os.path.join(directory, i)) for i in all_csvs}


def get_actual_predicted(dataframe):
    output = namedtuple("output", ["actual", "predicted"])
    actual, predicted = [], []
    for name, group in dataframe.groupby(["actual", "img_num"]):
        actual_class = group["actual"].unique()[0]
        predicted_class = Counter(group["predicted"]).most_common()[0][0]
        actual.append(actual_class)
        predicted.append(predicted_class)
    return output(actual, predicted)


def get_class_labels(dataframe):
    """
    return class labels in the same order as
    sklearn.metrics.confusion_matrix
    """
    predicted_labels = dataframe.predicted.unique()
    actual_labels = dataframe.actual.unique()
    all_labels = list(set(predicted_labels).union(set(actual_labels)))
    return sorted(all_labels)


def get_accuracy(dataframe):
    """determine classification accuracy"""
    actual, predicted = get_actual_predicted(dataframe)
    return sklearn.metrics.accuracy_score(actual, predicted)


def make_cm(dataframe, norm=True):
    """create a confusion matrix"""
    confusion_matrix = namedtuple("ConfusionMatrix", ["cm", "labels"])
    result = get_actual_predicted(dataframe)
    labels = get_class_labels(dataframe)
    cm = sklearn.metrics.confusion_matrix(result.actual, result.predicted)
    if norm:
        # normalise confusion matrix
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    return confusion_matrix(cm, labels)




def main(path=None):
    if path is None:
        path = sys.arv[1]
    # TODO
    pass


if __name__ == "__main__":
    main()
