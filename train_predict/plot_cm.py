"""
author: Scott Warchal
date  : 2018-05-28


Plot confusion matrices from score_*.json files.
"""

import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(file_path):
    """path to json, returns a dict"""
    with open(file_path, "r") as f:
        data = json.load(f)
    # get cell-line from the file_path
    cell_line = file_path.split(os.sep)[-1].split("_")[1].split(".")[0]
    data["cell_line"] = cell_line
    return data


def plot_cm(score_dict):
    """docstring"""
    cm_list = score_dict["cm"]
    cm_arr = np.asarray(cm_list)
    accuracy = float(score_dict["accuracy"]) * 100
    labels = score_dict["labels"]
    labels = [i.replace("_", " ") for i in labels]
    cell_line = score_dict["cell_line"]
    plt.figure(figsize=[8, 6])
    plt.grid(linestyle=":")
    plt.imshow(cm_arr, vmin=0, vmax=1, cmap=plt.cm.bone_r)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted MoA")
    plt.ylabel("True MoA")
    plt.title("{}\nAccuracy = {:.2f}%".format(cell_line, accuracy))
    save_path = "{}_confusion_matrix.pdf".format(score_dict["cell_line"])
    plt.tight_layout()
    plt.savefig(save_path)
    print("saved at '{}'".format(save_path))



def main():
    json_path = sys.argv[1]
    plot_cm(read_json(json_path))


if __name__ == "__main__":
    main()

