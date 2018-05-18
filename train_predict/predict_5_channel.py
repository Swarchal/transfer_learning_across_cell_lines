"""
author: Scott Warchal
date: 2018-05-17

Use a pre-trained network to predict MoA labels on 5 channel numpy arrays.

arg1: path to directory containing test and train subdirectories
arg2: path to model checkpoint
"""

import os
import sys
from collections import OrderedDict
import torch
import resnet
import data_utils

USE_GPU = torch.cuda.is_available()
NUM_CLASSES = 8


def load_model_weights(model, path_to_state_dict, use_gpu=True):
    """
    Load a model with a given state (pre-trained weights) from
    a saved checkpoint.

    Parameters:
    ----------
    model: pytorch Model
    path_to_state_dict: string

    Returns:
    ---------
    pytorch model with weights loaded from state_dict
    """
    if USE_GPU:
        # models were trained on a GPU, so good to go
        model_state = torch.load(path_to_state_dict)
    else:
        # need to map storage loc to 'cpu' if not using GPUs for prediction
        model_state = torch.load(path_to_state_dict,
                                 map_location=lambda storage, loc: "cpu")
    if is_distributed_model(model_state):
        model_state = strip_distributed_keys(model_state)
    model.load_state_dict(model_state)
    model.eval()
    if use_gpu:
        model = model.cuda()
    return model


def is_distributed_model(state_dict):
    """
    determines if the state dict is from a model trained on distributed GPUs
    """
    return all(k.startswith("module.") for k in state_dict.keys())


def strip_distributed_keys(state_dict):
    """
    if the state_dict was trained across multiple GPU's then the state_dict
    keys are prefixed with 'module.', which will not match the keys
    of the new model, when we try to load the model state
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        key = key[7:]  # skip "module." in key name
        new_state_dict[key] = value
    return new_state_dict


def make_label_dict(data_dir):
    """
    docstring

    Parameters:
    -----------
    data_dir: string
        path to directory containing sub-directories of classes and data
    
    Returns:
    ---------
    dictionary:
        {index => int: class_label => string}
    """
    path = os.path.join(data_dir, "test")
    dataset = data_utils.CellDataset(path)
    # reverse {label: index} dict used within CellDataset class
    return {v: k for k, v in dataset.label_dict.items()}


def parse_name(name):
    """
    extract the important stuff from the array filename.
    Returning the parent image the file is from.

    Parameters:
    ------------
    name: string
        file path of the numpy array
    
    Returns:
    ---------
    string:
        e.g "MCF7_img_13_90.npy"
        will return "MCF7_13"
    """
    # remove the file suffix
    assert name.endswith(".npy")
    cell_line, _, img_num, _ = name.split(".")[0].split("_")
    return "_".join([cell_line, img_num])


def main():
    """docstring"""
    data_dir, path_to_weights = sys.argv[1:]
    model = resnet.resnet18(num_classes=NUM_CLASSES)
    model = load_model_weights(model, path_to_weights, use_gpu=USE_GPU)
    dataset = data_utils.make_datasets(data_dir, return_name=True)
    dataloader = data_utils.make_dataloaders(dataset)["test"]
    label_dict = make_label_dict(data_dir)
    print("predicted", "actual", "img_name", sep="\t")
    for batch in dataloader:
        inputs, labels, img_names = batch
        if USE_GPU:
            inputs = torch.autograd.Variable(inputs.cuda())
            labels = torch.autograd.Variable(labels.cuda())
        else:
            inputs = torch.autograd.Variable(inputs)
            labels = torch.autograd.Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        labels = labels.view(-1)
        parsed_img_names = [parse_name(i) for i in img_names]
        batch_actual_labels = [label_dict[i.data[0]] for i in list(labels)]
        batch_predicted_labels = [label_dict[i] for i in list(preds)]
        for predicted, actual, img_name in zip(batch_predicted_labels,
                                               batch_actual_labels,
                                               parsed_img_names):
            print(predicted, actual, img_name, sep="\t", flush=True)


if __name__ == "__main__":
    main()
