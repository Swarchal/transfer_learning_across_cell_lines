"""
author: Scott Warchal
date: 2018-05-17

Use a pre-trained network to predict MoA labels on 5 channel numpy arrays.

arg1: path to directory containing test and train subdirectories
arg2: path to model checkpoint
"""

import os
import sys
from collections import OrderedDict, deque
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
        # as models were trained on a GPU
        model_state = torch.load(path_to_state_dict)
    else:
        model_state = torch.load(path_to_state_dict,
                                 map_location=lambda storage, loc: "cpu")
    # if the state_dict was trained across multiple GPU's then the state_dict
    # keys are prefixed with 'module.', which will not match the keys
    # of the new model, when we try to load the model state,
    # so these need to be removed
    if all(k.startswith("module.") for k in model_state.keys()):
        new_state_dict = OrderedDict()
        for key, value in model_state.items():
            key = key[7:]  # skip "module." in key name
            new_state_dict[key] = value
        model_state = new_state_dict
    model.load_state_dict(model_state)
    model.eval()
    if use_gpu:
        model = model.cuda()
    return model


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


def main():
    """docstring"""
    data_dir, path_to_weights = sys.argv[1:]
    model = resnet.resnet18(num_classes=NUM_CLASSES)
    model = load_model_weights(model, path_to_weights, use_gpu=USE_GPU)
    dataset = data_utils.make_datasets(data_dir, return_name=True)
    dataloader = data_utils.make_dataloaders(dataset)["test"]
    label_dict = make_label_dict(data_dir)
    names, actual_labels, predicted_labels = deque(), deque(), deque()
    predicted_labels = []
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
        names.extend(img_names)
        batch_actual_labels = [label_dict[i.data[0]] for i in list(labels)]
        actual_labels.extend(batch_actual_labels)
        batch_predicted_labels = [label_dict[i] for i in list(preds)]
        predicted_labels.append(batch_predicted_labels)
    # TODO: get image names from DataLoader
    # so it's possible to record the parent image the prediction is from
    # to construct a consensus classification of the overall image
    return list(zip(predicted_labels, actual_labels, names))


if __name__ == "__main__":
    print(main())
