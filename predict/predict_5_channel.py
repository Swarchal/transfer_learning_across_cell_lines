"""module docstring"""

import sys
from collections import OrderedDict
import torch
from torch.autograd import Variable
import dataset
import train
import resnet

def load_model_weights(model, path_to_state_dict, cuda=True):
    """
    docstring

    Parameters:
    ----------
    model: pytorch Model
    path_to_state_dict: string

    Returns:
    ---------
    pytorch model with weights loaded from state_dict
    """
    model_state = torch.load(path_to_state_dict)
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
    if cuda:
        model = model.cuda()
    return model


if __name__ == "__main__":
    # TODO: load model
    model = resnet.resnet18(num_classes=8)
    data_dir = sys.argv[1]
    datasets = make_datasets(data_dir)
    dataloader = make_dataloader(datasets)
    test_dataloader = dataloader["test"]
    for index, data in enumerate(test_dataloader):
        if USE_GPU:
            inputs = torch.autograd.Variable(inputs.cuda())
            labels = torch.autograd.Variable(inputs.cuda())
        else:
            inputs = torch.autograd.Variable(inputs)
            labels = torch.autograd.Variable(inputs)
        outputs = model(inputs)
        _, preds = torch.max(outputs,data, 1)
        labels = labels.view(-1)
        