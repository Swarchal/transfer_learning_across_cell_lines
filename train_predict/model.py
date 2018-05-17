"""
author: Scott Warchal
date: 2018-04-22

A simple example of a model that works on a 5 channel image
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init


class Net(nn.Module):
    def __init__(self, n_classes=8):
        super(Net, self).__init__()
        self.in_channels = 5 # number of wavelengths / channels
        self.out_channels = 20
        self.n_kernels = 16
        self.n_classes = n_classes
        # architecture
        self.conv1 = nn.Conv2d(self.in_channels,
                               self.out_channels,
                               self.n_kernels)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.out_channels,
                               self.out_channels,
                               self.n_kernels*2) # double n_kernels in 2nd conv
        self.fc1 = nn.Linear(self.out_channels * 55 * 55, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.n_classes)

    def forward(self, x, debug=False):
        x = self.pool(F.relu(self.conv1(x)))
        if debug:
            print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        if debug:
            print(x.size())
        x = x.view(-1, self.out_channels * 55 * 55)
        if debug:
            print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

