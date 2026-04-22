

"""
model.py - Defines the neural network architecture for Federated Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.

    Architecture:
        - Convolutional layer: 1 input channel, 32 output channels, 3x3 kernel, stride 1, padding 1.
        - ReLU activation and 2x2 max pooling.
        - Convolutional layer: 32 input channels, 64 output channels, 3x3 kernel, stride 1, padding 1.
        - ReLU activation and 2x2 max pooling.
        - Flatten layer.
        - Fully connected layer: 64 * 7 * 7 -> 128.
        - ReLU activation.
        - Fully connected layer: 128 -> 10 (output classes).
    """


    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


