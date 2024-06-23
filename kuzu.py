"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self, input_size=28*28, hidden_size=130, num_classes=10):
        super(NetFull, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # 1 input channel, 32 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)  # 32 input channels, 64 output channels, 5x5 kernel
        self.fc1 = nn.Linear(64*4*4, 128)  # Fully connected layer with 128 units
        self.fc2 = nn.Linear(128, 10)  # Output layer with 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Conv1 + ReLU
        x = F.max_pool2d(x, 2)  # MaxPool1
        x = F.relu(self.conv2(x))  # Conv2 + ReLU
        x = F.max_pool2d(x, 2)  # MaxPool2
        x = x.view(-1, 64*4*4)  # Flatten
        x = F.relu(self.fc1(x))  # FC1 + ReLU
        x = self.fc2(x)  # FC2
        return F.log_softmax(x, dim=1)  # LogSoftmax

