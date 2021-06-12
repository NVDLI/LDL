"""
The MIT License (MIT)

Copyright (c) 2021 NVIDIA

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
from utilities import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 128
BATCH_SIZE = 32

# Load training dataset into a single batch to compute mean and stddev.
transform = transforms.Compose([transforms.ToTensor()])
trainset = CIFAR10(root='./pt_data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
data = next(iter(trainloader))
mean = data[0].mean()
stddev = data[0].std()

# Load and standardize training and test dataset.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, stddev)])

trainset = CIFAR10(root='./pt_data', train=True, download=True, transform=transform)
testset = CIFAR10(root='./pt_data', train=False, download=True, transform=transform)

# Model with 2 convolutional and 1 fully-connected layer.
model = nn.Sequential(
    nn.Conv2d(3, 64, 4, stride=1, padding=1), # Output is 64x31x31.
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Conv2d(64, 64, 2, stride=2, padding=1), # Output is 64x16x16.
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Conv2d(64, 32, 3, stride=1, padding=1), # Output is 32x16x16.
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Conv2d(32, 32, 3, stride=1, padding=1), # Output is 32x16x16.
    nn.ReLU(),
    nn.MaxPool2d(2, 2), # Output is 32x8x8.
    nn.Flatten(),
    nn.Linear(32 * 8 * 8, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 10)
)

# Initialize weights with Xavier (Glorot) uniform for all weight layers.
for module in model.modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0.0)

# Loss function and optimizer
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

# Train the model.
train_model(model, device, EPOCHS, BATCH_SIZE, trainset, testset,
            optimizer, loss_function, 'acc')
