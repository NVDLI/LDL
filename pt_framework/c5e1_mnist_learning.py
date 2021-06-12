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
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
torch.manual_seed(7)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 1

# Load training dataset into a single batch to compute mean and stddev.
transform = transforms.Compose([transforms.ToTensor()])
trainset = MNIST(root='./pt_data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True)
data = next(iter(trainloader))
mean = data[0].mean()
stddev = data[0].std()

# Helper function needed to standardize data when loading datasets.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, stddev)])

trainset = MNIST(root='./pt_data', train=True, download=True, transform=transform)
testset = MNIST(root='./pt_data', train=False, download=True, transform=transform)

# Create a Sequential (feed-forward) model.
# 784 inputs.
# Two fully-connected layers with 25 and 10 neurons.
# tanh as activation function for hidden layer.
# Logistic (sigmoid) as activation function for output layer.
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 25),
    nn.Tanh(),
    nn.Linear(25, 10),
    nn.Sigmoid()
)

# Initialize weights.
for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, a=-0.1, b=0.1)
        nn.init.constant_(module.bias, 0.0)

# Use stochastic gradient descent (SGD) with
# learning rate of 0.01 and no other bells and whistles.
# MSE as loss function.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()

# Transfer model to GPU
model.to(device)

# Create DataLoader objects that will help create mini-batches.
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# Train the model. In PyTorch we have to implement the training loop ourselves.
for i in range(EPOCHS):
    model.train() # Set model in training mode.
    train_loss = 0.0
    train_correct = 0
    train_batches = 0
    for inputs, targets in trainloader:
        # Move data to GPU.
        one_hot_targets = nn.functional.one_hot(targets, num_classes=10).float()
        inputs, targets, one_hot_targets = inputs.to(device), targets.to(device), one_hot_targets.to(device)

        # Zero the parameter gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(inputs)
        loss = loss_function(outputs, one_hot_targets)

        # Accumulate metrics.
        _, indices = torch.max(outputs.data, 1)
        train_correct += (indices == targets).sum().item()
        train_batches +=  1
        train_loss += loss.item()

        # Backward pass and update.
        loss.backward()
        optimizer.step()

    train_loss = train_loss / train_batches
    train_acc = train_correct / (train_batches * BATCH_SIZE)

    # Evaluate the model on the test dataset. Identical to loop above but without
    # weight adjustment.
    model.eval() # Set model in inference mode.
    test_loss = 0.0
    test_correct = 0
    test_batches = 0
    for inputs, targets in testloader:
        one_hot_targets = nn.functional.one_hot(targets, num_classes=10).float()
        inputs, targets, one_hot_targets = inputs.to(device), targets.to(device), one_hot_targets.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, one_hot_targets)
        _, indices = torch.max(outputs, 1)
        test_correct += (indices == targets).sum().item()
        test_batches +=  1
        test_loss += loss.item()

    test_loss = test_loss / test_batches
    test_acc = test_correct / (test_batches * BATCH_SIZE)

    print(f'Epoch {i+1}/{EPOCHS} loss: {train_loss:.4f} - acc: {train_acc:0.4f} - val_loss: {test_loss:.4f} - val_acc: {test_acc:0.4f}')
