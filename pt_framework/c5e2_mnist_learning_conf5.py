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
BATCH_SIZE = 64

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

# Define model. Final activation is omitted since softmax is part of
# cross-entropy loss function in PyTorch.
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 25),
    nn.ReLU(),
    nn.Linear(25, 10)
)

# Retrieve layers for custom weight initialization.
layers = next(model.modules())
hidden_layer = layers[1]
output_layer = layers[3]

# Kaiming (He) initialization.
nn.init.kaiming_normal_(hidden_layer.weight)
nn.init.constant_(hidden_layer.bias, 0.0)

# Xavier (Glorot) initialization.
nn.init.xavier_uniform_(output_layer.weight)
nn.init.constant_(output_layer.bias, 0.0)

# Use the Adam optimizer
# Cross-entropy loss as loss function.
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

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
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(inputs)
        # Cross-entropy loss does not need one-hot targets in PyTorch.
        loss = loss_function(outputs, targets)

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
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        _, indices = torch.max(outputs, 1)
        test_correct += (indices == targets).sum().item()
        test_batches +=  1
        test_loss += loss.item()

    test_loss = test_loss / test_batches
    test_acc = test_correct / (test_batches * BATCH_SIZE)

    print(f'Epoch {i+1}/{EPOCHS} loss: {train_loss:.4f} - acc: {train_acc:0.4f} - val_loss: {test_loss:.4f} - val_acc: {test_acc:0.4f}')
