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
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 64

# Load MNIST dataset. In PyTorch it is in the range 0.0 - 1.0 so no need to
# rescale in this application.
transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = MNIST(root='./pt_data', train=True, download=True, transform=transform)
testset = MNIST(root='./pt_data', train=False, download=True, transform=transform)

# Create autoencoder model.
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 784),
    nn.Sigmoid(),
    nn.Unflatten(1, torch.Size([28, 28]))
)

# Create loss function and optimizer.
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.BCELoss()

# Training loop for autoencoder.

# Transfer model to GPU.
model.to(device)

# Create dataloaders.
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

for i in range(EPOCHS):
    model.train() # Set model in training mode
    train_loss = 0.0
    train_absolute_error = 0.0
    train_batches = 0
    for inputs, _ in trainloader:
        # Move data to GPU. Use same value for input and target because
        # we are training an auto-encoder.
        inputs = inputs.squeeze() # Needed to remove redundant dimension in inputs
        inputs, targets = inputs.to(device), inputs.to(device)

        # Zero the parameter gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        # Accumulate metrics.
        train_absolute_error += (targets - outputs.data).abs().sum().item()
        train_batches +=  1
        train_loss += loss.item()

        # Backward pass and update.
        loss.backward()
        optimizer.step()

    train_loss = train_loss / train_batches
    train_mae = train_absolute_error / (train_batches * BATCH_SIZE)

    # Evaluate the model on the test dataset.
    model.eval() # Set model in inference mode.
    test_loss = 0.0
    test_absolute_error = 0.0
    test_batches = 0
    for inputs, _ in testloader:
        # Use same value for input and target because we are training
        # an auto-encoder.
        inputs = inputs.squeeze()
        inputs, targets = inputs.to(device), inputs.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        test_absolute_error += (targets - outputs.data).abs().sum().item()
        test_batches += 1
        test_loss += loss.item()
    test_loss = test_loss / test_batches
    test_mae = test_absolute_error / (test_batches * BATCH_SIZE)
    print(f'Epoch {i+1}/{EPOCHS} loss: {train_loss:.4f} - mae: {train_mae:0.4f} - val_loss: {test_loss:.4f} - val_mae: {test_mae:0.4f}')

# Predict on test dataset.
testloader = DataLoader(dataset=testset, batch_size=10000, shuffle=False)
test_images, _ = next(iter(testloader))
test_images = test_images.to(device)
predict_images = model(test_images)

test_images = test_images.squeeze().cpu().numpy()
predict_images = predict_images.cpu().detach().numpy()

# Plot one input example and resulting prediction.
plt.subplot(1, 2, 1)
plt.imshow(test_images[0].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.subplot(1, 2, 2)
plt.imshow(predict_images[0].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.show()

# Load Fashion-MNIST dataset.
transform = transforms.Compose(
    [transforms.ToTensor()])

f_trainset = FashionMNIST(root='./pt_data', train=True, download=True, transform=transform)
f_testset = FashionMNIST(root='./pt_data', train=False, download=True, transform=transform)

# Predict and plot.
testloader = DataLoader(dataset=f_testset, batch_size=10000, shuffle=False)
f_test_images, _ = next(iter(testloader))
f_test_images = f_test_images.to(device)
f_predict_images = model(f_test_images)
f_test_images = f_test_images.squeeze().cpu().numpy()

f_predict_images = f_predict_images.cpu().detach().numpy()
plt.subplot(1, 2, 1)
plt.imshow(f_test_images[0].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.subplot(1, 2, 2)
plt.imshow(f_predict_images[0].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.show()

# Compute errors and plot.
error = np.mean(np.abs(test_images - predict_images), (1, 2))
f_error = np.mean(np.abs(f_test_images - f_predict_images), (1, 2))
_ = plt.hist((error, f_error), bins=50, label=['mnist',
                                               'fashion mnist'])
plt.legend()
plt.xlabel('mean absolute error')
plt.ylabel('examples')
plt.title("Autoencoder for outlier detection")
plt.show()

# Print outliers in mnist data.
index = error.argmax()
plt.subplot(1, 2, 1)
plt.imshow(test_images[index].reshape(28, 28), cmap=plt.get_cmap('gray'))
error[index] = 0
index = error.argmax()
plt.subplot(1, 2, 2)
plt.imshow(test_images[index].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.show()
