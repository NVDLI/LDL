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
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
# Using Keras Tokenizer for simplicity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text \
    import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence \
    import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 64
MAX_WORDS = 8
EMBEDDING_WIDTH = 4

# Load MNIST dataset.
trainset = MNIST(root='./pt_data', train=True, download=True)
testset = MNIST(root='./pt_data', train=False, download=True)

# Convert to numpy arrays to enable us to create a richer dataset.
train_images = trainset.data.numpy().astype(np.float32)
train_labels = trainset.targets.numpy()
test_images = testset.data.numpy().astype(np.float32)
test_labels = testset.targets.numpy()

# Standardize the data.
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

# Function to create second modality.
def create_text(tokenizer, labels):
    text = []
    for i, label in enumerate(labels):
        if i % 2 == 0:
            if label < 5:
                text.append('lower half')
            else:
                text.append('upper half')
        else:
            if label % 2 == 0:
                text.append('even number')
            else:
                text.append('odd number')
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text)
    return text

# Create second modality for training and test set.
vocabulary = ['lower', 'upper', 'half', 'even', 'odd', 'number']
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(vocabulary)
train_text = create_text(tokenizer, train_labels).astype(np.int64)
test_text = create_text(tokenizer, test_labels).astype(np.int64)

# Create datasets.
trainset = TensorDataset(torch.from_numpy(train_images),
                         torch.from_numpy(train_text),
                         torch.from_numpy(train_labels))

testset = TensorDataset(torch.from_numpy(test_images),
                         torch.from_numpy(test_text),
                         torch.from_numpy(test_labels))

# Define model.
class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(MAX_WORDS, EMBEDDING_WIDTH)
        nn.init.uniform_(self.embedding_layer.weight, -0.05, 0.05) # Default is -1, 1.
        self.lstm_layers = nn.LSTM(EMBEDDING_WIDTH, 8, num_layers=1, batch_first=True)

        self.linear_layer = nn.Linear(784+8, 25)
        self.relu_layer = nn.ReLU()
        self.output_layer = nn.Linear(25, 10)

    def forward(self, inputs):
        image_input = inputs[0]
        text_input = inputs[1]

        # Process textual data.
        x0 = self.embedding_layer(text_input)
        x0 = self.lstm_layers(x0)

        # Process image data.
        # Flatten the image.
        x1 = image_input.view(-1, 784)

        # Concatenate input branches and feed to output layer.
        x = torch.cat((x0[1][0][0], x1), dim=1)
        x = self.linear_layer(x)
        x = self.relu_layer(x)
        x = self.output_layer(x)
        return x

model = MultiModalModel()

# Loss function and optimizer
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

# Training loop for multi modal model.

# Transfer model to GPU.
model.to(device)

# Create dataloaders.
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

for i in range(EPOCHS):
    model.train() # Set model in training mode.
    train_loss = 0.0
    train_correct = 0
    train_batches = 0
    for image_inputs, text_inputs, targets in trainloader:
        # Move data to GPU.
        image_inputs = image_inputs.to(device)
        text_inputs = text_inputs.to(device)
        targets = targets.to(device)

        # Zero the parameter gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model([image_inputs, text_inputs])
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

    # Evaluate the model on the test dataset.
    model.eval() # Set model in inference mode.
    test_loss = 0.0
    test_correct = 0
    test_batches = 0
    for image_inputs, text_inputs, targets in testloader:
        image_inputs = image_inputs.to(device)
        text_inputs = text_inputs.to(device)
        targets = targets.to(device)
        outputs = model([image_inputs, text_inputs])
        loss = loss_function(outputs, targets)
        _, indices = torch.max(outputs.data, 1)
        test_correct += (indices == targets).sum().item()
        test_batches +=  1
        test_loss += loss.item()
    test_loss = test_loss / test_batches
    test_acc = test_correct / (test_batches * BATCH_SIZE)
    print(f'Epoch {i+1}/{EPOCHS} loss: {train_loss:.4f} - acc: {train_acc:0.4f} - val_loss: {test_loss:.4f} - val_acc: {test_acc:0.4f}')

# Print input modalities and output for one test example.
print(test_labels[0])
print(tokenizer.sequences_to_texts([test_text[0]]))
plt.figure(figsize=(1, 1))
plt.imshow(test_images[0], cmap=plt.get_cmap('gray'))
plt.show()

# Create two examples to experiment with impact of textual input.
images = test_images[0:2]
images[1] = test_images[0] # Same image (digit 7) in both examples
text = np.array([tokenizer.texts_to_sequences(['upper half'])[0],
                 tokenizer.texts_to_sequences(['lower half'])[0]]) # Incorrect text for 2nd example

# Convert to tensors and move to GPU if present.
experiment_images = torch.from_numpy(images)
experiment_text = torch.from_numpy(text)
experiment_images = experiment_images.to(device)
experiment_text = experiment_text.to(device)

# Do predictions and apply softmax on output.
output = model([experiment_images, experiment_text])
output = F.softmax(output, dim = 1)
y = output.detach().cpu().numpy()[0]
print('Predictions with correct input:')
for i in range(len(y)):
    index = y.argmax()
    print('Digit: %d,' %index, 'probability: %5.2e' %y[index])
    y[index] = 0

y = output.detach().cpu().numpy()[1]
print('\nPredictions with incorrect input:')
for i in range(len(y)):
    index = y.argmax()
    print('Digit: %d,' %index, 'probability: %5.2e' %y[index])
    y[index] = 0
