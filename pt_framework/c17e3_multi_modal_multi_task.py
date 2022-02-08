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

# Function to create question and answer text.
def create_question_answer(tokenizer, labels):
    text = []
    answers = np.zeros(len(labels))
    for i, label in enumerate(labels):
        question_num = i % 4
        if question_num == 0:
            text.append('lower half')
            if label < 5:
                answers[i] = 1.0
        elif question_num == 1:
            text.append('upper half')
            if label >= 5:
                answers[i] = 1.0
        elif question_num == 2:
            text.append('even number')
            if label % 2 == 0:
                answers[i] = 1.0
        elif question_num == 3:
            text.append('odd number')
            if label % 2 == 1:
                answers[i] = 1.0
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text).astype(np.int64)
    answers = answers.reshape((len(labels), 1))
    return text, answers

# Create second modality for training and test set.
vocabulary = ['lower', 'upper', 'half', 'even', 'odd', 'number']
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(vocabulary)
train_text, train_answers = create_question_answer(tokenizer,
                                                   train_labels)
test_text, test_answers = create_question_answer(tokenizer,
                                                 test_labels)

# Create datasets.
trainset = TensorDataset(torch.from_numpy(train_images), 
                         torch.from_numpy(train_text),
                         torch.from_numpy(train_labels),
                         torch.from_numpy(train_answers))

testset = TensorDataset(torch.from_numpy(test_images),
                         torch.from_numpy(test_text),
                         torch.from_numpy(test_labels),
                         torch.from_numpy(test_answers))

# Define model.
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(MAX_WORDS, EMBEDDING_WIDTH)
        nn.init.uniform_(self.embedding_layer.weight, -0.05, 0.05) # Default is -1, 1.
        self.lstm_layers = nn.LSTM(EMBEDDING_WIDTH, 8, num_layers=1, batch_first=True)

        self.linear_layer = nn.Linear(784+8, 25)
        self.relu_layer = nn.ReLU()
        self.class_output_layer = nn.Linear(25, 10)
        self.answer_output_layer = nn.Linear(25, 1)

    def forward(self, inputs):
        image_input = inputs[0]
        text_input = inputs[1]

        # Process textual data.
        x0 = self.embedding_layer(text_input)
        x0 = self.lstm_layers(x0)

        # Process image data.
        # Flatten the image.
        x1 = image_input.view(-1, 784)

        # Concatenate input branches and build shared trunk.
        x = torch.cat((x0[1][0][0], x1), dim=1)
        x = self.linear_layer(x)
        x = self.relu_layer(x)

        # Define two heads.
        class_output = self.class_output_layer(x)
        answer_output = self.answer_output_layer(x)
        return [class_output, answer_output]

model = MultiTaskModel()

# Loss function and optimizer
optimizer = torch.optim.Adam(model.parameters())
loss_function0 = nn.CrossEntropyLoss()
loss_function1 = nn.BCEWithLogitsLoss()

# Training loop for multi-modal multi-task model.
# Transfer model to GPU.
model.to(device)

# Create dataloaders.
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

for i in range(EPOCHS):
    model.train() # Set model in training mode.
    class_train_loss = 0.0
    class_train_correct = 0
    answer_train_loss = 0.0
    answer_train_correct = 0
    train_batches = 0
    for image_inputs, text_inputs, class_targets, answer_targets in trainloader:
        # Move data to GPU.
        image_inputs = image_inputs.to(device)
        text_inputs = text_inputs.to(device)
        class_targets = class_targets.to(device)
        answer_targets = answer_targets.to(device)

        # Zero the parameter gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model([image_inputs, text_inputs])
        class_loss = loss_function0(outputs[0], class_targets)
        answer_loss = loss_function1(outputs[1], answer_targets)
        loss = 0.5*class_loss + 0.5*answer_loss

        # Accumulate metrics.
        _, indices = torch.max(outputs[0].data, 1)
        class_train_correct += (indices == class_targets).sum().item()
        answer_train_correct += ((outputs[1].data > 0.0) == answer_targets).sum().item()
        train_batches +=  1
        class_train_loss += class_loss.item()
        answer_train_loss += answer_loss.item()

        # Backward pass and update.
        loss.backward()
        optimizer.step()

    class_train_loss = class_train_loss / train_batches
    class_train_acc = class_train_correct / (train_batches * BATCH_SIZE)
    answer_train_loss = answer_train_loss / train_batches
    answer_train_acc = answer_train_correct / (train_batches * BATCH_SIZE)

    # Evaluate the model on the test dataset.
    model.eval() # Set model in inference mode.
    class_test_loss = 0.0
    class_test_correct = 0
    answer_test_loss = 0.0
    answer_test_correct = 0
    test_batches = 0
    for image_inputs, text_inputs, class_targets, answer_targets in testloader:
        image_inputs = image_inputs.to(device)
        text_inputs = text_inputs.to(device)
        class_targets = class_targets.to(device)
        answer_targets = answer_targets.to(device)
        outputs = model([image_inputs, text_inputs])
        class_loss = loss_function0(outputs[0], class_targets)
        answer_loss = loss_function1(outputs[1], answer_targets)
        loss = 0.5*class_loss + 0.5*answer_loss
        _, indices = torch.max(outputs[0].data, 1)
        class_test_correct += (indices == class_targets).sum().item()
        answer_test_correct += ((outputs[1].data > 0.0) == answer_targets).sum().item()
        test_batches +=  1
        class_test_loss += class_loss.item()
        answer_test_loss += answer_loss.item()
    class_test_loss = class_test_loss / test_batches
    class_test_acc = class_test_correct / (test_batches * BATCH_SIZE)
    answer_test_loss = answer_test_loss / test_batches
    answer_test_acc = answer_test_correct / (test_batches * BATCH_SIZE)
    print(f'Epoch {i+1}/{EPOCHS} class loss: {class_train_loss:.4f} - answer loss: {answer_train_loss:.4f} - class acc: {class_train_acc:0.4f} - answer acc: {answer_train_acc:0.4f} - class val_loss: {class_test_loss:.4f} - answer val_loss: {answer_test_loss:.4f} - class val_acc: {class_test_acc:0.4f} - answer val_acc: {answer_test_acc:0.4f}')
