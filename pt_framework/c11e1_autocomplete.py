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
from sklearn.model_selection import train_test_split
import numpy as np
from utilities import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 32
BATCH_SIZE = 256
INPUT_FILE_NAME = '../data/frankenstein.txt'
WINDOW_LENGTH = 40
WINDOW_STEP = 3
BEAM_SIZE = 8
NUM_LETTERS = 11
MAX_LENGTH = 50

# Open the input file.
file = open(INPUT_FILE_NAME, 'r', encoding='utf-8-sig')
text = file.read()
file.close()

# Make lower-case and remove newline and extra spaces.
text = text.lower()
text = text.replace('\n', ' ')
text = text.replace('  ', ' ')

# Encode characters as indices.
unique_chars = list(set(text))
char_to_index = dict((ch, index) for index,
                     ch in enumerate(unique_chars))
index_to_char = dict((index, ch) for index,
                     ch in enumerate(unique_chars))
encoding_width = len(char_to_index)

# Create training examples.
fragments = []
targets = []
for i in range(0, len(text) - WINDOW_LENGTH, WINDOW_STEP):
    fragments.append(text[i: i + WINDOW_LENGTH])
    targets.append(text[i + WINDOW_LENGTH])

# Convert to one-hot encoded training data.
X = np.zeros((len(fragments), WINDOW_LENGTH, encoding_width), dtype=np.float32)
y = np.zeros(len(fragments), dtype=np.int64)
for i, fragment in enumerate(fragments):
    for j, char in enumerate(fragment):
        X[i, j, char_to_index[char]] = 1
    target_char = targets[i]
    y[i] = char_to_index[target_char]

# Split into training and test set.
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.05, random_state=0)

# Create Dataset objects.
trainset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
testset = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y))

# Define model.
class LastTimestep(nn.Module):
    def forward(self, inputs):
        return inputs[1][0][1] # Return hidden state for last timestep.

model = nn.Sequential(
    nn.LSTM(encoding_width, 128, num_layers=2, dropout=0.2, batch_first=True),
    LastTimestep(),
    nn.Dropout(0.2), # Add this since PyTorch LSTM does not apply dropout to top layer.
    nn.Linear(128, encoding_width)
)

# Loss function and optimizer
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

# Train the model.
train_model(model, device, EPOCHS, BATCH_SIZE, trainset, testset,
            optimizer, loss_function, 'acc')

# Create initial single beam represented by triplet
# (probability , string , one-hot encoded string).
letters = 'the body '
one_hots = []
for i, char in enumerate(letters):
    x = np.zeros(encoding_width)
    x[char_to_index[char]] = 1
    one_hots.append(x)
beams = [(np.log(1.0), letters, one_hots)]

# Predict NUM_LETTERS into the future.
for i in range(NUM_LETTERS):
    minibatch_list = []
    # Create minibatch from one-hot encodings, and predict.
    for triple in beams:
        minibatch_list.append(triple[2])
    minibatch = np.array(minibatch_list, dtype=np.float32)
    inputs = torch.from_numpy(minibatch)
    inputs = inputs.to(device)
    outputs = model(inputs)
    outputs = F.softmax(outputs, dim = 1)
    y_predict = outputs.cpu().detach().numpy()

    new_beams = []
    for j, softmax_vec in enumerate(y_predict):
        triple = beams[j]
        # Create BEAM_SIZE new beams from each existing beam.
        for k in range(BEAM_SIZE):
            char_index = np.argmax(softmax_vec)
            new_prob = triple[0] + np.log(softmax_vec[char_index])
            new_letters = triple[1] + index_to_char[char_index]
            x = np.zeros(encoding_width)
            x[char_index] = 1
            new_one_hots = triple[2].copy()
            new_one_hots.append(x)
            new_beams.append((new_prob, new_letters, new_one_hots))
            softmax_vec[char_index] = 0
    # Prune tree to only keep BEAM_SIZE most probable beams.
    new_beams.sort(key=lambda tup: tup[0], reverse=True)
    beams = new_beams[0:BEAM_SIZE]
for item in beams:
    print(item[1])
