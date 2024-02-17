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
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
# Using Keras Tokenizer for simplicity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text \
    import text_to_word_sequence
import numpy as np
from utilities import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 32
BATCH_SIZE = 256
INPUT_FILE_NAME = '../data/frankenstein.txt'
WINDOW_LENGTH = 40
WINDOW_STEP = 3
PREDICT_LENGTH = 3
MAX_WORDS = 7500
EMBEDDING_WIDTH = 100

# Open the input file.
file = open(INPUT_FILE_NAME, 'r', encoding='utf-8-sig')
text = file.read()
file.close()

# Make lower case and split into individual words.
text = text_to_word_sequence(text)

# Create training examples.
fragments = []
targets = []
for i in range(0, len(text) - WINDOW_LENGTH, WINDOW_STEP):
    fragments.append(text[i: i + WINDOW_LENGTH])
    targets.append(text[i + WINDOW_LENGTH])

# Convert to indices.
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='UNK')
tokenizer.fit_on_texts(text)
fragments_indexed = tokenizer.texts_to_sequences(fragments)
targets_indexed = tokenizer.texts_to_sequences(targets)

# Convert to appropriate input and output formats.
X = np.array(fragments_indexed, dtype=np.int64)
y = np.zeros(len(targets_indexed), dtype=np.int64)
for i, target_index in enumerate(targets_indexed):
    y[i] = target_index[0]

# Split into training and test set.
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.05, random_state=0)

# Create Dataset objects.
trainset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
testset = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y))

# Define model.
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = None
        self.use_state = False
        self.embedding_layer = nn.Embedding(MAX_WORDS, EMBEDDING_WIDTH)
        nn.init.uniform_(self.embedding_layer.weight, -0.05, 0.05) # Default is -1, 1.
        self.lstm_layers = nn.LSTM(EMBEDDING_WIDTH, 128, num_layers=2, dropout=0.2, batch_first=True)
        self.dropout_layer = nn.Dropout(0.2)
        self.linear_layer = nn.Linear(128, 128)
        self.relu_layer = nn.ReLU()
        self.output_layer = nn.Linear(128, MAX_WORDS)

    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        if(self.use_state):
            x = self.lstm_layers(x, self.state)
        else:
            x = self.lstm_layers(x)
        self.state = (x[1][0].detach().clone(), x[1][1].detach().clone()) # Store most recent internal state.
        x = self.dropout_layer(x[1][0][1])
        x = self.linear_layer(x)
        x = self.relu_layer(x)
        x = self.output_layer(x)
        return x

    # Functions to provide explicit control of LSTM state.
    def set_state(self, state):
        self.state = state
        self.use_state = True
        return

    def get_state(self):
        return self.state

    def clear_state(self):
        self.use_state = False
        return

model = LanguageModel()

# Loss function and optimizer
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

# Train the model.
train_model(model, device, EPOCHS, BATCH_SIZE, trainset, testset,
            optimizer, loss_function, 'acc')

# Provide beginning of sentence and
# predict next words in a greedy manner.
first_words = ['i', 'saw']
first_words_indexed = tokenizer.texts_to_sequences(
    first_words)
model.clear_state()
predicted_string = ''
# Feed initial words to the model.
for i, word_index in enumerate(first_words_indexed):
    x = np.zeros((1, 1), dtype=np.int64)
    x[0][0] = word_index[0]
    predicted_string += first_words[i]
    predicted_string += ' '
    inputs = torch.from_numpy(x)
    inputs = inputs.to(device)
    outputs = model(inputs)
    y_predict = outputs.cpu().detach().numpy()[0]
    state = model.get_state()
    model.set_state(state)
# Predict PREDICT_LENGTH words.
for i in range(PREDICT_LENGTH):
    new_word_index = np.argmax(y_predict)
    word = tokenizer.sequences_to_texts(
        [[new_word_index]])
    x[0][0] = new_word_index
    predicted_string += word[0]
    predicted_string += ' '
    inputs = torch.from_numpy(x)
    inputs = inputs.to(device)
    outputs = model(inputs)
    y_predict = outputs.cpu().detach().numpy()[0]
    state = model.get_state()
    model.set_state(state)
print(predicted_string)

# Explore embedding similarities.
it = model.modules()
next(it)
embeddings = next(it).weight
embeddings = embeddings.detach().clone().cpu().numpy()

lookup_words = ['the', 'saw', 'see', 'of', 'and',
                'monster', 'frankenstein', 'read', 'eat']
for lookup_word in lookup_words:
    lookup_word_indexed = tokenizer.texts_to_sequences(
        [lookup_word])
    print('words close to:', lookup_word)
    lookup_embedding = embeddings[lookup_word_indexed[0]]
    word_indices = {}
    # Calculate distances.
    for i, embedding in enumerate(embeddings):
        distance = np.linalg.norm(
            embedding - lookup_embedding)
        word_indices[distance] = i
    # Print sorted by distance.
    for distance in sorted(word_indices.keys())[:5]:
        word_index = word_indices[distance]
        word = tokenizer.sequences_to_texts([[word_index]])[0]
        print(word + ': ', distance)
    print('')
