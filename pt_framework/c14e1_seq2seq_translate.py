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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text \
    import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence \
    import pad_sequences
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 128
MAX_WORDS = 10000
READ_LINES = 60000
LAYER_SIZE = 256
EMBEDDING_WIDTH = 128
TEST_PERCENT = 0.2
SAMPLE_SIZE = 20
OOV_WORD = 'UNK'
PAD_INDEX = 0
OOV_INDEX = 1
START_INDEX = MAX_WORDS - 2
STOP_INDEX = MAX_WORDS - 1
MAX_LENGTH = 60
SRC_DEST_FILE_NAME = '../data/fra.txt'

# Function to read file.
def read_file_combined(file_name, max_len):
    file = open(file_name, 'r', encoding='utf-8')
    src_word_sequences = []
    dest_word_sequences = []
    for i, line in enumerate(file):
        if i == READ_LINES:
            break
        pair = line.split('\t')
        word_sequence = text_to_word_sequence(pair[1])
        src_word_sequence = word_sequence[0:max_len]
        src_word_sequences.append(src_word_sequence)
        word_sequence = text_to_word_sequence(pair[0])
        dest_word_sequence = word_sequence[0:max_len]
        dest_word_sequences.append(dest_word_sequence)
    file.close()
    return src_word_sequences, dest_word_sequences

# Functions to tokenize and un-tokenize sequences.
def tokenize(sequences):
    # "MAX_WORDS-2" used to reserve two indices
    # for START and STOP.
    tokenizer = Tokenizer(num_words=MAX_WORDS-2,
                          oov_token=OOV_WORD)
    tokenizer.fit_on_texts(sequences)
    token_sequences = tokenizer.texts_to_sequences(sequences)
    return tokenizer, token_sequences

def tokens_to_words(tokenizer, seq):
    word_seq = []
    for index in seq:
        if index == PAD_INDEX:
            word_seq.append('PAD')
        elif index == OOV_INDEX:
            word_seq.append(OOV_WORD)
        elif index == START_INDEX:
            word_seq.append('START')
        elif index == STOP_INDEX:
            word_seq.append('STOP')
        else:
            word_seq.append(tokenizer.sequences_to_texts(
                [[index]])[0])
    print(word_seq)

# Read file and tokenize.
src_seq, dest_seq = read_file_combined(SRC_DEST_FILE_NAME,
                                       MAX_LENGTH)
src_tokenizer, src_token_seq = tokenize(src_seq)
dest_tokenizer, dest_token_seq = tokenize(dest_seq)

# Prepare training data.
dest_target_token_seq = [x + [STOP_INDEX] for x in dest_token_seq]
dest_input_token_seq = [[START_INDEX] + x for x in
                        dest_target_token_seq]
src_input_data = pad_sequences(src_token_seq)
dest_input_data = pad_sequences(dest_input_token_seq,
                                padding='post')
dest_target_data = pad_sequences(
    dest_target_token_seq, padding='post', maxlen
    = len(dest_input_data[0]))

# Convert to same precision as model.
src_input_data = src_input_data.astype(np.int64)
dest_input_data = dest_input_data.astype(np.int64)
dest_target_data = dest_target_data.astype(np.int64)

# Split into training and test set.
rows = len(src_input_data[:,0])
all_indices = list(range(rows))
test_rows = int(rows * TEST_PERCENT)
test_indices = random.sample(all_indices, test_rows)
train_indices = [x for x in all_indices if x not in test_indices]

train_src_input_data = src_input_data[train_indices]
train_dest_input_data = dest_input_data[train_indices]
train_dest_target_data = dest_target_data[train_indices]

test_src_input_data = src_input_data[test_indices]
test_dest_input_data = dest_input_data[test_indices]
test_dest_target_data = dest_target_data[test_indices]

# Create a sample of the test set that we will inspect in detail.
test_indices = list(range(test_rows))
sample_indices = random.sample(test_indices, SAMPLE_SIZE)
sample_input_data = test_src_input_data[sample_indices]
sample_target_data = test_dest_target_data[sample_indices]

# Create Dataset objects.
trainset = TensorDataset(torch.from_numpy(train_src_input_data),
                         torch.from_numpy(train_dest_input_data),
                         torch.from_numpy(train_dest_target_data))
testset = TensorDataset(torch.from_numpy(test_src_input_data),
                         torch.from_numpy(test_dest_input_data),
                         torch.from_numpy(test_dest_target_data))

# Define models.
class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(MAX_WORDS, EMBEDDING_WIDTH)
        nn.init.uniform_(self.embedding_layer.weight, -0.05, 0.05) # Default is -1, 1.
        self.lstm_layers = nn.LSTM(EMBEDDING_WIDTH, LAYER_SIZE, num_layers=2, batch_first=True)

    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.lstm_layers(x)
        return x[1]

class DecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = None
        self.use_state = False
        self.embedding_layer = nn.Embedding(MAX_WORDS, EMBEDDING_WIDTH)
        nn.init.uniform_(self.embedding_layer.weight, -0.05, 0.05) # Default is -1, 1.
        self.lstm_layers = nn.LSTM(EMBEDDING_WIDTH, LAYER_SIZE, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(LAYER_SIZE, MAX_WORDS)

    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        if(self.use_state):
            x = self.lstm_layers(x, self.state)
        else:
            x = self.lstm_layers(x)
        self.state = (x[1][0].detach().clone(), x[1][1].detach().clone()) # Store most recent internal state.
        x = self.output_layer(x[0])
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

encoder_model = EncoderModel()
decoder_model = DecoderModel()

# Loss functions and optimizer.
encoder_optimizer = torch.optim.RMSprop(encoder_model.parameters(), lr=0.001)
decoder_optimizer = torch.optim.RMSprop(decoder_model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

# Using a custom training loop instead of our standard training function.
# Transfer model to GPU.
encoder_model.to(device)
decoder_model.to(device)

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False)

# Train and test repeatedly.
for i in range(EPOCHS):
    encoder_model.train() # Set model in training mode.
    decoder_model.train() # Set model in training mode.
    train_loss = 0.0
    train_correct = 0
    train_batches = 0
    train_elems = 0
    for src_inputs, dest_inputs, dest_targets in trainloader:
        # Move data to GPU.
        src_inputs, dest_inputs, dest_targets = src_inputs.to(
            device), dest_inputs.to(device), dest_targets.to(device)

        # Zero the parameter gradients.
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Forward pass.
        encoder_state = encoder_model(src_inputs)
        decoder_model.set_state(encoder_state)
        outputs = decoder_model(dest_inputs)
        loss = loss_function(outputs.view(-1, MAX_WORDS), dest_targets.view(-1))
        # Accumulate metrics.
        _, indices = torch.max(outputs.data, 2)
        train_correct += (indices == dest_targets).sum().item()
        train_elems += indices.numel()
        train_batches +=  1
        train_loss += loss.item()

        # Backward pass and update.
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    train_loss = train_loss / train_batches
    train_acc = train_correct / train_elems

    # Evaluate the model on the test dataset.
    encoder_model.eval() # Set model in inference mode.
    decoder_model.eval() # Set model in inference mode.
    test_loss = 0.0
    test_correct = 0
    test_batches = 0
    test_elems = 0
    for src_inputs, dest_inputs, dest_targets in testloader:
        # Move data to GPU.
        src_inputs, dest_inputs, dest_targets = src_inputs.to(
            device), dest_inputs.to(device), dest_targets.to(device)
        encoder_state = encoder_model(src_inputs)
        decoder_model.set_state(encoder_state)
        outputs = decoder_model(dest_inputs)
        loss = loss_function(outputs.view(-1, MAX_WORDS), dest_targets.view(-1))
        _, indices = torch.max(outputs, 2)
        test_correct += (indices == dest_targets).sum().item()
        test_elems += indices.numel()
        test_batches +=  1
        test_loss += loss.item()

    test_loss = test_loss / test_batches
    test_acc = test_correct / test_elems
    print(f'Epoch {i+1}/{EPOCHS} loss: {train_loss:.4f} - acc: {train_acc:0.4f} - val_loss: {test_loss:.4f} - val_acc: {test_acc:0.4f}')

    # Loop through samples to see result
    for (test_input, test_target) in zip(sample_input_data,
                                         sample_target_data):
        # Run a single sentence through encoder model.
        x = np.reshape(test_input, (1, -1))
        inputs = torch.from_numpy(x)
        inputs = inputs.to(device)
        last_states = encoder_model(inputs)

        # Provide resulting state and START_INDEX as input
        # to decoder model.
        decoder_model.set_state(last_states)
        prev_word_index = START_INDEX
        produced_string = ''
        pred_seq = []
        for j in range(MAX_LENGTH):
            x = np.reshape(np.array(prev_word_index), (1, 1))
            # Predict next word and capture internal state.
            inputs = torch.from_numpy(x)
            inputs = inputs.to(device)
            outputs = decoder_model(inputs)
            preds = outputs.cpu().detach().numpy()[0][0]
            state = decoder_model.get_state()
            decoder_model.set_state(state)

            # Find the most probable word.
            prev_word_index = preds.argmax()
            pred_seq.append(prev_word_index)
            if prev_word_index == STOP_INDEX:
                break
        tokens_to_words(src_tokenizer, test_input)
        tokens_to_words(dest_tokenizer, test_target)
        tokens_to_words(dest_tokenizer, pred_seq)
        print('\n\n')
