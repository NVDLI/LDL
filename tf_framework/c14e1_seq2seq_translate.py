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

import numpy as np
import random
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text \
    import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence \
    import pad_sequences
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

# Constants
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
src_input_data = pad_sequences(src_token_seq, padding='post')
dest_input_data = pad_sequences(dest_input_token_seq,
                                padding='post')
dest_target_data = pad_sequences(
    dest_target_token_seq, padding='post', maxlen
    = len(dest_input_data[0]))

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

# Build encoder model.
# Input is input sequence in source language.
enc_embedding_input = Input(shape=(None, ))

# Create the encoder layers.
enc_embedding_layer = Embedding(
    output_dim=EMBEDDING_WIDTH, input_dim
    = MAX_WORDS, mask_zero=True)
enc_layer1 = LSTM(LAYER_SIZE, return_state=True,
                  return_sequences=True)
enc_layer2 = LSTM(LAYER_SIZE, return_state=True)

# Connect the encoder layers.
# We don't use the last layer output, only the state.
enc_embedding_layer_outputs = \
    enc_embedding_layer(enc_embedding_input)
enc_layer1_outputs, enc_layer1_state_h, enc_layer1_state_c = \
    enc_layer1(enc_embedding_layer_outputs)
_, enc_layer2_state_h, enc_layer2_state_c = \
    enc_layer2(enc_layer1_outputs)

# Build the model.
enc_model = Model(enc_embedding_input,
                  [enc_layer1_state_h, enc_layer1_state_c,
                   enc_layer2_state_h, enc_layer2_state_c])
enc_model.summary()

# Build decoder model.
# Input to the network is input sequence in destination
# language and intermediate state.
dec_layer1_state_input_h = Input(shape=(LAYER_SIZE,))
dec_layer1_state_input_c = Input(shape=(LAYER_SIZE,))
dec_layer2_state_input_h = Input(shape=(LAYER_SIZE,))
dec_layer2_state_input_c = Input(shape=(LAYER_SIZE,))
dec_embedding_input = Input(shape=(None, ))

# Create the decoder layers.
dec_embedding_layer = Embedding(output_dim=EMBEDDING_WIDTH,
                                input_dim=MAX_WORDS,
                                mask_zero=True)
dec_layer1 = LSTM(LAYER_SIZE, return_state = True,
                  return_sequences=True)
dec_layer2 = LSTM(LAYER_SIZE, return_state = True,
                  return_sequences=True)
dec_layer3 = Dense(MAX_WORDS, activation='softmax')

# Connect the decoder layers.
dec_embedding_layer_outputs = dec_embedding_layer(
    dec_embedding_input)
dec_layer1_outputs, dec_layer1_state_h, dec_layer1_state_c = \
    dec_layer1(dec_embedding_layer_outputs,
    initial_state=[dec_layer1_state_input_h,
                   dec_layer1_state_input_c])
dec_layer2_outputs, dec_layer2_state_h, dec_layer2_state_c = \
    dec_layer2(dec_layer1_outputs,
    initial_state=[dec_layer2_state_input_h,
                   dec_layer2_state_input_c])
dec_layer3_outputs = dec_layer3(dec_layer2_outputs)

# Build the model.
dec_model = Model([dec_embedding_input,
                   dec_layer1_state_input_h,
                   dec_layer1_state_input_c,
                   dec_layer2_state_input_h,
                   dec_layer2_state_input_c],
                  [dec_layer3_outputs, dec_layer1_state_h,
                   dec_layer1_state_c, dec_layer2_state_h,
                   dec_layer2_state_c])
dec_model.summary()

# Build and compile full training model.
# We do not use the state output when training.
train_enc_embedding_input = Input(shape=(None, ))
train_dec_embedding_input = Input(shape=(None, ))
intermediate_state = enc_model(train_enc_embedding_input)
train_dec_output, _, _, _, _ = dec_model(
    [train_dec_embedding_input] +
    intermediate_state)
training_model = Model([train_enc_embedding_input,
                        train_dec_embedding_input],
                        train_dec_output)
optimizer = RMSprop(learning_rate=0.01)
training_model.compile(loss='sparse_categorical_crossentropy',
                       optimizer=optimizer, metrics =['accuracy'])
training_model.summary()

# Train and test repeatedly.
for i in range(EPOCHS):
    print('step: ' , i)
    # Train model for one epoch.
    history = training_model.fit(
        [train_src_input_data, train_dest_input_data],
        train_dest_target_data, validation_data=(
            [test_src_input_data, test_dest_input_data],
            test_dest_target_data), batch_size=BATCH_SIZE,
        epochs=1)

    # Loop through samples to see result
    for (test_input, test_target) in zip(sample_input_data,
                                         sample_target_data):
        # Run a single sentence through encoder model.
        x = np.reshape(test_input, (1, -1))
        last_states = enc_model.predict(
            x, verbose=0)
        # Provide resulting state and START_INDEX as input
        # to decoder model.
        prev_word_index = START_INDEX
        produced_string = ''
        pred_seq = []
        for j in range(MAX_LENGTH):
            x = np.reshape(np.array(prev_word_index), (1, 1))
            # Predict next word and capture internal state.
            preds, dec_layer1_state_h, dec_layer1_state_c, \
                dec_layer2_state_h, dec_layer2_state_c = \
                    dec_model.predict(
                        [x] + last_states, verbose=0)
            last_states = [dec_layer1_state_h,
                           dec_layer1_state_c,
                           dec_layer2_state_h,
                           dec_layer2_state_c]
            # Find the most probable word.
            prev_word_index = np.asarray(preds[0][0]).argmax()
            pred_seq.append(prev_word_index)
            if prev_word_index == STOP_INDEX:
                break
        tokens_to_words(src_tokenizer, test_input)
        tokens_to_words(dest_tokenizer, test_target)
        tokens_to_words(dest_tokenizer, pred_seq)
        print('\n\n')
