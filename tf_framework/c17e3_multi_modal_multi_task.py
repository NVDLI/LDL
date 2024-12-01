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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text \
    import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence \
    import pad_sequences
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import numpy as np
import logging
tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 20
MAX_WORDS = 8
EMBEDDING_WIDTH = 4

# Load training and test datasets.
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images,
                               test_labels) = mnist.load_data()

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
    text = pad_sequences(text)
    return text, answers

# Create second modality for training and test set.
vocabulary = ['lower', 'upper', 'half', 'even', 'odd', 'number']
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(vocabulary)
train_text, train_answers = create_question_answer(tokenizer,
                                                   train_labels)
test_text, test_answers = create_question_answer(tokenizer,
                                                 test_labels)

# Create model with functional API.
image_input = Input(shape=(28, 28))
text_input = Input(shape=(2, ))

# Declare layers.
embedding_layer = Embedding(output_dim=EMBEDDING_WIDTH,
                            input_dim = MAX_WORDS)
lstm_layer = LSTM(8)
flatten_layer = Flatten()
concat_layer = Concatenate()
dense_layer = Dense(25,activation='relu')
class_output_layer = Dense(10, activation='softmax')
answer_output_layer = Dense(1, activation='sigmoid')

# Connect layers.
embedding_output = embedding_layer(text_input)
lstm_output = lstm_layer(embedding_output)
flatten_output = flatten_layer(image_input)
concat_output = concat_layer([lstm_output, flatten_output])
dense_output = dense_layer(concat_output)
class_outputs = class_output_layer(dense_output)
answer_outputs = answer_output_layer(dense_output)

# Build and train model.
model = Model([image_input, text_input], [class_outputs,
                                          answer_outputs])
model.compile(loss=['sparse_categorical_crossentropy',
                    'binary_crossentropy'], optimizer='adam',
                    metrics=['accuracy', 'accuracy'],
                    loss_weights = [0.5, 0.5])
model.summary()
history = model.fit([train_images, train_text],
                    [train_labels, train_answers],
                    validation_data=([test_images, test_text],
                    [test_labels, test_answers]), epochs=EPOCHS,
                    batch_size=64, verbose=2, shuffle=True)
