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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
import numpy as np
import logging
import copy
tf.get_logger().setLevel(logging.ERROR)

MAX_MODEL_SIZE = 500000
CANDIDATE_EVALUATIONS = 500
EVAL_EPOCHS = 3
FINAL_EPOCHS = 20

layer_types = ['DENSE', 'CONV2D', 'MAXPOOL2D']
param_values = dict([('size', [16, 64, 256, 1024, 4096]),
                ('activation', ['relu', 'tanh', 'elu']),
                ('kernel_size', [(1, 1), (2, 2), (3, 3), (4, 4)]),
                ('stride', [(1, 1), (2, 2), (3, 3), (4, 4)]),
                ('dropout', [0.0, 0.4, 0.7, 0.9])])

layer_params = dict([('DENSE', ['size', 'activation', 'dropout']),
                     ('CONV2D', ['size', 'activation',
                                 'kernel_size', 'stride',
                                 'dropout']),
                     ('MAXPOOL2D', ['kernel_size', 'stride',
                                    'dropout'])])

# Load dataset.
cifar_dataset = keras.datasets.cifar10
(train_images, train_labels), (test_images,
                    test_labels) = cifar_dataset.load_data()

# Standardize dataset.
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

# Change labels to one-hot.
train_labels = to_categorical(train_labels,
                              num_classes=10)
test_labels = to_categorical(test_labels,
                             num_classes=10)

# Methods to create a model definition.
def generate_random_layer(layer_type):
    layer = {}
    layer['layer_type'] = layer_type
    params = layer_params[layer_type]
    for param in params:
        values = param_values[param]
        layer[param] = values[np.random.randint(0, len(values))]
    return layer

def generate_model_definition():
    layer_count = np.random.randint(2, 9)
    non_dense_count = np.random.randint(1, layer_count)
    layers = []
    for i in range(layer_count):
        if i < non_dense_count:
            layer_type = layer_types[np.random.randint(1, 3)]
            layer = generate_random_layer(layer_type)
        else:
            layer = generate_random_layer('DENSE')
        layers.append(layer)
    return layers

def compute_weight_count(layers):
    last_shape = (32, 32, 3)
    total_weights = 0
    for layer in layers:
        layer_type = layer['layer_type']
        if layer_type == 'DENSE':
            size = layer['size']
            weights = size * (np.prod(last_shape) + 1)
            last_shape = (layer['size'])
        else:
            stride = layer['stride']
            if layer_type == 'CONV2D':
                size = layer['size']
                kernel_size = layer['kernel_size']
                weights = size * ((np.prod(kernel_size) *
                                   last_shape[2]) + 1)
                last_shape = (np.ceil(last_shape[0]/stride[0]),
                              np.ceil(last_shape[1]/stride[1]),
                              size)
            elif layer_type == 'MAXPOOL2D':
                weights = 0
                last_shape = (np.ceil(last_shape[0]/stride[0]),
                              np.ceil(last_shape[1]/stride[1]),
                              last_shape[2])
        total_weights += weights
    total_weights += ((np.prod(last_shape) + 1) * 10)
    return total_weights

# Methods to create and evaluate model based on model definition.
def add_layer(model, params, prior_type):
    layer_type = params['layer_type']
    if layer_type == 'DENSE':
        if prior_type != 'DENSE':
            model.add(Flatten())
        size = params['size']
        act = params['activation']
        model.add(Dense(size, activation=act))
    elif layer_type == 'CONV2D':
        size = params['size']
        act = params['activation']
        kernel_size = params['kernel_size']
        stride = params['stride']
        model.add(Conv2D(size, kernel_size, activation=act,
                         strides=stride, padding='same'))
    elif layer_type == 'MAXPOOL2D':
        kernel_size = params['kernel_size']
        stride = params['stride']
        model.add(MaxPooling2D(pool_size=kernel_size,
                               strides=stride, padding='same'))
    dropout = params['dropout']
    if(dropout > 0.0):
        model.add(Dropout(dropout))

def create_model(layers):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=(32, 32, 3)))
    prev_layer = 'LAMBDA' # Dummy layer to set input_shape
    for layer in layers:
        add_layer(model, layer, prev_layer)
        prev_layer = layer['layer_type']
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

def create_and_evaluate_model(model_definition):
    weight_count = compute_weight_count(model_definition)
    if weight_count > MAX_MODEL_SIZE:
        return 0.0
    model = create_model(model_definition)
    history = model.fit(train_images, train_labels,
                        validation_data=(test_images, test_labels),
                        epochs=EVAL_EPOCHS, batch_size=64,
                        verbose=2, shuffle=False)
    acc = history.history['val_accuracy'][-1]
    print('Size: ', weight_count)
    print('Accuracy: %5.2f' %acc)
    return acc

# Pure random search.
np.random.seed(7)
val_accuracy = 0.0
for i in range(CANDIDATE_EVALUATIONS):
    valid_model = False
    while(valid_model == False):
        model_definition = generate_model_definition()
        acc = create_and_evaluate_model(model_definition)
        if acc > 0.0:
            valid_model = True
    if acc > val_accuracy:
        best_model = model_definition
        val_accuracy = acc
    print('Random search, best accuracy: %5.2f' %val_accuracy)

# Helper method for hill climbing and evolutionary algorithm.
def tweak_model(model_definition):
    layer_num = np.random.randint(0, len(model_definition))
    last_layer = len(model_definition) - 1
    for first_dense, layer in enumerate(model_definition):
        if layer['layer_type'] == 'DENSE':
            break
    if np.random.randint(0, 2) == 1:
        delta = 1
    else:
        delta = -1
    if np.random.randint(0, 2) == 1:
        # Add/remove layer.
        if len(model_definition) < 3:
            delta = 1 # Layer removal not allowed
        if delta == -1:
            # Remove layer.
            if layer_num == 0 and first_dense == 1:
                layer_num += 1 # Require >= 1 non-dense layer
            if layer_num == first_dense and layer_num == last_layer:
                layer_num -= 1 # Require >= 1 dense layer
            del model_definition[layer_num]
        else:
            # Add layer.
            if layer_num < first_dense:
                layer_type = layer_types[np.random.randint(1, 3)]
            else:
                layer_type = 'DENSE'
            layer = generate_random_layer(layer_type)
            model_definition.insert(layer_num, layer)
    else:
        # Tweak parameter.
        layer = model_definition[layer_num]
        layer_type = layer['layer_type']
        params = layer_params[layer_type]
        param = params[np.random.randint(0, len(params))]
        current_val = layer[param]
        values = param_values[param]
        index = values.index(current_val)
        max_index = len(values)
        new_val = values[(index + delta) % max_index]
        layer[param] = new_val

# Hill climbing, starting from best model from random search.
model_definition = best_model

for i in range(CANDIDATE_EVALUATIONS):
    valid_model = False
    while(valid_model == False):
        old_model_definition = copy.deepcopy(model_definition)
        tweak_model(model_definition)
        acc = create_and_evaluate_model(model_definition)
        if acc > 0.0:
            valid_model = True
        else:
            model_definition = old_model_definition
    if acc > val_accuracy:
        best_model = copy.deepcopy(model_definition)
        val_accuracy = acc
    else:
        model_definition = old_model_definition
    print('Hill climbing, best accuracy: %5.2f' %val_accuracy)

# Evaluate final model for larger number of epochs.
model = create_model(best_model)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
history = model.fit(
    train_images, train_labels, validation_data =
    (test_images, test_labels), epochs=FINAL_EPOCHS, batch_size=64,
    verbose=2, shuffle=True)
