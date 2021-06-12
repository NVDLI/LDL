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
import matplotlib.pyplot as plt
import idx2numpy

np.random.seed(7) # To make repeatable
LEARNING_RATE = 0.1
EPOCHS = 20
TRAIN_IMAGE_FILENAME = '../data/mnist/train-images-idx3-ubyte'
TRAIN_LABEL_FILENAME = '../data/mnist/train-labels-idx1-ubyte'
TEST_IMAGE_FILENAME = '../data/mnist/t10k-images-idx3-ubyte'
TEST_LABEL_FILENAME = '../data/mnist/t10k-labels-idx1-ubyte'
BATCH_SIZE = 32

# Function to read dataset.
def read_mnist():
    train_images = idx2numpy.convert_from_file(
        TRAIN_IMAGE_FILENAME)
    train_labels = idx2numpy.convert_from_file(
        TRAIN_LABEL_FILENAME)
    test_images = idx2numpy.convert_from_file(
        TEST_IMAGE_FILENAME)
    test_labels = idx2numpy.convert_from_file(
        TEST_LABEL_FILENAME)

    # Reformat and standardize.
    x_train = train_images.reshape(60000, 784)
    mean = np.mean(x_train)
    stddev = np.std(x_train)
    x_train = (x_train - mean) / stddev
    x_test = test_images.reshape(10000, 784)
    x_test = (x_test - mean) / stddev

    # One-hot encoded output.
    y_train = np.zeros((60000, 10))
    y_test = np.zeros((10000, 10))
    for i, y in enumerate(train_labels):
        y_train[i][y] = 1
    for i, y in enumerate(test_labels):
        y_test[i][y] = 1
    return x_train, y_train, x_test, y_test

# Read train and test examples.
x_train, y_train, x_test, y_test = read_mnist()
index_list = list(range(int(len(x_train)/BATCH_SIZE)))

def layer_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count+1))
    for i in range(neuron_count):
        for j in range(1, (input_count+1)):
            weights[i][j] = np.random.uniform(-0.1, 0.1)
    return weights

# Declare matrices and vectors representing the neurons.
hidden_layer_w = layer_w(25, 784)
hidden_layer_y = np.zeros((25, BATCH_SIZE))
hidden_layer_error = np.zeros((25, BATCH_SIZE))

output_layer_w = layer_w(10, 25)
output_layer_y = np.zeros((10, BATCH_SIZE))
output_layer_error = np.zeros((10, BATCH_SIZE))

chart_x = []
chart_y_train = []
chart_y_test = []
def show_learning(epoch_no, train_acc, test_acc):
    global chart_x
    global chart_y_train
    global chart_y_test
    print('epoch no:', epoch_no, ', train_acc: ',
          '%6.4f' % train_acc,
          ', test_acc: ', '%6.4f' % test_acc)
    chart_x.append(epoch_no + 1)
    chart_y_train.append(1.0 - train_acc)
    chart_y_test.append(1.0 - test_acc)

def plot_learning():
    plt.plot(chart_x, chart_y_train, 'r-',
             label='training error')
    plt.plot(chart_x, chart_y_test, 'b-',
             label='test error')
    plt.axis([0, len(chart_x), 0.0, 1.0])
    plt.xlabel('training epochs')
    plt.ylabel('error')
    plt.legend()
    plt.show()

def forward_pass(x):
    global hidden_layer_y
    global output_layer_y
    # Activation function for hidden layer.
    hidden_layer_z = np.matmul(hidden_layer_w, x)
    hidden_layer_y = np.tanh(hidden_layer_z)
    hidden_output_array = np.concatenate(
        (np.ones((1, BATCH_SIZE)), hidden_layer_y))
    # Activation function for output layer.
    output_layer_z = np.matmul(output_layer_w,
        hidden_output_array)
    output_layer_y = 1.0 / (1.0 + np.exp(-output_layer_z))

def backward_pass(y_truth):
    global hidden_layer_error
    global output_layer_error
    # Backpropagate error for each output neuron.
    error_prime = -(y_truth - output_layer_y)
    output_log_prime = output_layer_y * (
        1.0 - output_layer_y)
    output_layer_error = error_prime * output_log_prime
    # Backpropagate error for each hidden neuron.
    hidden_tanh_prime = 1.0 - hidden_layer_y**2
    hidden_weighted_error = np.matmul(np.matrix.transpose(
        output_layer_w[:, 1:]), output_layer_error)
    hidden_layer_error = (
        hidden_tanh_prime * hidden_weighted_error)

def adjust_weights(x):
    global output_layer_w
    global hidden_layer_w
    delta_matrix = np.zeros((len(hidden_layer_error[:, 0]),
                             len(x[:, 0])))
    for i in range(BATCH_SIZE):
        delta_matrix += np.outer(hidden_layer_error[:, i],
                                 x[:, i]) * LEARNING_RATE
    delta_matrix /= BATCH_SIZE
    hidden_layer_w -= delta_matrix
    hidden_output_array = np.concatenate(
        (np.ones((1, BATCH_SIZE)), hidden_layer_y))
    delta_matrix = np.zeros(
        (len(output_layer_error[:, 0]),
         len(hidden_output_array[:, 0])))
    for i in range(BATCH_SIZE):
        delta_matrix += np.outer(
            output_layer_error[:, i],
            hidden_output_array[:, i]) * LEARNING_RATE
    delta_matrix /= BATCH_SIZE
    output_layer_w -= delta_matrix

# Network training loop.
for i in range(EPOCHS): # Train EPOCHS iterations
    np.random.shuffle(index_list) # Randomize order
    correct_training_results = 0
    for j in index_list: # Train on all examples
        j *= BATCH_SIZE
        x = np.ones((785, BATCH_SIZE))
        y = np.zeros((10, BATCH_SIZE))
        for k in range(BATCH_SIZE):
            x[1:, k] = x_train[j + k]
            y[:, k] = y_train[j + k]
        forward_pass(x)
        for k in range(BATCH_SIZE):
            if(output_layer_y[:, k].argmax()
                    == y[:, k].argmax()):
                correct_training_results += 1
        backward_pass(y)
        adjust_weights(x)

    correct_test_results = 0
    for j in range(0, (len(x_test) - BATCH_SIZE),
                   BATCH_SIZE): # Evaluate network
        x = np.ones((785, BATCH_SIZE))
        y = np.zeros((10, BATCH_SIZE))
        for k in range(BATCH_SIZE):
            x[1:, k] = x_test[j + k]
            y[:, k] = y_test[j + k]
        forward_pass(x)
        for k in range(BATCH_SIZE):
            if(output_layer_y[:, k].argmax()
                    == y[:, k].argmax()):
                correct_test_results += 1
    # Show progress
    show_learning(i, correct_training_results/len(x_train),
                  correct_test_results/len(x_test))
plot_learning() # Create plot
