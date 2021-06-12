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
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import copy
import random
from collections import OrderedDict
from utilities import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# Load training dataset into a single batch to compute mean and stddev.
transform = transforms.Compose([transforms.ToTensor()])
trainset = CIFAR10(root='./pt_data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
data = next(iter(trainloader))
mean = data[0].mean()
stddev = data[0].std()

# Load and standardize training and test dataset.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, stddev)])

trainset = CIFAR10(root='./pt_data', train=True, download=True, transform=transform)
testset = CIFAR10(root='./pt_data', train=False, download=True, transform=transform)

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
def add_layer(model_list, params, prior_type, last_shape):
    layer_num = len(model_list)
    act = None
    layer_type = params['layer_type']
    if layer_type == 'DENSE':
        if prior_type != 'DENSE':
            model_list.append(("layer" + str(layer_num), nn.Flatten()))
            layer_num += 1
            last_shape = int(np.prod(last_shape))
        size = params['size']
        act = params['activation']
        model_list.append(("layer" + str(layer_num), nn.Linear(last_shape, size)))
        last_shape = (size)
    elif layer_type == 'CONV2D':
        size = params['size']
        act = params['activation']
        kernel_size = params['kernel_size']
        stride = params['stride']
        padding = int(kernel_size[0] / 2)
        model_list.append(("layer" + str(layer_num),
                           nn.Conv2d(last_shape[2], size, kernel_size, stride=stride, padding=padding)))
        last_shape = (int((last_shape[0]+2*padding-(kernel_size[0]-1)-1)/stride[0]+1),
                      int((last_shape[1]+2*padding-(kernel_size[1]-1)-1)/stride[1]+1),
                      size)
    elif layer_type == 'MAXPOOL2D':
        kernel_size = params['kernel_size']
        stride = params['stride']
        padding = int(kernel_size[0] / 2)
        model_list.append(("layer" + str(layer_num),
                           nn.MaxPool2d(kernel_size, stride=stride, padding=padding)))
        last_shape = (int((last_shape[0]+2*padding-(kernel_size[0]-1)-1)/stride[0]+1),
                      int((last_shape[1]+2*padding-(kernel_size[1]-1)-1)/stride[1]+1),
                      last_shape[2])
    layer_num += 1
    if(act != None):
        if (act == 'relu'):
            model_list.append(("layer" + str(layer_num), nn.ReLU()))
        elif (act == 'elu'):
            model_list.append(("layer" + str(layer_num), nn.ELU()))
        elif (act == 'tanh'):
            model_list.append(("layer" + str(layer_num), nn.Tanh()))
        layer_num += 1
    dropout = params['dropout']
    if(dropout > 0.0):
        model_list.append(("layer" + str(layer_num), nn.Dropout(p=dropout)))
    return last_shape

def create_model(layers):
    model_list = []
    prev_layer = 'NONE'
    last_shape = (32, 32, 3)
    for layer in layers:
        last_shape = add_layer(model_list, layer, prev_layer, last_shape)
        prev_layer = layer['layer_type']
    model_list.append(("layer" + str(len(model_list)), nn.Linear(last_shape, 10)))
    model = nn.Sequential(OrderedDict(model_list))
    return model

def create_and_evaluate_model(model_definition):
    weight_count = compute_weight_count(model_definition)
    if weight_count > MAX_MODEL_SIZE:
        return 0.0
    model = create_model(model_definition)

    # Loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()
    train_result = train_model(model, device, EVAL_EPOCHS, 64, trainset, testset,
                optimizer, loss_function, 'acc')
    acc = train_result[1]
    print('Size: ', weight_count)
    print('Accuracy: %5.2f' %acc)
    del model
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

# Helper method for hill climbing algorithm.
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
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()
train_result = train_model(model, device, FINAL_EPOCHS, 64, trainset, testset,
                  optimizer, loss_function, 'acc')
