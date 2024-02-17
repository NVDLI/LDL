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
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import \
    text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import \
    pad_sequences
import pickle
import gzip
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 128
MAX_WORDS = 10000
READ_IMAGES = 90000
LAYER_SIZE = 256
EMBEDDING_WIDTH = 128
OOV_WORD = 'UNK'
PAD_INDEX = 0
OOV_INDEX = 1
START_INDEX = MAX_WORDS - 2
STOP_INDEX = MAX_WORDS - 1
MAX_LENGTH = 60
TRAINING_FILE_DIR = 'pt_data/feature_vectors/'
TEST_FILE_DIR = '../data/test_images/'
TEST_IMAGES = ['boat.jpg',
               'cat.jpg',
               'table.jpg',
               'bird.jpg']

# Function to read file.
def read_training_file(file_name, max_len):
    pickle_file = gzip.open(file_name, 'rb')
    image_dict = pickle.load(pickle_file)
    pickle_file.close()
    image_paths = []
    dest_word_sequences = []
    for i, key in enumerate(image_dict):
        if i == READ_IMAGES:
            break
        image_item = image_dict[key]
        image_paths.append(image_item[0])
        caption = image_item[1]
        word_sequence = text_to_word_sequence(caption)
        dest_word_sequence = word_sequence[0:max_len]
        dest_word_sequences.append(dest_word_sequence)
    return image_paths, dest_word_sequences

# Functions to tokenize and un-tokenize sequences.
def tokenize(sequences):
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

# Read files.
image_paths, dest_seq = read_training_file(TRAINING_FILE_DIR \
    + 'caption_file.pickle.gz', MAX_LENGTH)
dest_tokenizer, dest_token_seq = tokenize(dest_seq)

# Dataset class to read input files on the fly.
class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, dest_input_data,
                 dest_target_data):
        self.image_paths = image_paths
        self.dest_input_data = dest_input_data
        self.dest_target_data = dest_target_data

    def __len__(self):
        return len(self.dest_input_data)

    def __getitem__(self, idx):
        image_id = self.image_paths[idx]
        dest_input = self.dest_input_data[idx]
        dest_target = self.dest_target_data[idx]
        image_features = []
        file_name = TRAINING_FILE_DIR  \
            + image_id + '.pickle.gzip'
        pickle_file = gzip.open(file_name, 'rb')
        feature_vector = pickle.load(pickle_file)
        pickle_file.close()
        return torch.from_numpy(np.array(feature_vector)), \
               torch.from_numpy(np.array(dest_input)), \
               torch.from_numpy(np.array(dest_target))

# Prepare training data.
dest_target_token_seq = [x + [STOP_INDEX] for x in dest_token_seq]
dest_input_token_seq = [[START_INDEX] + x for x in
                        dest_target_token_seq]
dest_input_data = pad_sequences(dest_input_token_seq,
                                padding='post')
dest_target_data = pad_sequences(
    dest_target_token_seq, padding='post',
    maxlen=len(dest_input_data[0]))

# Convert to same precision as model.
dest_input_data = dest_input_data.astype(np.int64)
dest_target_data = dest_target_data.astype(np.int64)

trainset = ImageCaptionDataset(
    image_paths, dest_input_data, dest_target_data)

# Load the pre-trained VGG19 model.
vgg19_model = torchvision.models.vgg19(weights='DEFAULT')
model_blocks = list(vgg19_model.children())
layers = list(model_blocks[0].children())
vgg19_model = nn.Sequential(*layers[0:-1])
vgg19_model.eval()

# Define Captioning model.
class CaptioningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = None
        self.use_state = False
        self.avg_pool2d = nn.AvgPool2d(14)
        self.enc_h = nn.Linear(512, LAYER_SIZE)
        self.enc_c = nn.Linear(512, LAYER_SIZE)
        self.embedding_layer = nn.Embedding(MAX_WORDS, EMBEDDING_WIDTH)
        nn.init.uniform_(self.embedding_layer.weight, -0.05, 0.05) # Default is -1, 1.
        self.lstm_layer = nn.LSTM(EMBEDDING_WIDTH, LAYER_SIZE, batch_first=True)
        self.dec_query_layer = nn.Linear(LAYER_SIZE, 512) # used for attention
        self.output_layer = nn.Linear(512+LAYER_SIZE, MAX_WORDS)

    def forward(self, feature_vector, caption):
        x = self.embedding_layer(caption)
        if(self.use_state):
            x = self.lstm_layer(x, self.state)
        else:
            mean_output = self.avg_pool2d(feature_vector)
            mean_output = mean_output.view(-1, 512)
            h = F.relu(self.enc_h(mean_output))
            h = h.view(1, -1, LAYER_SIZE)
            c = F.relu(self.enc_c(mean_output))
            c = c.view(1, -1, LAYER_SIZE)
            x = self.lstm_layer(x, (h, c))
        self.state = (x[1][0].detach().clone(), x[1][1].detach().clone()) # Store most recent internal state.

        # Attention mechanism begins.
        # Compute query for each decoder time step.
        # Dimensions = (batch, seq, 512).
        query = self.dec_query_layer(x[0])

        # Reshape key/data from encoder into 196 spatial locations.
        # Dimensions = (batch, features=512, locations=196).
        key_data = feature_vector.view(-1, 512, 196)

        # Compute normalized attention scores.
        # Dimensions = (batch, seq, locations=196).
        scores = torch.matmul(query, key_data)
        probs = F.softmax(scores, dim = 2)

        # Move column in key_data:
        # Dimensions = (batch, locations=196, features=512).
        key_data = torch.movedim(key_data, 2, 1)

        # Last step in attention mechanism.
        # Compute weighted sum (attention output).
        # Dimensions = (batch, seq, 512).
        weighted_sum = torch.matmul(probs, key_data)

        # Concatenate with x[0] and feed to output layer.
        x = torch.cat((weighted_sum, x[0]), dim=2)
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

captioning_model = CaptioningModel()

# Loss function and optimizer.
captioning_optimizer = torch.optim.RMSprop(captioning_model.parameters())
loss_function = nn.CrossEntropyLoss()

# Transfer model to GPU.
vgg19_model.to(device)
captioning_model.to(device)

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)

# Train and test repeatedly.
for i in range(EPOCHS):
    captioning_model.train() # Set model in training mode.
    captioning_model.clear_state()
    train_loss = 0.0
    train_correct = 0
    train_batches = 0
    train_elems = 0
    for feature_inputs, dest_inputs, dest_targets in trainloader:
        # Move data to GPU.
        feature_inputs, dest_inputs, dest_targets = feature_inputs.to(
            device), dest_inputs.to(device), dest_targets.to(device)

        # Zero the parameter gradients.
        captioning_optimizer.zero_grad()

        # Forward pass.
        outputs = captioning_model(feature_inputs, dest_inputs)
        loss = loss_function(outputs.view(-1, MAX_WORDS), dest_targets.view(-1))
        # Accumulate metrics.
        _, indices = torch.max(outputs.data, 2)
        train_correct += (indices == dest_targets).sum().item()
        train_elems += indices.numel()
        train_batches +=  1
        train_loss += loss.item()

        # Backward pass and update.
        loss.backward()
        captioning_optimizer.step()
    train_loss = train_loss / train_batches
    train_acc = train_correct / train_elems

    print(f'Epoch {i+1}/{EPOCHS} loss: {train_loss:.4f} - acc: {train_acc:0.4f}')

    for filename in TEST_IMAGES:
        # Load and preprocess image.
        # Resize so shortest side is 256 pixels.
        # Crop to center 224x224 region.
        image = Image.open(TEST_FILE_DIR + filename).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image)

        # Rearrange array to have one more
        # dimension representing batch size = 1.
        inputs = input_tensor.unsqueeze(0)

        # Call model.
        inputs = inputs.to(device)
        with torch.no_grad():
            feature_vector = vgg19_model(inputs)
        captioning_model.clear_state()

        # Predict sentence word for word.
        prev_word_index = START_INDEX
        produced_string = ''
        pred_seq = []
        for j in range(MAX_LENGTH):
            x = np.reshape(np.array(prev_word_index), (1, 1))
            # Predict next word and capture internal state.
            inputs = torch.from_numpy(x)
            inputs = inputs.to(device)
            outputs = captioning_model(feature_vector, inputs)
            preds = outputs.cpu().detach().numpy()[0][0]
            state = captioning_model.get_state()
            captioning_model.set_state(state)

            # Find the most probable word.
            prev_word_index = preds.argmax()
            pred_seq.append(prev_word_index)
            if prev_word_index == STOP_INDEX:
                break
        tokens_to_words(dest_tokenizer, pred_seq)
        print('\n\n')
