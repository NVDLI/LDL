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
from torchvision import transforms
from PIL import Image
import json
import numpy as np
import pickle
import gzip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAINING_FILE_DIR = '../data/coco/'
OUTPUT_FILE_DIR = 'pt_data/feature_vectors/'

with open(TRAINING_FILE_DIR \
          + 'captions_train2014.json') as json_file:
    data = json.load(json_file)
image_dict = {}
for image in data['images']:
    image_dict[image['id']] = [image['file_name']]
for anno in data['annotations']:
    image_dict[anno['image_id']].append(anno['caption'])

# Create network without top layers.
model = torchvision.models.vgg19(weights='DEFAULT')
model_blocks = list(model.children())
layers = list(model_blocks[0].children())
model = nn.Sequential(*layers[0:-1])
model.eval()

# Transfer model to GPU
model.to(device)

# Run all images through the network and save the output.
for i, key in enumerate(image_dict.keys()):
    if i % 1000 == 0:
        print('Progress: ' + str(i) + ' images processed')
    item = image_dict.get(key)
    filename = TRAINING_FILE_DIR + 'train2014/' + item[0]

    # Load and preprocess image.
    # Resize so shortest side is 256 pixels.
    # Crop to center 224x224 region.
    image = Image.open(filename).convert('RGB')
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

    # Call model and save resulting tensor to disk.
    inputs = inputs.to(device)
    with torch.no_grad():
        y = model(inputs)[0].cpu().numpy()
    save_filename = OUTPUT_FILE_DIR + \
        item[0] + '.pickle.gzip'
    pickle_file = gzip.open(save_filename, 'wb')
    pickle.dump(y, pickle_file)
    pickle_file.close()

# Save the dictionary containing captions and filenames.
save_filename = OUTPUT_FILE_DIR + 'caption_file.pickle.gz'
pickle_file = gzip.open(save_filename, 'wb')
pickle.dump(image_dict, pickle_file)
pickle_file.close()
