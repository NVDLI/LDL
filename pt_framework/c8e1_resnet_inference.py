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
import torchvision
from torchvision import transforms
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and preprocess image.
image = Image.open('../data/dog.jpg')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) # Parameters are documented at pytorch.org.
input_tensor = preprocess(image)

# Convert to 4-dimensional tensor.
inputs = input_tensor.unsqueeze(0)

# Load the pre-trained model.
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Transfer model to GPU.
model.to(device)

# Do prediction.
inputs = inputs.to(device)
with torch.no_grad():
    outputs = model(inputs)

# Convert to probabilities, since final SoftMax activation is not in pretrained model.
probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# Print class ID for top 5 predictions.
_, indices = torch.sort(probabilities, descending=True)
for i in range(0, 5):
    print('ImageNet class:', indices[i].item(), ', probability = %4.3f' % probabilities[indices[i]].item())

# Show image.
image.show()
