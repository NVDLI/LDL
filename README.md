# LDL
This code repository contains code examples associated with the book
"Learning Deep Learning: Theory and Practice of Neural Networks,
Computer Vision, Natural Language Processing, and Transformers Using
TensorFlow" (ISBN: 9780137470358), and the video series "Learning Deep
Learning: From Perceptron to Large Language Models" (ISBN: 9780138177553)
by Magnus Ekman. In this readme file, "Learning Deep Learning" and "LDL"
are used interchangeably.

Related web sites:
* Publisher (Pearson) site with supplemental material and links to buy
  the LDL book: http://informit.com/title/9780137470358
* Publisher (Pearson) site with supplemental material and links to buy
  the LDL videos: https://www.informit.com/store/learning-deep-learning-from-perceptron-to-large-language-9780138177553
* Author's site: http://ldlbook.com
* NVIDIA Deep Learning Institute (DLI) book site:
  http://www.nvidia.com/dli-books

The LDL book can also be purchased at Amazon.com:
https://www.amazon.com/Learning-Deep-Practice-Transformers-TensorFlow/dp/0137470355

The LDL video series can also be accessed at O'REILLY:
https://learning.oreilly.com/course/learning-deep-learning/9780138177652/

Some of the code examples rely on datasets that are not in the repository
itself. This file provides all information needed to obtain these datasets
and to run the programming examples. Further, the code examples are well
documented in the Jupyter notebook versions. However, the purpose of the
examples are to illustrate concepts taught in the LDL book and videos.
The examples should be considered in that context, and are best consumed in
conjunction with reading the book.

## Docker Files
The repository contains two Docker files to simplify running the code examples under Docker:
* `Dockerfile_tf` - for the TensorFlow versions of the code examples
* `Dockerfile_pt` - for the PyTorch versions of the code examples

See the author's site for blog posts describing how to set up and run Docker on Linux and Windows: http://ldlbook.com

## Code Examples
The code examples can be divided into three categories in the following
three directories:
* `stand_alone` - stand-alone examples not relying on a Deep Learning (DL)
 framework
* `tf_framework` - examples based on the TensorFlow DL framework
* `pt_framework` - examples based on the PyTorch DL framework

There is a one-to-one mapping between the code examples in the tf_framework
and pt_framework directory. Pick a framework of your choice or learn both!

The initial versions of these programming examples were tested with
versions 2.4 and 2.5 of TensorFlow and versions 1.8.0 and 1.9.0 of
PyTorch. The most recent versions have been tested with version 2.16.1 of 
TensorFlow (with Keras 3) and version 2.5.0 of PyTorch. TensorFlow is 
sometimes rather verbose when using GPU acceleration. To make it less
verbose, set the environment variable `TF_CPP_MIN_LOG_LEVEL` to the value 2.
If you are using bash, this can be done with `export TF_CPP_MIN_LOG_LEVEL=2`.

The naming of each code example from the book follows the pattern
`cXeY_DESCRIPTION.py` where X represents the chapter number, Y the example
number in that chapter, and DESCRIPTION is a brief description of what the
example is doing. The examples named `aFeY_DESCRIPTION.py` are not from a
regular chapter but from *Appendix F*. The code examples from the book are
available both as Python files (.py) and as Jupyter notebooks (.ipynb).

The naming of each code example from the videos follows the pattern
`vX_Y_DESCRIPTION.ipynb` where X_Y represent the video number and DESCRIPTION
is a brief description of what the example is doing. The code examples from
the videos are available only as Jupyter notebooks (.ipynb).

Apart from the three directories containing code examples, there is a single
directory named `data` that is supposed to contain datasets needed by some of
the code examples. The repository contains some of these assets but the
user needs to download additional datasets to fully populate it. Instructions
to that is found in the section [Datasets](#datasets) below.

Each code example is expected to be run from within the directory where the
code example is located, as it uses a relative path to access the dataset. That
is, you first need to change to the stand_alone directory before running code
examples located in that directory.

Because of the stochastic nature of DL algorithms, the results may vary from
run to run. That is, it is expected that your results will not exactly reproduce
the results stated in the book.

### Alternative Versions
Some of the code examples have alternative versions to work around issues observed
on some platforms. This applies to the following code examples:
* `tf_framework/c11_e1_autocomplete_no_rdo` - this version does not use recurrent
 dropout, which causes hangs on some platforms
* `tf_framework/c12_e1_autocomplete_embedding_no_rdo` - this version does not use recurrent
 dropout, which causes hangs on some platforms
* `tf_framework/c17_e4_nas_random_hill_multiprocess` - this version spawns multiple
 processes, which works around a memory leak problem observed on some platforms
* `tf_framework/c17_e5_nas_evolution_multiprocess` - this version spawns multiple
 processes, which works around a memory leak problem observed on some platforms

## Supporting Spreadsheet
Apart from the code examples, this repository also contains a spreadsheet named
[network_example.xlsx](network_example.xlsx). The spreadsheet provides additional insight into the basic
workings of neurons and the learning process. It is unlikely that this
spreadsheet is useful without first reading the corresponding description in LDL.

The spreadsheet consists of three tabs, each corresponding to a specific section
of the initial chapters:
* `perceptron_learning` corresponds to the section *The Perceptron Learning
 Algorithm* in Chapter 1, *The Rosenblatt Perceptron*.
* `backprop_learning` corresponds to the section *Using Backpropagation to
 Compute the Gradient* in Chapter 3, *Sigmoid Neurons and Backpropagation*.
* `xor_example` corresponds to the section *Programming Example: Learning
 the XOR Function* in Chapter 3.

## Datasets
Some of the programming examples rely on datasets accessible through the DL
framework but others need to be downloaded and placed in the appropriate location.
This section describes how to obtain the ones that need to be downloaded. All
program examples assume that the downloaded datasets are placed in the directory
named data in the root of the code example directory tree.

### MNIST
The MNIST Database of handwritten digits can be obtained from
http://yann.lecun.com/exdb/mnist.

Download the following files:
* `train-images-idx3-ubyte.gz`
* `train-labels-idx1-ubyte.gz`
* `t10k-images-idx3-ubyt.gz`
* `t10k-labels-idx1-ubyte.gz`

Once downloaded, gunzip them to the `data/mnist/` directory. You need the
Python package idx2numpy to use this version of the MNIST dataset.

Alternative links if the URL above does not work:
* https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
* https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
* https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
* https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

### BOOKSTORE SALES DATA FROM US CENSUS BUREAU
Sales data from the United States Census Bureau can be obtained from
https://www.census.gov/econ/currentdata.

Select Monthly Retail Trade and Food Services and click the Submit button. That
should take you to a page where you need to specify five different steps. Select:
* Monthly Retail Trade and Food Services
* Start: 1992 End: 2020
* 451211: Book Stores
* Sales - Monthly
* U.S. Total

Make sure that the checkbox Not Seasonally Adjusted is checked. Then click the
GET DATA button. That should result in a table with data values. Download it to
a comma-separated values (CSV) file by clicking the link **TXT**. Remove the first few
lines in the downloaded CSV file so the file starts with a single line containing
headings saying "Period,Value" followed by one line for each month. Further, remove
any lines with non-numerical values, such as "NA", at the end of the file. Name the
file book_store_sales.csv and copy to the `data` directory.

### FRANKENSTEIN FROM PROJECT GUTENBERG
The text for Mary Shelley's Frankenstein can be downloaded from
https://www.gutenberg.org/files/84/84-0.txt.
Rename the file to `frankenstein.txt` and copy to the `data` directory.

### GloVe WORD EMBEDDINGS
The GloVe word embeddings file, which is close to 1 GB in size, can be downloaded
from http://nlp.stanford.edu/data/glove.6B.zip. Unzip it after downloading and copy
the file glove.6B.100d.txt to the `data` directory.

### ANKI BILINGUAL SENTENCE PAIRS
The Anki bilingual sentence pairs can be downloaded from
http://www.manythings.org/anki/fra-eng.zip. Unzip it after download and copy the
file `fra.txt` to the `data` directory.

### COCO
Create a directory named `coco` inside of the data directory.
Download the following file:
http://images.cocodataset.org/annotations/annotations_trainval2014.zip. Unzip it and
copy the file `captions_train2014.json` to the directory `coco`.
Download the following 13 GB file: http://images.cocodataset.org/zips/train2014.zip.
Unzip it into the `data/coco/` directory so the path to the unzipped directory is
`data/coco/train2014/`.
