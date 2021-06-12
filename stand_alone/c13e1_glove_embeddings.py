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
import scipy.spatial

# Read embeddings from file.
def read_embeddings():
    FILE_NAME = '../data/glove.6B.100d.txt'
    embeddings = {}
    file = open(FILE_NAME, 'r', encoding='utf-8')
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:],
                            dtype='float32')
        embeddings[word] = vector
    file.close()
    print('Read %s embeddings.' % len(embeddings))
    return embeddings

def print_n_closest(embeddings, vec0, n):
    word_distances = {}
    for (word, vec1) in embeddings.items():
        distance = scipy.spatial.distance.cosine(
            vec1, vec0)
        word_distances[distance] = word
    # Print words sorted by distance.
    for distance in sorted(word_distances.keys())[:n]:
        word = word_distances[distance]
        print(word + ': %6.3f' % distance)

embeddings = read_embeddings()

lookup_word = 'hello'
print('\nWords closest to ' + lookup_word)
print_n_closest(embeddings,
                embeddings[lookup_word], 3)

lookup_word = 'precisely'
print('\nWords closest to ' + lookup_word)
print_n_closest(embeddings,
                embeddings[lookup_word], 3)

lookup_word = 'dog'
print('\nWords closest to ' + lookup_word)
print_n_closest(embeddings,
                embeddings[lookup_word], 3)

lookup_word = 'king'
print('\nWords closest to ' + lookup_word)
print_n_closest(embeddings,
                embeddings[lookup_word], 3)

lookup_word = '(king - man + woman)'
print('\nWords closest to ' + lookup_word)
vec = embeddings['king'] - embeddings[
    'man'] + embeddings['woman']
print_n_closest(embeddings, vec, 3)

lookup_word = 'sweden'
print('\nWords closest to ' + lookup_word)
print_n_closest(embeddings,
                embeddings[lookup_word], 3)

lookup_word = 'madrid'
print('\nWords closest to ' + lookup_word)
print_n_closest(embeddings,
                embeddings[lookup_word], 3)

lookup_word = '(madrid - spain + sweden)'
print('\nWords closest to ' + lookup_word)
vec = embeddings['madrid'] - embeddings[
    'spain'] + embeddings['sweden']
print_n_closest(embeddings, vec, 3)
