"""
Performs classification of dialect data. Total of 
"""
import numpy as np
np.random.seed(1337)  # for reproducibility

import sys
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras import backend as K
from keras.utils import np_utils

from collections import defaultdict
import itertools as it
import codecs
from dsl_preprocessing import load_data

nb_epoch = 5
batch_size = 32
nb_filter = 64
maxlen = 200
hidden_dims = 64
embedding_dims = 10
filter_length = 3

def load_data(fname, maxlen, vocab=None):
    char_dict = defaultdict(int)
    segments, labels, unique_chars, unique_labels = [], [], [], []
        
    seg_lengths = []
    
    for line in codecs.open(fname,"r","utf-8"):
        line = line.strip()

        if "\t" not in line:
            continue
        segment, label = line.split("\t")
        labels.append(label)
        seg_lengths.append(len(segment))
        for x in segment:
            char_dict[x] += 1
        segments.append(segment[:maxlen])
        if label not in unique_labels:
            unique_labels.append(label)
    
    print
    if vocab is None:
        segments = [[s if char_dict[s] > 5 else "UNK" for s in x] for x in segments]
    else:
        segments = [[s if s in vocab else "UNK" for s in x] for x in segments]
    
    for s in segments:
        for x in s:
            if x not in unique_chars:
                unique_chars.append(x)
    
    segments = [[unique_chars.index(s) for s in x] for x in segments]
    labels = [unique_labels.index(l) for l in labels]
    
    print "Maximum, min, average segment length ", max(seg_lengths), min(seg_lengths), sum(seg_lengths)*1.0/len(seg_lengths)
    
    return np.array(segments), np.array(labels), unique_chars

X_train, Y_train, train_chars = load_data("DSL-training/task1-train.txt", maxlen)
X_test, Y_test, test_chars = load_data("DSL-training/task1-dev.txt", maxlen, vocab=train_chars)

print "Training characters \n", train_chars, len(train_chars)
print "Development characters \n", test_chars, len(test_chars)


nb_classes = np.max(Y_train)+1

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

#sys.exit(1)
print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(len(train_chars),
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        activation='relu',
                        subsample_length=1))
# we use max pooling:
model.add(MaxPooling1D(pool_length=3))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test))

