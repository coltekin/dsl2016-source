from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Embedding
from keras.layers import AveragePooling1D, Convolution1D, MaxPooling1D
from keras.utils import np_utils
import sys, time
from collections import defaultdict

maxlen = 200
maxfeatures = 1000
maxchars = 200
embedding_dims = 20
batch_size = 32
nb_epoch = 50
nb_filter = 250
filter_length = 3

data_path = "../data"

def read_data(data_file):
    labels = []
    documents = []
    with open(data_file, "r") as fp:
        linenum = 0
        for line in fp:
            if len(line) < 3: # skip empty lines and ^Z at the end
                continue
            doc, label = line.strip().split("\t")
            labels.append(label)
            documents.append(doc)
            linenum += 1
    return (documents, labels)

def char_tokenizer(s):
    return str.encode(s)

def getVocab(D):
    charSet = defaultdict(int)
    for s in D:
        d = char_tokenizer(s)
        for c in d:
            charSet[c] += 1
    genexp = ((k, charSet[k]) for k in sorted(charSet, key=charSet.get, reverse=True))
    bigrams, values = zip(*genexp)
    return bigrams[:maxfeatures]

def ngram_transform(D, features):
    """Converts a document to a vector of n-gram counts"""
    nD = []
    for i, d in enumerate(D):
        nd = []
        counter = defaultdict(float)
        for c in char_tokenizer(d):
            counter[c] += 1.0
        for f in features:
            nd.append(counter[f])
        nD.append(nd)
    return np.array(nD)

print("Reading the training set... ", end="")
sys.stdout.flush()
pt = time.time()
doc_train, y_train = read_data(data_path + "/task1-train.txt")
print(time.time() - pt)

print("Reading the test set... ", end="")
sys.stdout.flush()
pt = time.time()
doc_test, y_test = read_data(data_path + "/task1-dev.txt")
print(time.time() - pt)

print("Transforming the datasets... ", end="")
sys.stdout.flush()
pt = time.time()
features = getVocab(doc_train)
#features = list(char_vocab.keys())

x_train = ngram_transform(doc_train, features)
x_test = ngram_transform(doc_test, features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(time.time() - pt)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print("Transforming the labels... ", end="")
sys.stdout.flush()
pt = time.time()
unique_labels = ['es-MX', 'bs', 'es-AR', 'es-ES', 'fr-CA', 'id', 'pt-PT', 'hr', 'pt-BR', 'fr-FR', 'sr', 'my']
print("Labels ",unique_labels)
n_classes = len(unique_labels)
indim = x_train.shape[1]
y_train = [[unique_labels.index(y)] for y in y_train]
y_test = [[unique_labels.index(y)] for y in y_test]
y_train = np_utils.to_categorical(np.array(y_train), len(unique_labels))
y_test = np_utils.to_categorical(np.array(y_test), len(unique_labels))
print(time.time() - pt)

model=Sequential()
model.add(Dense(256, input_dim=indim))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(n_classes, activation = 'softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=32, nb_epoch=10,
          validation_data=(x_test, y_test))







