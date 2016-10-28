from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Embedding
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.utils import np_utils
import sys, time
from collections import defaultdict
from keras.layers import LSTM, GRU

data_path = "../data"

maxlen = 512
embedding_dims = 30
batch_size = 128
nb_epoch = 20
nb_filter = 32
filter_length = 3
pool_length = 5
minfreq = 50
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
    return list(s)

def getVocab(D):
    charSet = defaultdict(int)
    max_features = 1
    for d in D:
        for c in char_tokenizer(d):
            charSet[c] += 1
    for c in charSet:
        if charSet[c] > minfreq:
            max_features += 1
    return charSet, max_features

def transform(D, vocab):
#    features = vocab.keys()
    features = []
    for k in vocab.keys():
        if vocab[k] > minfreq:
            features.append(k)

    start_char = 1
    #print(len(features))
    X = []
    for j, d in enumerate(D):
        x = [start_char]
        for c in char_tokenizer(d):
            if c in features:
                x.append(features.index(c)+start_char)
        X.append(x)
        #X.append(x[:maxlen])#clip sentence length
    return X
    
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
char_vocab, max_features = getVocab(doc_train)
print("Number of features= ", max_features)
x_train = transform(doc_train, char_vocab)
x_test = transform(doc_test, char_vocab)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(time.time() - pt)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print("Transforming the labels... ", end="")
sys.stdout.flush()
pt = time.time()
unique_labels = list(set(y_test))
n_classes = len(unique_labels)
indim = x_train.shape[1]
y_train = [[unique_labels.index(y)] for y in y_train]
y_test = [[unique_labels.index(y)] for y in y_test]
y_train = np_utils.to_categorical(np.array(y_train), len(unique_labels))
y_test = np_utils.to_categorical(np.array(y_test), len(unique_labels))
print(time.time() - pt)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(GRU(output_dim=20, activation='sigmoid', inner_activation='hard_sigmoid'))
#model.add(AveragePooling1D(pool_length=model.output_shape[1]))
#model.add(Flatten())
model.add(Dense(n_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(x_test, y_test))







