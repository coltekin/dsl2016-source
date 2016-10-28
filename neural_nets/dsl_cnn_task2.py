from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Embedding, Convolution1D, MaxPooling1D
#from keras.layers import AveragePooling1D
from keras.utils import np_utils
import sys, time, re
from collections import defaultdict

_white_spaces = re.compile(r"\s\s+")
maxlen = 512
maxchars = 200
embedding_dims = 16
batch_size = 32
nb_epoch = 32
nb_filter = 32
filter_length = 3
pool_length = 3
minfreq = 10
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
            doc = _white_spaces.sub(" ", doc)
            labels.append(label)
            documents.append(doc)
            linenum += 1
    return (documents, labels)

def char_tokenizer(s):
    return list(s)

def getVocab(D):
    charSet = defaultdict(int)
    max_features = 3
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
    oov_char = 2
    index_from = 3
    #print(len(features))
    X = []
    for j, d in enumerate(D):
        x = [start_char]
        for c in char_tokenizer(d):
            freq = vocab[c]
            if c in features and freq > minfreq:
                x.append(features.index(c)+index_from)
            elif c in vocab and freq <= minfreq:
                x.append(oov_char)
            else:
                continue
        X.append(x)
        #X.append(x[:maxlen])#clip sentence length
    return X
    
print("Reading the training set... ", end="")
sys.stdout.flush()
pt = time.time()
doc_train, y_train = read_data(data_path + "/task2-train-shuf.txt")
print(time.time() - pt)

print("Transforming the datasets... ", end="")
sys.stdout.flush()
pt = time.time()
char_vocab, max_features = getVocab(doc_train)
print("Number of features= ", max_features)
x_train = transform(doc_train, char_vocab)

print(len(x_train), 'train sequences')

print(time.time() - pt)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

print('x_train shape:', x_train.shape)


print("Transforming the labels... ", end="")
sys.stdout.flush()
pt = time.time()
unique_labels = list(set(y_train))
n_classes = len(unique_labels)
indim = x_train.shape[1]
y_train = [[unique_labels.index(y)] for y in y_train]
y_train = np_utils.to_categorical(np.array(y_train), len(unique_labels))
print(time.time() - pt)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen, dropout=0.2))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(Flatten())
model.add(Dense(180, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_split=0.3)







