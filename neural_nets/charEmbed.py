import numpy as np
np.random.seed(1337)

import json, re, time, sys
from keras.layers import Embedding, Convolution1D, MaxPooling1D, LSTM, Merge, Input, merge
from keras.layers import Dense, Flatten, Dropout, Activation, AveragePooling1D
from keras.models import Model, Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils

maxlen = 100
data_path = "../data"
embeddings_dim = 50

from gensim.utils import simple_preprocess
tokenize = lambda x: simple_preprocess(x)

def load_vocab(vocab_path):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """

    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word
    
def word2vec_embedding_layer(embeddings_path):
    """
    Generate an embedding layer word2vec embeddings
    :param embeddings_path: where the embeddings are saved (as a numpy file)
    :return: the generated embedding layer
    """

    weights = np.load(open(embeddings_path, 'rb'))
    #layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])
    return weights

def read_data(data_file):
    labels = []
    documents = []
    with open(data_file, "r") as fp:
        linenum = 0
        for line in fp:
            if len(line) < 3: # skip empty lines and ^Z at the end
                continue
            doc, label = line.strip().split("\t")
            doc = tokenize(doc)#Performs word tokenization
            #doc = list(doc)
            #doc = _white_spaces.sub(" ", doc)
            labels.append(label)
            documents.append(doc)
            linenum += 1
    return (documents, labels)
    
_white_spaces = re.compile(r"\s\s+")
print("Reading the training set... ", end="")
sys.stdout.flush()
pt = time.process_time()
doc_train, y_train = read_data(data_path + "/task1-train.txt")
print(time.process_time() - pt)

print("Reading the test set... ", end="")
sys.stdout.flush()
pt = time.process_time()
doc_test, y_test = read_data(data_path + "/task1-dev.txt")
print(time.process_time() - pt)

y_classes = list(set(y_train))
n_classes = len(y_classes)
print("Class labels = ",y_classes)
word2idx, idx2word = load_vocab("../wordembeddings/map.json")

X_train, Y_train, X_test, Y_test = [], [], [], []
for d in doc_train:
    x = []
    for c in d:
        if c not in word2idx:
            print("character in training ",c, " not found")
        else:
            x.append(int(word2idx[c])+1)
    X_train.append(x)

for d in doc_test:
    x = []
    for c in d:
        if c not in word2idx:
            print("character in testing ",c, " not found")
        else:
            x.append(int(word2idx[c])+1)
    X_test.append(x)

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print("Transforming the labels... ", end="")
sys.stdout.flush()
pt = time.time()
unique_labels = list(set(y_test))
n_classes = len(unique_labels)

Y_train = [[unique_labels.index(y)] for y in y_train]
Y_test = [[unique_labels.index(y)] for y in y_test]
Y_train = np_utils.to_categorical(np.array(Y_train), len(unique_labels))
Y_test = np_utils.to_categorical(np.array(Y_test), len(unique_labels))
print(time.time() - pt)

wordembeddings = word2vec_embedding_layer("../wordembeddings/embeddings.npz")
wordembeddings = np.vstack([np.zeros(embeddings_dim), embeddings])
max_features = len(word2idx.keys())+1

print('Build model...')
#model0 = Sequential()
#model0.add(Embedding(max_features, embeddings_dim, input_length=maxlen, weights=[embeddings], trainable=False))
#model0.add(Flatten())

#model.add(Flatten())
#model.add(Dropout(0.2))

model=Sequential()
#model.add(merged)
model.add(Embedding(max_features, embeddings_dim, input_length=maxlen, weights=[embeddings], trainable=False))
#model.add(Convolution1D(nb_filter=64, filter_length=7))#If you add 2048 filters, then it works better.
#model.add(Convolution1D(nb_filter=64, filter_length=7))
#model.add(Convolution1D(nb_filter=128, filter_length=3, activation="relu"))
model.add(AveragePooling1D(pool_length=2))
#model.add(Convolution1D(nb_filter=128, filter_length=3, activation="relu"))
#model.add(MaxPooling1D(pool_length=model.output_shape[1]))
model.add(Flatten())
#model.add(Dense(2048))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(n_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=32,
          nb_epoch=16,
          validation_data=(X_test, Y_test))

#model.fit([X_train, X_train, X_train, X_train], Y_train,
#          batch_size=32,
#          nb_epoch=5,
#          validation_data=([X_test, X_test, X_test, X_test], Y_test))


