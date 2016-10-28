#!/usr/bin/env python3

import sys, time

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer


data_path = "../data"

# read in the data

def read_data(data_file):
    labels = list()
    documents = list()
    char_dict = dict()
    with open(data_file, "r") as fp:
        linenum = 0
        for line in fp:
            if len(line) < 3: # skip empty lines and ^Z at the end
                continue
            doc, label = line.strip().split("\t")
            for ch in doc:
                char_dict[ch] = char_dict.get(ch, 0) + 1
            labels.append(label)
            documents.append(doc)
            linenum += 1
    return (documents, labels, char_dict)

def doc_to_int(documents, char_dict):
    documents_int = []
    char_index = sorted(char_dict.items(), key=lambda x: x[1],reverse=True)
    char_index = dict([(char_index[i][0], i + 1) for i in range(len(char_index))])
    for doc in documents:
        di = []
        for ch in doc:
            di.append(char_index.get(ch, 0))
        documents_int.append(di)
    return documents_int

print("Reading the training set... ", end="")
sys.stdout.flush()
pt = time.process_time()
doc_train, y_train, char_index = read_data(data_path + "/task1-train.txt")
print(time.process_time() - pt)

output_encoder = LabelBinarizer()
output_encoder.fit(y_train)
y_train = output_encoder.transform(y_train)

print("Reading the test set... ", end="")
sys.stdout.flush()
pt = time.process_time()
doc_test, y_test, _ = read_data(data_path + "/task1-dev.txt")
print(time.process_time() - pt)

y_test = output_encoder.transform(y_test)


max_doc_len = 500
embedding_dim = 30
nb_filter = 100
conv_activation = 'relu'
filter_len = 5
hidden_dim = 100
hidden_activation = 'relu'

x_train = doc_to_int(doc_train, char_index)
x_test = doc_to_int(doc_test, char_index)

x_train = sequence.pad_sequences(x_train, maxlen=max_doc_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_doc_len)

m = Sequential()

m.add(Embedding(input_dim=len(char_index) + 1,
                output_dim=embedding_dim,
                input_length=max_doc_len))

m.add(Convolution1D(nb_filter=nb_filter,
                    filter_length=filter_len,
                    border_mode='valid',
                    activation=conv_activation))
m.add(Dropout(0.1))

m.add(MaxPooling1D(pool_length=m.output_shape[1]))
m.add(Flatten())
m.add(Dense(hidden_dim, activation=hidden_activation))
m.add(Dropout(0.1))
m.add(Dense(len(output_encoder.classes_), activation='sigmoid'))
m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

m.fit(x_train, y_train, validation_data=(x_test, y_test),
        batch_size=64, nb_epoch=50)

