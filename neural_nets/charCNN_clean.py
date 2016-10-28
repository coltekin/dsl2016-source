"""
Performs classification of dialect data.
"""
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys, random
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution2D, MaxPooling2D, LSTM, AveragePooling2D
from keras.utils import np_utils
random.seed(1337)
from collections import defaultdict
import itertools as it
import codecs, time, re
#from preprocessing import load_data

nb_epoch = 8
batch_size = 32
maxlen = 512
minfreq = 0
data_path = "../data"

_white_spaces = re.compile(r"\s\s+")

def cleanChars(D, charSet):
    nD = []
    n_minlen = 0
    for d in D:
        nd = []
        for c in d:
            if charSet[c] > minfreq:
                nd.append(c)
        nd = "".join(nd)
        if len(nd) < maxlen:
            n_minlen += 1
            nd = nd.center(maxlen,"0")
        else:
            nd = nd[:maxlen]
        nD.append(nd)
    D=nD
    print(n_minlen," Sentences with length below ",maxlen)
    return D

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
    max_features = 0
    features = []
    for d in D:
        for c in char_tokenizer(d):
            charSet[c] += 1
    for c in charSet:
        if charSet[c] > minfreq:
            max_features += 1
            features.append(c)
    return charSet, max_features+1, features

print("Reading the training set... ", end="")
sys.stdout.flush()
pt = time.process_time()
doc_train, y_train = read_data(data_path + "/bshr.train.txt")
print(time.process_time() - pt)

print("Reading the test set... ", end="")
sys.stdout.flush()
pt = time.process_time()
doc_test, y_test = read_data(data_path + "/bshr.dev.txt")
print(time.process_time() - pt)

print("Cleaning the datasets... ", end="")
sys.stdout.flush()
pt = time.process_time()
char_vocab, maxfeatures, features = getVocab(doc_train)
print("Total chars ", len(char_vocab.keys()), "Maximum features ", maxfeatures)
doc_train = cleanChars(doc_train, char_vocab)
doc_test = cleanChars(doc_test, char_vocab)
print("Number of features= ", maxfeatures)
print("\n",features)
data_lines = list(zip(doc_train, y_train))
n_samples = len(y_train)
y_classes = list(set(y_train))
n_classes = len(y_classes)
print("Class labels = ",y_classes)
print('Build model...')
model = Sequential()
#model.add(LSTM(64,input_shape=(maxlen, maxfeatures)))
model.add(Convolution2D(32,maxfeatures,3,input_shape=(1, maxfeatures, maxlen)))
#model.add(AveragePooling2D(pool_size=(maxfeatures, model.output_shape[1])))
#model.add(Convolution2D(32,1,3))
#model.add(MaxPooling2D(pool_size=(1, 3)))
#model.add(Convolution2D(32,1,3))
#model.add(MaxPooling2D(pool_size=(1, 3)))
model.add(Flatten())

#model.add(Dense(model.output_shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation = 'softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
index_list = range(n_samples)
X_test, Y_test = None, None
print("Training...", end=" ")
print("No. of samples= ", n_samples, "No. of classes= ", n_classes)
for k in range(1, nb_epoch+1):
    random.shuffle(data_lines)
    #subsetData = data_lines[:40000]
    n_example = 0
    epoch_accuracy = 0.0
    for n_batch in range(0,n_samples,batch_size):
        X, Y = [], []
        for x, y in data_lines[n_batch:n_batch+batch_size]:
            w2d = []
            for w in x:
                temp = maxfeatures*[0]
                if w == "0":
                    w2d.append(temp)
                else:
                    if w == '':
                        print(x, n_example)
                    idx = features.index(w)+1
                    temp[idx] = 1
                    w2d.append(temp)
            
            z = np.array(w2d).T#for convolution
            #z = np.array(w2d)#for LSTM
            Y.append(y_classes.index(y))
            X.append(z)
            #print n_example, z.shape, Y.shape, len(X)
            n_example+=1
        try:
            X = np.array(X, np.int8)
        except:
            for p in data_lines[n_batch:n_batch+batch_size]: print("".join(p.split("\t")[:-1]))
            sys.exit(1)
        X = X.reshape(len(X), 1, maxfeatures, maxlen)
        Y = np_utils.to_categorical(np.array(Y), n_classes)
        loss, acc = model.train_on_batch(X, Y)
        #print('x_train shape:', X.shape)
        epoch_accuracy += acc
        
        if n_batch%1000 == 0: print("Accuracy ",acc, " in ",n_batch)
    n_batches = int(n_samples/batch_size)+1.0
    print("Epoch ", k, " Accuracy ",epoch_accuracy/n_batches, " in ", n_batches, " batches")
    #if k < nb_epoch: continue
    print("Testing...", end=" ")
    test_lines = zip(doc_test, y_test)
    n_testsamples = len(y_test)
    n_testexample = 0
    print("No. of samples= ", n_testsamples, "No. of classes= ", n_classes)
    if X_test is not None:
        loss_acc = model.evaluate(X_test, Y_test)
        print("Validation loss accuracy ",loss_acc)
        continue
    X_test, Y_test = [], []
    for x, y in test_lines:
        w2d = []
        for w in x:
            temp = maxfeatures*[0]
            if w == "0":
                w2d.append(temp)
            else:
                if w == '':
                    print(x, n_example)
                idx = features.index(w)+1
                temp[idx] = 1
                w2d.append(temp)
            
        z = np.array(w2d).T
        #z = np.array(w2d)
        Y_test.append(y_classes.index(y))
        X_test.append(z)
        #print n_example, z.shape, Y.shape, len(X)
        n_example+=1
    try:
        X_test = np.array(X_test,np.int8)
    except:
        print("Could not convert x_test")
        sys.exit(1)
    X_test = X_test.reshape(len(X_test), 1, maxfeatures, maxlen)
    Y_test = np_utils.to_categorical(np.array(Y_test), n_classes)
    loss_acc = model.evaluate(X_test, Y_test)
    print("Validation loss accuracy ",loss_acc)




