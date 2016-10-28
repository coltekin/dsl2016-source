# -*- coding: utf-8 -*--
"""
Performs classification of dialect data. Total of 
"""
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys, random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
random.seed(1337)
from collections import defaultdict
import itertools as it
import codecs
#from preprocessing import load_data

nb_epoch = 3
batch_size = 128
maxlen = 400


def load_data(fname, maxlen, vocab=None, unique_labels=None):
    exclude_chars = [u"«", u"»", u"°",u"…", u"❤", u"º", u"¿", u"•", u"™", u"▪", u"♦", u"■", u"●", u"▲", u"▼", u"›", u"ﾃ", u"½", u"®", u"²"]
    
    char_dict = defaultdict(int)
    segments, labels, unique_chars = [], [], []
    
    n_example = 0
    for line in codecs.open(fname,"r","utf-8"):
        line = line.strip()

        if "\t" not in line:
            continue
        
        segment, label = line.split("\t")
        labels.append(label)
        segment = segment
        clone_segment = []
        for x in segment:
            if x in exclude_chars:
                char_dict[x] = 0
                continue
            elif x == '':
                print "Null character ", segment
            else:
                if x.isalpha():
                    y = x.lower()
                    clone_segment.append(y)
                    char_dict[y] += 1
                    if y == '': print "Null character ", x, y
                    if vocab and y not in vocab:
                        print "Development ", y
                else:
                    clone_segment.append(x)
                    char_dict[x] += 1
                    if vocab and x not in vocab:
                        print "Development ", x
                        
        #segments.append(segment)
        clone_segment = "".join(clone_segment)
        clone_segment = clone_segment.strip()
        len_segment = len(clone_segment)
        
        if len_segment < maxlen:
            clone_segment = clone_segment.center(maxlen,"0")
        else:
            clone_segment = clone_segment[:maxlen]
        
        segments.append(clone_segment)
        
        if not unique_labels:
            unique_labels = []
        if label not in unique_labels:
            unique_labels.append(label)
        
        #if n_example == 2649:
        #    print segment, "\n"
        #    print clone_segment
        n_example += 1
    if vocab is None:
        segments = [[s if char_dict[s] > 5 else "UNK" for s in x] for x in segments]
    else:
        segments = [[s if s in vocab else "UNK" for s in x] for x in segments]
    
    labels = [unique_labels.index(l) for l in labels]
    print "Unique labels "
    print unique_labels
    f = codecs.open(fname+".vectors", "w","utf8")
    for x, y in zip(segments, labels):
        f.write("\t".join(x)+"\t"+str(y)+"\n")
        
    for s in segments:
        for x in s:
            if x == u"": print s
            if x not in unique_chars:
                unique_chars.append(x)
    
    f.close()
    if not vocab:
        f1 = codecs.open("train_chars.txt", "w", "utf8")
        f1.write("\t".join(unique_chars)+"\t"+str(len(unique_chars))+"\n")
    #    f1.write("\t".join(char_dict.keys())+"\t"+str(len(unique_chars))+"\n")
        f1.close()
    segments = []
    
    return unique_chars, len(unique_labels), unique_labels

def wrd_to_2d(w, unique_chars):
    w2d = []
    for x in w:
        temp = len(unique_chars)*[0]
        if x == "0":
            w2d.append(temp)
        else:
            idx = unique_chars.index(x)
            temp[idx] = 1
            w2d.append(temp)
    return np.array(w2d).T


def generate_arrays_from_file(path):
    while 1:
        f = codecs.open(path, "r", "utf8")
        for line in f:
            # create Numpy arrays of input data
            # and labels, from each line in the file
            line = line.strip()
            x, y  = line.split("\t")
            
            w2d = []
            for w in x.split("|||"):
                temp = len(train_chars)*[0]
                if w == "0":
                    w2d.append(temp)
                else:
                    idx = train_chars.index(w)
                    temp[idx] = 1
                    w2d.append(temp)
            
            x = np.array(w2d).T
            x = x.reshape(1, 1, len(train_chars), maxlen)
           
            z = np.zeros((1, n_train_classes))
            z[int(y)] = 1.
            #print z
            yield (x, z)
        f.close()

train_chars, n_train_classes, unique_labels = load_data("DSL-training/task1-train.shuf.txt", maxlen)
test_chars, n_test_classes, unique_labels = load_data("DSL-training/task1-dev.txt", maxlen, vocab=train_chars, unique_labels=unique_labels)

ndim = len(train_chars)


#sys.exit(1)
print('Build model...')
model = Sequential()

model.add(Convolution2D(32,ndim,3,input_shape=(1, ndim, maxlen)))
model.add(MaxPooling2D(pool_size=(1, 3)))
model.add(Convolution2D(32,1,3))
model.add(MaxPooling2D(pool_size=(1, 3)))
model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_train_classes, activation = 'softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

###############Training###################
f = codecs.open("DSL-training/task1-train.shuf.txt.vectors", "r", "utf8")
data_lines = f.readlines()
n_samples = len(data_lines)

print "No. of samples= ", n_samples, "No. of classes= ", n_train_classes
for k in range(nb_epoch):
    random.shuffle(data_lines)
    n_example = 0
    epoch_accuracy = 0.0
    for n_batch in range(0,n_samples,batch_size):
        X, Y = [], []
        
        for line in data_lines[n_batch:n_batch+batch_size]:
            line = line.strip()
            arr  = line.split("\t")
            x, y = arr[0:-1], arr[-1]
            w2d = []
            #for w in x.split("@#@"):
            #print "Label ", y
            for w in x:
                temp = len(train_chars)*[0]
                if w == "0":
                    w2d.append(temp)
                else:
                    if w == '':
                        print x, n_example
                    idx = train_chars.index(w)
                    temp[idx] = 1
                    w2d.append(temp)
            
            z = np.array(w2d).T
            Y.append([int(y)])
            X.append(z)
            #print n_example, z.shape, Y.shape, len(X)
            n_example+=1
        try:
            X = np.array(X)
        except:
            for p in data_lines[n_batch:n_batch+batch_size]: print "".join(p.split("\t")[:-1])
            sys.exit(1)
        X = X.reshape(len(X), 1, ndim, maxlen)
        Y = np_utils.to_categorical(np.array(Y), n_train_classes)
        loss, acc = model.train_on_batch(X, Y)
        epoch_accuracy += acc
        #print n_batch, acc
    n_batches = n_samples/batch_size
    print "Epoch Accuracy ",epoch_accuracy/n_batches, k, "iteration in ", n_batches, " batches"


##########TESTING#######
f = codecs.open("DSL-training/task1-dev.txt.vectors", "r", "utf8")
data_lines = f.readlines()
n_samples = len(data_lines)
n_example = 0
print "No. of samples= ", n_samples, "No. of classes= ", n_test_classes
X_test, Y_test = [], []
for line in data_lines:
    line = line.strip()
    arr  = line.split("\t")
    x, y = arr[0:-1], arr[-1]
    w2d = []
    for w in x:
        temp = len(train_chars)*[0]
        if w == "0":
            w2d.append(temp)
        else:
            if w == '':
                print x, n_example
            idx = train_chars.index(w)
            temp[idx] = 1
            w2d.append(temp)
            
    z = np.array(w2d).T
    Y_test.append([int(y)])
    #Y_test[int(y)] = 1.
    X_test.append(z)
    #print n_example, z.shape, Y.shape, len(X)
    n_example+=1
try:
    X_test = np.array(X_test)
except:
    sys.exit(1)
X_test = X_test.reshape(len(X_test), 1, ndim, maxlen)
Y_test = np_utils.to_categorical(np.array(Y_test), n_train_classes)
loss_acc = model.evaluate(X_test, Y_test)
print "Validation loss accuracy ",loss_acc




