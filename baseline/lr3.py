#!/usr/bin/env python3

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
import numpy as np
import sys, time, re, os.path
import joblib


import warnings
warnings.filterwarnings("ignore")

# simple tokenizer to extract sequences of letters, or non-space
# sinle symbols
tokenizer_re = re.compile("\w+|\S")

def get_ngrams(s, ngmin=1, ngmax=1, tokenizer=list, separator="|"):
    """ Return all ngrams with in range ngmin-ngmax.
        The function specified by 'tokenizer' should return a list of
        tokens. By default, the tokenizer splits the string into 
        its characters.
    """
    ngrams = [[] for x in range(ngmin, ngmax + 1)]
    s = tokenizer(s)
    for i, ch in enumerate(s):
        for ngsize in range(ngmin, ngmax + 1):
            if (i + ngsize) <= len(s):
                ngrams[ngsize - 1].append(separator.join(s[i:i+ngsize]))
    return ngrams

# from: https://gist.github.com/zachguo/10296432
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[8]) # 8 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end="")
    for label in labels: 
        print("%{0}s".format(columnwidth) % label, end="")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end="")
        for j in range(len(labels)): 
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end="")
        print()

def get_features(s):
    features = get_ngrams(s, ngmax=char_ngmax)
    word_feat = get_ngrams(s, ngmax=word_ngmax, tokenizer=tokenizer_re.findall) 
    features += ['<' + x + '>' for sublist in word_feat for x in sublist]
    return [x for sublist in features for x in sublist]

data_path = "../data"

# read in the data

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

print(set(y_train))
print(set(y_test))

def read_cached(name,  min_df, binary, char_ngmax, word_ngmax):
    fname = ('data/' + name + "-" +
             str(min_df) + "-" +
             str(binary) + "-" +
             str(char_ngmax) + "-" +
             str(word_ngmax))
    if os.path.isfile(fname):
        print("Reading {}... ".format(fname))
        d = joblib.load(fname)
        return d
    else:
        return None

def write_cached(obj, name, min_df, binary, char_ngmax, word_ngmax):
    fname = ('data/' + name + "-" +
             str(min_df) + "-" +
             str(binary) + "-" +
             str(char_ngmax) + "-" +
             str(word_ngmax))
    joblib.dump(obj, fname, compress=3)

for vectorizer in (CountVectorizer, TfidfVectorizer):
    for min_df in (5, 3, 2):
        for binary in (True, False):
            for char_ngmax in (3, 4, 5, 6):
                for word_ngmax in (2, 3):
                    v = read_cached(vectorizer.__name__, min_df, 
                                    binary, char_ngmax, word_ngmax)
                    if v is None:
                        if vectorizer is CountVectorizer:
                            if binary:
                                dtype = np.bool_
                            else:
                                dtype = np.uint32
                        else:
                            dtype = np.float32
                        print("Creating the doc-term matrix... ", end="")
                        sys.stdout.flush()
                        v = vectorizer(analyzer=get_features,
                                lowercase=False, min_df=min_df, dtype=dtype)
                        v.fit(doc_train)
                        write_cached(v, vectorizer.__name__, min_df, 
                                    binary, char_ngmax, word_ngmax)
                        print("{} ...".format(time.process_time() - pt), end="")
                        sys.stdout.flush()
                        x_train = v.transform(doc_train)
                        write_cached(x_train, "train", min_df, binary, 
                                    char_ngmax, word_ngmax)
                        x_test = v.transform(doc_test)
                        write_cached(x_test, "test", min_df, binary,
                                char_ngmax, word_ngmax)
                    else:
                        x_train = read_cached("train", min_df,
                                binary, char_ngmax, word_ngmax)
                        if x_train is None:
                            x_train = v.transform(doc_train)
                            write_cached(x_train, "train", min_df, 
                                        binary, x_train, word_ngmax)
                        x_test = read_cached("test", min_df,
                                binary, char_ngmax, word_ngmax)
                        if x_test is None:
                            x_test = v.transform(doc_test)
                            write_cached(x_test, "test", min_df, 
                                        binary, char_ngmax, word_ngmax)

                    print(time.process_time() - pt)

                    print("number of features: {}".format(len(v.vocabulary_)))
                    print(v)
                    v = None

                    for classifier in (LogisticRegression, LinearSVC):
                        for mclass in (OneVsOneClassifier, OneVsRestClassifier):
                            print("vectorizer: {}".format(vectorizer))
                            print("char_ngmax = {}, word_ngmax = {}".format(
                                char_ngmax, word_ngmax))
                            print("classfier: {}".format(classifier))
                            print()

                            print("Training... ", end="")
                            sys.stdout.flush()
                            pt = time.process_time()
                            m = mclass(classifier(), n_jobs=-1)
                            print(m)
                            m.fit(x_train, y_train)
                            write_cached(m,
                                "model" + "-" + classifier.__name__ +
                                "-" + mclass.__name__,
                                min_df, binary, char_ngmax, word_ngmax)
                            print(time.process_time() - pt)
                            print(classifier)

                            print("Testing... ")
                            pt = time.process_time()
                            print("Accuracy: {}".format(m.score(x_test, y_test)))
                            print_cm(confusion_matrix(m.predict(x_test), y_test), m.classes_)
                            print("Testing took {} seconds.".format(
                                time.process_time() - pt))
