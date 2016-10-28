#!/usr/bin/env python3

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
import sys, time, re

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


data_path = "../data"

# read in the data

def read_data(data_file):
    d = dict()
    documents = []
    doc_labels = []
    with open(data_file, "r") as fp:
        for line in fp:
            if len(line) < 3: # skip empty lines and ^Z at the end
                continue
            doc, label = line.strip().split("\t")
            documents.append(doc)
            doc_labels.append(label)
            for w in tokenizer_re.findall(doc):
                d[(w, label)] = d.get((w, label), 0) + 1
    return (d, documents, doc_labels)

print("Reading the training set... ", end="")
sys.stdout.flush()
pt = time.process_time()
train_data, _, _ = read_data(data_path + "/task1-train.txt")
print(time.process_time() - pt)

print("Reading the test set... ", end="")
sys.stdout.flush()
pt = time.process_time()
test_data, test_docs, doc_labels = read_data(data_path + "/task1-dev.txt")
print(time.process_time() - pt)

ngmin = 1
ngmax = 6

w_train, y_train = tuple(zip(*train_data.keys()))
w_test, y_test = tuple(zip(*test_data.keys()))

counts = tuple(train_data[(w, l)] for (w, l) in zip(w_train, y_train))

print("Creating the 'doc-term' matrix... ", end="")
sys.stdout.flush()
pt = time.process_time()
v = CountVectorizer(tokenizer=list, ngram_range=(ngmin,ngmax))
v.fit(w_train)
x_train = v.transform(w_train)
x_test = v.transform(w_test)
print(time.process_time() - pt)

print("Training... ", end="")
sys.stdout.flush()
pt = time.process_time()
m = OneVsRestClassifier(LogisticRegression(), n_jobs=-1)
m.fit(x_train, y_train)
print(time.process_time() - pt)

print("Testing... ")
pt = time.process_time()
print(m.score(x_test, y_test))
print("Testing took {} seconds.".format(time.process_time() - pt))

from collections import Counter
doc_predicted = []
tokenizer_re2 = re.compile("\w+")
for doc, lang in zip(test_docs, y_test):
    tokens = tokenizer_re.findall(doc)
    xx_test = v.transform(tokens)
#    w_predict = m.predict(xx_test)
#    doc_predicted.append(Counter(w_predict).most_common()[0][0])
    w_predict = m.predict_proba(xx_test)
    doc_predicted.append(sorted(list(zip(m.classes_, w_predict.sum(axis=0))), key=lambda x: x[1])[-1][0])

