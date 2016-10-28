#!/usr/bin/env python3

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
import sys, time


# def char_ngrams(s, ngmin=1, ngmax=1):
#     ngrams = [[] for x in range(ngmin, ngmax + 1)]
#     for i, ch in enumerate(s):
#         for ngsize in range(ngmin, ngmax + 1):
#             if (i + ngsize) <= len(s):
#                 ngrams[ngsize - 1].append(s[i:i+ngsize])
#     return ngrams

ngmin = 1 
ngmax = 5

if len(sys.argv) == 3:
    ngmin = int(sys.argv[1])
    ngmax = int(sys.argv[2])
elif len(sys.argv) == 2:
    ngmax = int(sys.argv[1])

print("ngmin = {}, ngmax = {}".format(ngmin, ngmax))

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


print("Creating the doc-term matrix... ", end="")
sys.stdout.flush()
pt = time.process_time()
v = CountVectorizer(tokenizer=char_tokenizer, ngram_range=(ngmin,ngmax))
v.fit(doc_train + doc_test)
x_train = v.transform(doc_train)
x_test = v.transform(doc_test)
# acc = cross_val_score(m, docterm, labels, cv=10, n_jobs=-1)
print(time.process_time() - pt)

print("Training... ", end="")
sys.stdout.flush()
pt = time.process_time()
m = LogisticRegression(solver="sag", max_iter=200)
m.fit(x_train, y_train)
print(time.process_time() - pt)


print("Testing... ")
pt = time.process_time()
print(m.score(x_test, y_test))
print("Testing took {} seconds.".format(time.process_time() - pt))
