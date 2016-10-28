#!/usr/bin/env python3

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import cross_validation
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
import sys, time
from collections import defaultdict
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# def char_ngrams(s, ngmin=1, ngmax=1):
#     ngrams = [[] for x in range(ngmin, ngmax + 1)]
#     for i, ch in enumerate(s):
#         for ngsize in range(ngmin, ngmax + 1):
#             if (i + ngsize) <= len(s):
#                 ngrams[ngsize - 1].append(s[i:i+ngsize])
#     return ngrams

data_path = "../data"
ngmin = 1 
ngmax = 1
minfreq = 5
exclude_chars = [x.replace("\n","") for x in open("characters.txt", "r")]
print(exclude_chars)
#Clean the document
def getVocab(D):
    charSet = defaultdict(int)
    for d in D:
        for c in char_tokenizer(d):
            charSet[c] += 1
    return charSet

def cleanChars(D, charSet):
    nD = []
    for d in D:
        nd = []
        for c in d:
            #if charSet[c] >= minfreq or c not in exclude_chars:
            if charSet[c] >= minfreq:
                nd.append(c)
        nD.append("".join(nd).strip())
        #if len(d) != len(nd): print(len(d),len(nd))
        #print("Lengths before and after ", len(d), len(nd))
    return nD

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
            wrds = doc.split(" ")
            if len(wrds) >70:
                doc = " ".join(wrds[:70])
            labels.append(label)
            documents.append(doc)
            linenum += 1
    return (documents, labels)

def char_tokenizer(s):
    return list(s)

if len(sys.argv) == 3:
    ngmin = int(sys.argv[1])
    ngmax = int(sys.argv[2])
elif len(sys.argv) == 2:
    ngmax = int(sys.argv[1])

print("ngmin = {}, ngmax = {}".format(ngmin, ngmax))

print("Reading the training set... ", end="")
sys.stdout.flush()
pt = time.process_time()
doc_train, y_train = read_data(data_path + "/task1-train.txt")
print(time.process_time() - pt)

print("Reading the test set... ", end="")
sys.stdout.flush()
pt = time.process_time()
doc_test, y_test = read_data(data_path + "/tweet.dev.txt")
print(time.process_time() - pt)

charSet = getVocab(doc_train)
print("Total chars ", len(charSet.keys()))
doc_train = cleanChars(doc_train, charSet)
doc_test = cleanChars(doc_test, charSet)


print("Creating the doc-term matrix... ", end=" ")
sys.stdout.flush()
pt = time.process_time()
#v = CountVectorizer(analyzer="char", ngram_range=(ngmin,ngmax), binary=True, dtype = np.int8)#Also test binary=True
#v = TfidfVectorizer(tokenizer=char_tokenizer, ngram_range=(ngmin,ngmax),lowercase=False, sublinear_tf=True, min_df=0.001)#Also test binary=True. Binary=True does not work so well.
#v = TfidfVectorizer(analyzer="char", ngram_range=(ngmin,ngmax),lowercase=False, sublinear_tf=True, min_df=0.001)#Remove min_df for maximum performance
v = TfidfVectorizer(analyzer="char", ngram_range=(ngmin,ngmax),lowercase=False, sublinear_tf=True)
#v = TfidfVectorizer(analyzer="char", ngram_range=(ngmin,ngmax),lowercase=False, use_idf=False, binary=True)
#v.fit(doc_train + doc_test)
v.fit(doc_train)
doc_train = v.transform(doc_train)
doc_test = v.transform(doc_test)
v = None
print(time.process_time() - pt)


print("Training data shape ", doc_train.shape)
print("Training...", end=" ")
sys.stdout.flush()
pt = time.process_time()
#n_estimators = 27
m = OneVsOneClassifier(svm.LinearSVC(dual=False))
#m = OneVsOneClassifier(BaggingClassifier(svm.SVC(kernel="poly",degree=2), max_samples=1.0 / n_estimators, n_estimators=n_estimators, bootstrap=False, n_jobs=4, verbose=2))
#m = svm.SVC(kernel="poly",degree=2)
#m = MultinomialNB()
#scores = cross_validation.cross_val_score(m, doc_train, y_train, cv=5, n_jobs=5, verbose=1)
#print(scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(m.fit(doc_train, y_train))
print(time.process_time() - pt)


print("Testing... ")
pt = time.process_time()
print(m.score(doc_train, y_train)*100.0)
print(m.score(doc_test, y_test)*100.0)
print("Testing took {} seconds.".format(time.process_time() - pt))

