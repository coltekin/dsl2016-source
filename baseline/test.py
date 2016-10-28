#!/usr/bin/env python3

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
import sys, time
from collections import defaultdict
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import joblib

import warnings
warnings.filterwarnings("ignore")

vectorizer = sys.argv[1]
model_file = sys.argv[2]
test_file = sys.argv[3]

data_path = "../data/"

# read in the data
def read_data(data_file):
    documents = []
    with open(data_file, "r") as fp:
        linenum = 0
        for line in fp:
            if len(line) < 3: # skip empty lines and ^Z at the end
                continue
            doc = line.split()
            if len(doc) > 70:
                doc = doc[:70]
            documents.append(" ".join(doc))
            linenum += 1
    return documents

print("Reading the test set... ", end="")
sys.stdout.flush()
pt = time.process_time()
doc_test = read_data(data_path + test_file)
print(time.process_time() - pt)

v = joblib.load(vectorizer)
doc_test = v.transform(doc_test)

m = joblib.load(model_file)

print(m.predict(doc_test))
