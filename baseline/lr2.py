#!/usr/bin/env python3

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
import sys, time, re

from optparse import OptionParser

opt = OptionParser
opt.add_option("-v", "--vectorizer", dest="vectorizer")
opt.add_option("-f", "--min-df", dest="min_df", type="int")
opt.add_option("-c", "--max-char-ng", dest="char_ngmax")
opt.add_option("-w", "--max-word-ng", dest="word_ngmax")
opt.add_option("-o", "--optimizer", dest="optimizer")
opt.add_option("-m", "--multi-class", dest="multi_class")
(options, ars) = opt.parse_args()

vectorizer = CountVectorizer
min_df = 1
optimizer = "liblinear"
char_ngmax = 6
word_ngmax = 2
if options.vectorizer and options.vectorizer.startswith("tf"):
    vectorizer = TfidfVectorizer
if options.min_df:
    min_df = options.min_df
if options.optimizer:
    optimizer = options.optimizer
if options.word_ngmax:
    word_ngmax = options.word_ngmax
if options.char_ngmax:
    char_ngmax = options.char_ngmax
if options.multi_class:
    multi_class = options.muti_class

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

print("Creating the doc-term matrix... ", end="")
sys.stdout.flush()
pt = time.process_time()
v = vectorizer(analyzer=get_features, min_df=min_df)
v.fit(doc_train)
x_train = v.transform(doc_train)
x_test = v.transform(doc_test)
# acc = cross_val_score(m, docterm, labels, cv=10, n_jobs=-1)
print(time.process_time() - pt)
print("number of features: {}".format(len(v.vocabulary_)))
print(v)


print("Training... ", end="")
sys.stdout.flush()
pt = time.process_time()
m = LogisticRegression(solver=optimizer, n_jobs=n_jobs,
        multi_class=multi_class)
m.fit(x_train, y_train)
print(time.process_time() - pt)

print("Testing... ")
pt = time.process_time()
print(m.score(x_test, y_test))
print("Testing took {} seconds.".format(time.process_time() - pt))
