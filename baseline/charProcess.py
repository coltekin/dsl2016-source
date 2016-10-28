# -*- coding: utf-8 -*--
from collections import defaultdict
import operator, codecs, re

data_path = "../data"
min_freq = 10

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
    
def getVocab(D):
    charSet = defaultdict(int)
    for d in D:
        for c in char_tokenizer(d):
            charSet[c] += 1
    return charSet

doc_train, y_train = read_data(data_path + "/task1-train.txt")
doc_test, y_test = read_data(data_path + "/task1-dev.txt")
x = getVocab(doc_train)
sorted_x = sorted(x.items(), key=operator.itemgetter(1))

for k, v in sorted_x:
    print(k, v)
print("Total chars ", len(x.keys()))
