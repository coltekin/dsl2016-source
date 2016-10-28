import os
import json
import numpy as np
from gensim.models import Word2Vec

data_path = "../wordembeddings"
from gensim.utils import simple_preprocess
tokenize = lambda x: simple_preprocess(x)

def create_embeddings(data_dir, embeddings_path, vocab_path, **params):
    class SentenceGenerator(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            for fname in os.listdir(self.dirname):
                for line in open(os.path.join(self.dirname, fname)):
                    yield tokenize(line)

    sentences = SentenceGenerator(data_dir)

    model = Word2Vec(sentences, **params)
    weights = model.syn0
    np.save(open(embeddings_path, 'wb'), weights)

    vocab = dict([(k, v.index) for k, v in model.vocab.items()])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))
        
create_embeddings(data_path, data_path + "/embeddings.npz", data_path +  "/map.json", size=50, min_count=0, window=5, sg=1, iter=10)
