import gzip
import os
import urllib.request
import numpy as np

URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
       "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz")
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "glove-wiki-gigaword-50.gz")


class GloveVectors:
    def __init__(self, words, matrix):
        self.index_to_key = words
        self.key_to_index = {w: i for i, w in enumerate(words)}
        self.vectors = matrix
        self.vector_size = matrix.shape[1]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._unit = matrix / np.maximum(norms, 1e-9)

    def __contains__(self, word):
        return word in self.key_to_index

    def __getitem__(self, word):
        return self.vectors[self.key_to_index[word]]

    def most_similar(self, positive=(), negative=(), topn=5):
        if isinstance(positive, str):
            positive = [positive]
        if isinstance(negative, str):
            negative = [negative]
        query = sum(self._unit[self.key_to_index[w]] for w in positive)
        query = query - sum(self._unit[self.key_to_index[w]] for w in negative) if negative else query
        query = query / max(np.linalg.norm(query), 1e-9)
        sims = self._unit @ query
        exclude = {self.key_to_index[w] for w in list(positive) + list(negative)}
        out = []
        for i in np.argsort(-sims):
            if int(i) not in exclude:
                out.append((self.index_to_key[i], float(sims[i])))
                if len(out) == topn:
                    break
        return out


def load_glove(path=DEFAULT_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading GloVe (~66 MB) ...")
        urllib.request.urlretrieve(URL, path)
    words, vecs = [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first = f.readline().split()
        if len(first) != 2:
            f.seek(0)
        for line in f:
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vecs.append(np.asarray(parts[1:], dtype=np.float32))
    return GloveVectors(words, np.vstack(vecs))