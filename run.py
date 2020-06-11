# Vectorising tf-idf
import math
import pickle
from collections import Counter
from typing import List

import numpy as np
from nltk.tokenize import word_tokenize

import preprocessor


# TF-IDF Cosine Similarity Ranking
def cosine_sim(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim


class Runner(object):
    D: any
    DF: dict

    def __init__(self):
        self.D = None
        self.tf_idf = None
        self.DF = None
        self.total_vocab = None
        self.total_vocab_size = None
        self.dataset = None
        self.N = None
        self.alpha = 0.3

    @classmethod
    def initialize(cls):
        res = cls()
        with open("data.pkl", "rb") as open_file:
            file_data = pickle.load(open_file)
        res.D = np.array(file_data["D"])
        res.tf_idf = preprocessor.unmap_keys(file_data["tf_idf"])
        res.DF = file_data["DF"]
        res.total_vocab = file_data["total_vocab"]
        res.total_vocab_size = file_data["total_vocab_size"]
        res.dataset = file_data["dataset"]
        res.N = file_data["N"]
        return res

    def gen_vector(self, tokens):
        Q = np.zeros((len(self.total_vocab)))

        counter = Counter(tokens)
        words_count = len(tokens)

        for token in np.unique(tokens):

            tf = counter[token] / words_count
            df = self.doc_frequency(token)
            idf = math.log((self.N + 1) / (df + 1))

            try:
                ind = self.total_vocab.index(token)
                Q[ind] = tf * idf
            except:
                pass
        return Q

    def format_doc_custom(self, ids):
        res = []
        for uid in ids:
            tuple_ = self.dataset[uid]
            location = tuple_[0]
            title = tuple_[1]
            res.append({
                "title": title,
                "location": location
            })
        return res

    def cosine_similarity(self, k, query):
        preprocessed_query = preprocessor.preprocess(query)
        tokens = word_tokenize(str(preprocessed_query))

        d_cosines = []

        query_vector = self.gen_vector(tokens)
        for d in self.D:
            d_cosines.append(cosine_sim(query_vector, d))

        out = np.array(d_cosines).argsort()[-k:][::-1]

        return self.format_doc_custom(out)

    def doc_frequency(self, word: str):
        return self.DF.get(word, 0)

    def update_df(self, processed: List[str]):
        df = {}
        for token in processed:
            try:
                df[token].add(self.N)
            except Exception as e:
                df[token] = {self.N}

        for i in df:
            if i not in self.DF:
                self.DF[i] = 0
            self.DF[i] += 1

    def calculate_tf_idf(self, processed_title: str, processed_text: str):
        tf_idf = {}
        tokens = processed_text

        counter = Counter(tokens + processed_title)
        words_count = len(tokens + processed_title)

        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = self.doc_frequency(token)
            idf = np.log((self.N + 1) / (df + 1))

            tf_idf[self.N, token] = tf * idf
        return tf_idf

    def reload_d(self):
        self.D = np.zeros((self.N, self.total_vocab_size))
        for i in self.tf_idf:
            try:
                ind = self.total_vocab.index(i[1])
                self.D[i[0]][ind] = self.tf_idf[i]
            except:
                pass

    def save_data(self):
        file_data = {
            "D": self.D.tolist(),
            "tf_idf": preprocessor.remap_keys(self.tf_idf),
            "DF": self.DF,
            "total_vocab": self.total_vocab,
            "total_vocab_size": self.total_vocab_size,
            "dataset": self.dataset,
            "N": self.N
        }
        with open("data.pkl", "wb") as open_file:
            pickle.dump(file_data, open_file, protocol=pickle.HIGHEST_PROTOCOL)

    def process_new_text(self, title: str, body: str):
        self.dataset.append(("uploads/{}".format(title), title))
        processed_title = preprocessor.process_text(title)
        processed_body = preprocessor.process_text(body)
        processed = []
        processed.extend(processed_title)
        processed.extend(processed_body)
        self.update_df(processed)
        self.N += 1
        self.total_vocab_size = len(self.DF)
        self.total_vocab = list(self.DF.keys())
        tf_idf = self.calculate_tf_idf(processed_title, processed_body)
        tf_idf_title = self.calculate_tf_idf(processed_body, processed_title)
        for i in tf_idf:
            tf_idf[i] *= self.alpha
        for i in tf_idf_title:
            tf_idf[i] = tf_idf_title[i]
        for i in tf_idf:
            self.tf_idf[i] = tf_idf[i]
        self.reload_d()
        self.save_data()
