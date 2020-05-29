# Vectorising tf-idf
import codecs
import json
import math
import pickle
from collections import Counter

import numpy as np
from nltk.tokenize import word_tokenize

import preprocessor

D = None
tf_idf = None
DF = None
total_vocab = None
total_vocab_size = None
dataset = None
N = None


def load_data():
    with open("data.pkl", "rb") as open_file:
        file_data = pickle.load(open_file)
    global D, tf_idf, DF, total_vocab, total_vocab_size, dataset, N
    D = np.array(file_data["D"])
    tf_idf = preprocessor.unmap_keys(file_data["tf_idf"])
    DF = file_data["DF"]
    total_vocab = file_data["total_vocab"]
    total_vocab_size = file_data["total_vocab_size"]
    dataset = file_data["dataset"]
    N = file_data["N"]


load_data()


def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


# TF-IDF Cosine Similarity Ranking
def cosine_sim(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim


def gen_vector(tokens):
    Q = np.zeros((len(total_vocab)))

    counter = Counter(tokens)
    words_count = len(tokens)

    query_weights = {}

    for token in np.unique(tokens):

        tf = counter[token] / words_count
        df = doc_freq(token)
        idf = math.log((N + 1) / (df + 1))

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf * idf
        except:
            pass
    return Q


def print_doc_custom(ids):
    for id in ids:
        tuple_ = dataset[id]
        location = tuple_[0]
        title = tuple_[1]
        print(title)
        # print(location)
        print("")


def cosine_similarity(k, query):
    #print("Cosine Similarity")
    preprocessed_query = preprocessor.preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))

    #print("\nQuery:", query)
    #print("")
    #print(tokens)

    d_cosines = []

    query_vector = gen_vector(tokens)
    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]

    #print("")

    #print(out)

    print("")

    print_doc_custom(out)
