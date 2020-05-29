#!/usr/bin/python3
import os
import pickle
import re
from collections import Counter

import numpy as np
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt')
import preprocessor

title = "stories"
alpha = 0.3

###################################Taking all folders

folders = [x[0] for x in os.walk(str(os.getcwd()) + '/' + title + '/')]
folders[0] = folders[0][:len(folders[0]) - 1]
# print(folders)

#######Collecting the file names and titles

dataset = []

c = False


def parse_from_generic_files():
    for folder in folders:
        files = os.listdir(folder)
        for file_name in files:
            file_path = "{}/{}".format(str(folder), file_name)
            with open(file_path, "r") as open_file:
                dataset.append((file_path, file_name))


#parse_from_generic_files()


def parse_from_index():
    c = False
    for i in folders:
        file = open(i + "/index.html", 'r')
        text = file.read().strip()
        file.close()

        file_name = re.findall('><A HREF="(.*)">', text)
        file_title = re.findall('<BR><TD> (.*)\n', text)

        if c == False:
            file_name = file_name[2:]
            c = True

        # print(len(file_name), len(file_title))

        for j in range(len(file_name)):
            dataset.append((str(i) + "/" + str(file_name[j]), file_title[j]))

parse_from_index()
# print("Length of dataset: ", len(dataset))

N = len(dataset)


def print_doc(id):
    print(dataset[id])
    file = open(dataset[id][0], 'r', encoding='cp1250')
    text = file.read().strip()
    file.close()
    print(text)


# Preprocessing


# Extracting Data

processed_text = []
processed_title = []

for i in dataset[:N]:
    file = open(i[0], 'r', encoding="utf8", errors='ignore')
    text = file.read().strip()
    file.close()

    processed_text.append(word_tokenize(str(preprocessor.preprocess(text))))
    processed_title.append(word_tokenize(str(preprocessor.preprocess(i[1]))))

# Calculating DF for all words

DF = {}

for i in range(N):
    tokens = processed_text[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}

    tokens = processed_title[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
for i in DF:
    DF[i] = len(DF[i])

total_vocab_size = len(DF)

# print("Total vocabulary size: ", total_vocab_size)

total_vocab = [x for x in DF]


# print(total_vocab[:20])


def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


# Calculating TF-IDF for body, we will consider this as the actual tf-idf as we will add the title weight to this.


doc = 0

tf_idf = {}

for i in range(N):

    tokens = processed_text[i]

    counter = Counter(tokens + processed_title[i])
    words_count = len(tokens + processed_title[i])

    for token in np.unique(tokens):
        tf = counter[token] / words_count
        df = doc_freq(token)
        idf = np.log((N + 1) / (df + 1))

        tf_idf[doc, token] = tf * idf

    doc += 1

# Calculating TF-IDF for Title

doc = 0

tf_idf_title = {}

for i in range(N):

    tokens = processed_title[i]
    counter = Counter(tokens + processed_text[i])
    words_count = len(tokens + processed_text[i])

    for token in np.unique(tokens):
        tf = counter[token] / words_count
        df = doc_freq(token)
        idf = np.log((N + 1) / (df + 1))  # numerator is added 1 to avoid negative values

        tf_idf_title[doc, token] = tf * idf

    doc += 1

# print(tf_idf[(0,"go")])

# print(tf_idf_title[(0,"go")])


# Merging the TF-IDF according to weights

for i in tf_idf:
    tf_idf[i] *= alpha

for i in tf_idf_title:
    tf_idf[i] = tf_idf_title[i]

# print(len(tf_idf))
D = np.zeros((N, total_vocab_size))
for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass


def save_data():
    file_data = {
        "D": D.tolist(),
        "tf_idf": preprocessor.remap_keys(tf_idf),
        "DF": DF,
        "total_vocab": total_vocab,
        "total_vocab_size": total_vocab_size,
        "dataset": dataset,
        "N": N
    }
    with open("data.pkl", "wb") as open_file:
        pickle.dump(file_data, open_file, protocol=pickle.HIGHEST_PROTOCOL)


save_data()
# Q = cosine_similarity(10, "Without the drive of Rebeccah's insistence, Kate lost her momentum. She stood next a slatted oak bench, canisters still clutched, surveying")


# Q = cosine_similarity(int(sys.argv[1]), str(sys.argv[2]))


print("Finished")
