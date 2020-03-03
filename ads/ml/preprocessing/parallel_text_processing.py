import multiprocessing
from itertools import chain
from multiprocessing.pool import Pool

from nltk import WordNetLemmatizer
from tqdm import tqdm

import re
from nltk.corpus import stopwords
import time
from sklearn.datasets import fetch_20newsgroups

global_stop_words = stopwords.words('english')

train = fetch_20newsgroups(subset="all")
text = " ".join(train.data)
lemmatizer = WordNetLemmatizer()
num_cores = multiprocessing.cpu_count()


def get_bathes(tokens, n):
    bathes = []
    idx = len(tokens) // n
    idxs = []
    for i in range(n):
        start = i * idx
        end = (i + 1) * idx
        if i + 1 == n:
            end = len(tokens)
        idxs.append((start, end))
        bathes.append(tokens[start: end])
    assert len(set(tokens)) == len(set(chain.from_iterable(bathes)))
    return bathes


def process_tokens(tokens):
    words = []
    for word in tqdm(tokens):
        word = lemmatizer.lemmatize(word)
        condition = word not in global_stop_words
        condition = condition and len(word) >= 3
        if condition:
            words.append(word)
    return words


def process_tokens_parallel(bathes, cores):
    with Pool(cores) as pool:
        df = pool.map(process_tokens, bathes)
        pool.close()
        pool.join()
        return df


tokens = re.split("\W+", text.lower())
start = time.time()
print(len(set(process_tokens(tokens))))
print("taken", time.time() - start)

bathes = get_bathes(tokens, n=num_cores)
start = time.time()
print(len(set(chain.from_iterable(process_tokens_parallel(bathes, num_cores)))))
print("taken", time.time() - start)
