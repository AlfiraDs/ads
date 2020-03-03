import itertools

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from tqdm import trange
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_rnd_context(sentences, window=5):
    sent = sentences[np.random.randint(0, len(sentences) - 1)]
    wordID = np.random.randint(0, len(sent) - 1)

    o_words = sent[max(0, wordID - window):wordID]
    if wordID + 1 < len(sent):
        o_words = np.concatenate([o_words, sent[wordID + 1:min(len(sent), wordID + window + 1)]])

    c_word = sent[wordID]
    o_words = [w for w in o_words if w != c_word]

    if len(o_words) > 0:
        return c_word, o_words
    else:
        return get_rnd_context(sentences, window)


def naiveSoftmaxLossAndGradient_vect(c_vecs_bs, o_vecs, y_true, words=None):
    c_vecs_bs = torch.from_numpy(c_vecs_bs).requires_grad_(True)
    o_vecs = torch.from_numpy(o_vecs).requires_grad_(True)
    y_true = torch.from_numpy(y_true).requires_grad_(True)

    dot = torch.mm(o_vecs, c_vecs_bs)
    dot_exp = torch.exp(dot)
    probs = dot_exp / dot_exp.sum(dim=0)
    # loss = torch.nn.functional.softmax(dot, dim=0)[o_word]
    loss = -(y_true * torch.log(probs)).sum()
    loss.backward()
    c_bs_grad = c_vecs_bs.grad
    o_grad = o_vecs.grad
    return loss.detach().numpy(), c_bs_grad.numpy(), o_grad.numpy()


def getNegativeSamples(y_true, words, K=5):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = np.zeros_like(y_true)
    for k in range(K):
        for i in range(y_true.shape[1]):
            newidx = words[np.random.randint(0, len(words) - 1)]
            while newidx in np.nonzero(y_true[:, i])[0]:
                newidx = words[np.random.randint(0, len(words) - 1)]
            negSampleWordIndices[newidx, i] = 1
    return negSampleWordIndices


def negSamplingLossAndGradient_vect(c_vecs_bs, o_vecs, y_true, words):
    c_vecs_bs = torch.from_numpy(c_vecs_bs).requires_grad_(True)
    o_vecs = torch.from_numpy(o_vecs).requires_grad_(True)
    y_true = torch.from_numpy(y_true).requires_grad_(False)

    loss = torch.sigmoid(torch.mm(o_vecs, c_vecs_bs)) * y_true
    loss = -torch.log(loss[loss != 0]).sum()

    negSampleWordIndices = getNegativeSamples(y_true.numpy(), words)
    negSampleWordIndices = torch.from_numpy(negSampleWordIndices).requires_grad_(True)
    # indices = [o_word] + negSampleWordIndices
    loss_1 = torch.sigmoid(torch.mm(o_vecs, c_vecs_bs)) * negSampleWordIndices
    loss -= torch.log(loss_1[loss_1 != 0]).sum()

    loss.backward()
    c_grad = c_vecs_bs.grad
    o_grad = o_vecs.grad

    return loss.detach().numpy(), c_grad.numpy(), o_grad.numpy()


def fn(c_vecs, o_vecs, sentences, words, bs=50, ws=5):
    c_vecs_bs = np.empty((c_vecs.shape[1], bs))
    y_true = np.zeros((o_vecs.shape[0], bs))
    for i in range(bs):
        c_word, o_words = get_rnd_context(sentences, np.random.randint(1, ws))
        c_vecs_bs[:, i] = c_vecs[c_word]
        y_true[o_words, i] = 1
    # loss, c_grads, o_grads = naiveSoftmaxLossAndGradient_vect(c_vecs_bs, o_vecs, y_true, words=words)
    loss, c_grads, o_grads = negSamplingLossAndGradient_vect(c_vecs_bs, o_vecs, y_true, words=words)
    loss /= bs
    c_grads = c_grads.sum(axis=1) / bs
    o_grads /= bs
    return loss, c_grads, o_grads


def sgd(c_vecs, o_vecs, sentences, words, lr=0.3, niter=20000):
    for i in trange(niter):
        loss, c_grads, o_grads = fn(c_vecs, o_vecs, sentences, words)
        c_vecs -= lr * c_grads
        o_vecs -= lr * o_grads
        if i % 1000 == 0:
            print(loss)
        if i % 20000 == 0:
            lr *= 0.5
    return c_vecs


def visualize(visualizeWords, visualizeIdx, c_vecs):
    visualizeVecs = c_vecs[visualizeIdx, :]
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp.dot(U[:, 0:2])

    for i in range(len(visualizeWords)):
        plt.text(coord[i, 0], coord[i, 1], visualizeWords[i],
                 bbox=dict(facecolor='green', alpha=0.1))

    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

    # plt.savefig('word_vectors_1.png')
    plt.show()


if __name__ == '__main__':
    # df = pd.read_csv('/Users/aliaksandr.lashkov/work/projects/kaggle.House Prices/data/data.csv')
    n_sents = 100000
    n_words = 1000
    dim = 10
    max_sent_len = 20
    sentences = np.random.randint(0, n_words, (n_sents, max_sent_len))
    words = list(itertools.chain.from_iterable(sentences))
    c_vecs = (np.random.rand(n_words, dim) - 0.5) / dim
    o_vecs = np.zeros((n_words, dim))
    visualizeWords = [1,2,3,4,5,6,7]
    visualizeIdx = [1,2,3,4,5,6,7]
    # visualize(visualizeWords, visualizeIdx, c_vecs)

    c_vecs = sgd(c_vecs, o_vecs, sentences, words)

    visualize(visualizeWords, visualizeIdx, c_vecs)
