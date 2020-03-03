import itertools

import numpy as np
import torch
from tqdm import trange


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


def naiveSoftmaxLossAndGradient(c_vec, o_vecs, o_word, words=None):
    centerWordVec = torch.from_numpy(c_vec).requires_grad_(True)
    outsideVectors = torch.from_numpy(o_vecs).requires_grad_(True)
    loss = torch.nn.functional.softmax((outsideVectors * centerWordVec).sum(dim=1), dim=0)[o_word]
    loss = -torch.log(loss)
    loss.backward()
    c_grad = centerWordVec.grad
    o_grad = outsideVectors.grad
    return loss, c_grad, o_grad


def getNegativeSamples(outsideWordIdx, words, K=5):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = words[np.random.randint(0, len(words) - 1)]
        while newidx == outsideWordIdx:
            newidx = words[np.random.randint(0, len(words) - 1)]
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(c_vec, o_vecs, o_word, words):
    negSampleWordIndices = getNegativeSamples(o_word, words)
    indices = [o_word] + negSampleWordIndices

    o_vecs = torch.from_numpy(o_vecs).requires_grad_(True)
    c_vec = torch.from_numpy(c_vec).requires_grad_(True)
    loss = -torch.log(torch.sigmoid((o_vecs * c_vec).sum(dim=1))[o_word])
    loss -= torch.log(torch.sigmoid((-o_vecs[negSampleWordIndices] * c_vec).sum(dim=1))).sum()
    loss.backward()
    gradCenterVec = c_vec.grad
    gradOutsideVecs = o_vecs.grad

    return loss, gradCenterVec, gradOutsideVecs


def fn(c_vecs, o_vecs, sentences, words, bs=50, ws=5):
    loss = 0
    c_grads = torch.zeros(c_vecs.shape)
    o_grads = torch.zeros(o_vecs.shape)
    for i in range(bs):

        loss_c_o = 0
        c_grad = torch.zeros(c_vecs.shape)
        o_grad = torch.zeros(o_vecs.shape)

        c_word, o_words = get_rnd_context(sentences, np.random.randint(1, ws))

        for o_word in o_words:
            l, gc, go = negSamplingLossAndGradient(c_vec=c_vecs[c_word], o_vecs=o_vecs, o_word=o_word, words=words)
            loss_c_o += l
            c_grad += gc
            o_grad += go

        loss += loss_c_o / bs  # TODO could be divided after all the loops
        c_grads += c_grad / bs
        o_grads += o_grad / bs

    return loss, c_grads, o_grads


def sgd(c_vecs, o_vecs, sentences, words, lr=0.1, niter=20001):
    for i in trange(niter):
        loss, c_grads, o_grads = fn(c_vecs, o_vecs, sentences, words)
        c_vecs -= (lr * c_grads.numpy())
        o_vecs -= (lr * o_grads.numpy())
        if i % 100 == 0:
            print(loss)
        if i % 1000 == 0:
            lr *= 0.5


if __name__ == '__main__':
    n_sents = 100000
    n_words = 1000
    dim = 10
    max_sent_len = 20
    sentences = np.random.randint(0, n_words, (n_sents, max_sent_len))
    words = list(itertools.chain.from_iterable(sentences))
    c_vecs = (np.random.rand(n_words, dim) - 0.5) / dim
    o_vecs = np.zeros((n_words, dim))
    sgd(c_vecs, o_vecs, sentences, words)
