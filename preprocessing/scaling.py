import numpy as np


def mean_normalize(X):
    return X - np.mean(X, axis=0)


def standardize(X, with_mean=True, with_std=True):
    if with_mean:
        X = mean_normalize(X)
    if with_std:
        X = X / np.std(X, axis=0)
    return X

def minmax_normailze(X):
    min_ = np.min(X, axis=0)
    max_ = np.max(X, axis=0)
    return (X - min_) / (max_ - min_)


def maxabs_normalize(X):
    min_ = np.min(X, axis=0)
    max_ = np.max(X, axis=0)
    maxabs = np.maximum(min_, max_)
    return X / maxabs
