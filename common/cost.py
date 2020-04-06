import numpy as np


def reduce_mean_sse(params, X, y):
    return np.sum(np.square(np.dot(X, params.T) - y)) / X.shape[0]

def reduce_mean_sse_grad(params, X, y):
    return 1/X.shape[0] * np.sum((np.dot(X, params.T) - y) * X, axis=0)
