import numpy as np


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
