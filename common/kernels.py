import numpy as np


def linear(x1, x2):
    return np.dot(x1.T, x2)

def polynomial(x1, x2, d=3, gamma=0.5, c=0):
    return np.power(gamma * np.dot(x1.T, x2) + c, d)

def rbf(x1, x2, gamma=0.5):
    """Radial basis function (RBF) kernel, also called 'Gaussian' kernel.
    
    Args:
        x1 (...): [description]
        x2 ([type]): [description]
        gamma (float): [description]
    
    Returns:
        [type]: [description]
    """
    return np.exp(-gamma * np.sum(np.square(x2 - x1)))

def sigmoid(x1, x2, gamma=0.5, c=0):
    return np.tanh(gamma * np.dot(x1.T, x2) + c)
