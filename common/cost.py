import numpy as np

from .activations import sigmoid


# Regularization
def l2_reg(params, l=0.01):
    return l * np.sum(np.square(params))

def l2_reg_grad(params, l=0.01):
    return 2 * l * params


# Sum of squared errors
def reduce_mean_sse(W, b, X, y):
    return np.mean(np.square((np.dot(X, W) + b) - y)) / 2

def reduce_mean_sse_grad(W, b, X, y):
    err = (np.dot(X, W) + b) - y
    dW = np.mean(err * X, axis=0, keepdims=True).T
    db = np.mean(err, keepdims=True)
    return dW, db


# Sum of squared errors (Regularized)
def reduce_mean_sse_reg(W, b, X, y, l=0.01):
    return reduce_mean_sse(W, b, X, y) + l2_reg(W, l) / (2*X.shape[0])

def reduce_mean_sse_reg_grad(W, b, X, y, l=0.01):
    dW, db = reduce_mean_sse_grad(W, b, X, y)
    dW = dW + l2_reg_grad(W, l) / (2*X.shape[0])
    return dW, db


# Sigmoid
def sigmoid_cross_entropy(W, b, X, y):
    a = sigmoid(np.dot(X, W) + b)
    return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))

def sigmoid_cross_entropy_grad(W, b, X, y):
    err = sigmoid(np.dot(X, W) + b) - y
    dW = np.mean(err * X, axis=0, keepdims=True).T
    db = np.mean(err, keepdims=True)
    return dW, db


# Sigmoid (Regularized)
def sigmoid_cross_entropy_reg(W, b, X, y, l=0.01):
    return sigmoid_cross_entropy(W, b, X, y) + l2_reg(W, l) / (2*X.shape[0])

def sigmoid_cross_entropy_reg_grad(W, b, X, y, l=0.01):
    dW, db = sigmoid_cross_entropy_grad(W, b, X, y)
    dW = dW + l2_reg_grad(W, l) / (2*X.shape[0])
    return dW, db
