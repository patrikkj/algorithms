import numpy as np


# Utils
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Regularization
def l2_reg(params, l=0.01):
    return l * np.sum(np.square(params))

def l2_reg_grad(params, l=0.01):
    return 2 * l * params


# Sum of squared errors
def reduce_mean_sse(params, X, y):
    return 1/2 * np.mean(np.square(np.dot(X, params) - y))

def reduce_mean_sse_reg(params, X, y, l=0.01):
    return reduce_mean_sse(params, X, y) + l2_reg(params, l) / (2*X.shape[0])

def reduce_mean_sse_grad(params, X, y):
    return np.mean((np.dot(X, params) - y) * X, axis=0, keepdims=True).T

def reduce_mean_sse_reg_grad(params, X, y, l=0.01):
    cost = reduce_mean_sse_grad(params, X, y)
    cost[1:] = cost[1:] + l2_reg_grad(params[1:], l) / (2*X.shape[0])
    return cost


# Sigmoid
def sigmoid_cross_entropy(params, X, y):
    y_hat = sigmoid(np.dot(X, params))
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def sigmoid_cross_entropy_reg(params, X, y, l=0.01):
    return sigmoid_cross_entropy(params, X, y) + l2_reg(params, l) / (2*X.shape[0])

def sigmoid_cross_entropy_grad(params, X, y):
    y_hat = sigmoid(np.dot(X, params))
    return -np.mean((y_hat - y) * X, axis=0, keepdims=True).T

def sigmoid_cross_entropy_reg_grad(params, X, y, l=0.01):
    cost = sigmoid_cross_entropy_grad(params, X, y)
    cost[1:] = cost[1:] + l2_reg_grad(params[1:], l) / (2*X.shape[0])
    return cost
