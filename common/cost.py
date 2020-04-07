import numpy as np


# Regularization
def l2_reg(params, l=0.01):
    return l * np.sum(np.square(params))

def l2_reg_grad(params, l=0.01):
    return 2 * l * params

# Non-regularized
def reduce_mean_sse(params, X, y):
    return np.sum(np.square(np.dot(X, params.T) - y)) / (2*X.shape[0])

def reduce_mean_sse_grad(params, X, y):
    return np.sum((np.dot(X, params.T) - y) * X, axis=0) / X.shape[0]

# Regularized
def reduce_mean_sse_reg(params, X, y, l=0.01):
    return reduce_mean_sse(params, X, y) + l2_reg(params, l) / X.shape[0]

def reduce_mean_sse_reg_grad(params, X, y, l=0.01):
    cost = reduce_mean_sse_grad(params, X, y)
    cost[1:] = cost[1:] + l2_reg_grad(params[1:], l) / X.shape[0]
    return cost