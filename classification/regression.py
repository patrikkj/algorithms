import numpy as np

import common.cost as cost
import optimization.minimize as minimize


def linear_regression(X, y, alpha=0.01, epochs=100, k=64):
    """Numerical implementation of simple linear regression.

    Args:
        X (ndarray[m, 1]):          input features
        y (ndarray[m, 1]):          output labels
        alpha (float, optional):    learning rate (defaults to 0.01)
        epochs (int, optional):     number of iterations (defaults to 100)
        k (int, optional):          mini-batch size (defaults to 64)

    Returns:
        W (ndarray[1, 1]):          linear coefficient
        b (ndarray[1, 1]):          linear intercept
    """
    return polynomial_regression(X, y, 1, alpha, epochs, k)


def polynomial_regression(X, y, p=2, alpha=0.01, epochs=100, k=64, l=0):
    """Numerical implementation of polynomial regression.

    Args:
        X (ndarray[m, 1]):          input features
        y (ndarray[m, 1]):          output labels
        p (int, optional):          degree of polynomial (defaults to 2)
        alpha (float, optional):    learning rate (defaults to 0.01)
        epochs (int, optional):     number of iterations (defaults to 100)
        k (int, optional):          mini-batch size (defaults to 64)
        l (float, optional):        regularization parameter (defaults to 0)

    Returns:
        W (ndarray[p, 1]):          polynomial coefficients (increasing deg.)
        b (ndarray[1, 1]):          polynomial intercept
    """
    # Create polynomial features
    # [[x0^1, x0^2, ..., x0^p], [x1^1, x1^2, ..., x1^p], ... ]
    X = np.power(X, np.arange(1, p+1))

    # Objective function
    cost_func = cost.reduce_mean_sse_reg
    grad_func = cost.reduce_mean_sse_reg_grad

    W, b = np.zeros((p, 1)), np.zeros((1, 1))
    W, b, *_ = minimize.adam(W, b, X, y, cost_func,
                             grad_func, alpha, epochs, k, l)
    return W, b


def multivariate_regression(X, y, alpha=0.01, epochs=100, k=64, l=0):
    """Numerical implementation of multivariate linear regression.

    Args:
        X (ndarray[m, n]):          input features
        y (ndarray[m, 1]):          output labels
        alpha (float, optional):    learning rate (defaults to 0.01)
        epochs (int, optional):     number of iterations (defaults to 100)
        k (int, optional):          mini-batch size (defaults to 64)
        l (float, optional):        regularization parameter (defaults to 0)

    Returns:
        W (ndarray[n, 1]):          regression coefficients [w1; w2; ...; wn]
        b (ndarray[1, 1]):          bias
    """
    # Objective function
    cost_func = cost.reduce_mean_sse_reg
    grad_func = cost.reduce_mean_sse_reg_grad

    W, b = np.zeros((X.shape[1], 1)),  np.zeros((1, 1))
    W, b, *_ = minimize.adam(W, b, X, y, cost_func,
                             grad_func, alpha, epochs, k, l)
    return W, b


def logistic_regression(X, y, alpha=0.01, epochs=100, k=64, l=0):
    """Numerical implementation of logistic regression.
    In order to make predictions, use:
        predictions = np.dot(X, W) + b

    Args:
        X (ndarray[m, n]):          input features
        y (ndarray[m, 1]):          output labels
        alpha (float, optional):    learning rate (defaults to 0.01)
        epochs (int, optional):     number of iterations (defaults to 100)
        k (int, optional):          mini-batch size (defaults to 64)
        l (float, optional):        regularization parameter (defaults to 0)

    Returns:
        W (ndarray[n, 1]):          regression coefficients [w1; w2; ...; wn]
        b (ndarray[1, 1]):          bias
    """
    # Objective function
    cost_func = cost.sigmoid_cross_entropy_reg
    grad_func = cost.sigmoid_cross_entropy_reg_grad

    W, b = np.zeros((X.shape[1], 1)), np.zeros((1, 1))
    W, b, *_ = minimize.adam(W, b, X, y, cost_func,
                             grad_func, alpha, epochs, k, l)
    return W, b


# Alias
binary_classifier = logistic_regression
