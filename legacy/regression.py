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
        theta (ndarray[2, 1]):      regression coefficients (increasing degree)
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
        theta (ndarray[p+1, 1]):    regression coefficients (increasing degree)
    """
    # Create polynomial features
    # [[x0^0, x0^1, ..., x0^p], [x1^0, x1^1, ..., x1^p], ... ]
    X = np.power(X, np.arange(0, p+1))

    # Objective function
    cost_func = cost.reduce_mean_sse_reg
    grad_func = cost.reduce_mean_sse_reg_grad

    theta = np.random.rand(p+1, 1)
    theta, *_ = minimize.adam(theta, X, y, cost_func, grad_func, alpha, epochs, k, l)
    return theta

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
        theta (ndarray[n+1, 1]):    regression coefficients [bias; c0; c1; ...; cn]]
    """
    # Create bias feature
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Objective function
    cost_func = cost.reduce_mean_sse_reg
    grad_func = cost.reduce_mean_sse_reg_grad

    theta = np.random.rand(X.shape[1], 1)
    theta, *_ = minimize.adam(theta, X, y, cost_func, grad_func, alpha, epochs, k, l)
    return theta

def logistic_regression(X, y, alpha=0.01, epochs=100, k=64, l=0):
    """Numerical implementation of logistic regression.
    In order to make predictions, use:
        predictions = np.dot(X, theta)

    Args:
        X (ndarray[m, n]):          input features
        y (ndarray[m, 1]):          output labels
        alpha (float, optional):    learning rate (defaults to 0.01)
        epochs (int, optional):     number of iterations (defaults to 100)
        k (int, optional):          mini-batch size (defaults to 64)
        l (float, optional):        regularization parameter (defaults to 0)
    
    Returns:
        theta (ndarray[n+1, 1]):    regression coefficients [bias; c0; c1; ...; cn]]
    """
    # Create bias feature
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Objective function
    cost_func = cost.sigmoid_cross_entropy_reg
    grad_func = cost.sigmoid_cross_entropy_reg_grad

    theta = np.random.rand(X.shape[1], 1)
    theta, *_ = minimize.adam(theta, X, y, cost_func, grad_func, alpha, epochs, k, l)
    return theta

# Alias
binary_classifier = logistic_regression