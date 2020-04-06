import numpy as np
import common.optimize as optimize
import common.cost as cost


def linear_regression(X, y, alpha=0.01, epochs=100):
    return polynomial_regression(X, y, 1, alpha, epochs)

def polynomial_regression(X, y, p=1, alpha=0.01, epochs=100):
    # Create polynomial features
    # [[x0^0, x0^1, ..., x0^p], [x1^0, x1^1, ..., x1^p], ... ]
    X = np.power(X, np.arange(0, p+1))

    # Objective function
    cost_func = cost.reduce_mean_sse
    grad_func = cost.reduce_mean_sse_grad

    theta = np.random.rand(1, p+1)
    theta, costs, grads = optimize.gradient_descent(theta, X, y, cost_func, grad_func, alpha, epochs)
    return theta, X

def multivariate_regression(X, y, alpha=0.01, epochs=100):
    # Objective function
    cost_func = cost.reduce_mean_sse
    grad_func = cost.reduce_mean_sse_grad

    theta = np.random.rand(1, X.shape[1])
    theta, costs, grads = optimize.gradient_descent(theta, X, y, cost_func, grad_func, alpha, epochs)
    return theta, X
