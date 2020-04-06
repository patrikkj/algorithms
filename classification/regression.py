import numpy as np
import matplotlib.pyplot as plt
import common.optimize as optimize
import common.cost as cost
import examples


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

def linear_regression(X, y, alpha=0.01, epochs=100):
    return polynomial_regression(X, y, 1, alpha, epochs)


def demo_lin_reg():
    m = 50

    # Actual
    X, y, coeffs = examples.polynomial_fit(m, 1)
    theta, rX = linear_regression(X, y, alpha=0.005, epochs=10000)

    # Regression
    rX = np.hstack((np.ones(X.shape), X))
    ry = np.dot(rX, theta.T)

    # Plot data
    fig = plt.figure()
    plt.plot(X, y, "x", X, ry, "-")
    plt.savefig('demo-lin-reg.png')
    plt.close(fig)

demo_lin_reg()


def demo_poly_reg():
    m = 50
    p = 3

    # Actual
    X, y, coeffs = examples.polynomial_fit(m, p)
    theta, rX = polynomial_regression(X, y, p=p, alpha=0.005, epochs=10000)

    # Regression
    rX = np.power(X, np.arange(0, p+1))
    ry = np.dot(rX, theta.T)

    # Plot data
    fig = plt.figure()
    plt.plot(X, y, "x", X, ry, "-")
    plt.savefig('demo-poly-reg.png')
    plt.close(fig)

demo_poly_reg()