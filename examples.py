import numpy as np


def polynomial_fit(m=100, p=2):
    X = np.linspace(-3, 3, m).reshape(m, 1)
    coeffs = np.random.randint(-5, 5, p+1).reshape(1, p+1)
    poly_features = np.power(X, np.arange(0, p+1))
    noise = np.random.randn(m).reshape(m, 1) * 2
    y = np.dot(poly_features, coeffs.T) + noise
    return X, y, coeffs
"""
def multivariate_fit(m=100, dims):
    X = np.linspace(-3, 3, m).reshape(m, 1)
    coeffs = np.random.randint(-5, 5)
    poly_features = np.power(X, np.arange(0, p+1))
    noise = np.random.randn(m).reshape(m, 1) * 2
    Y = np.dot(poly_features, coeffs.T) + noise
    Z = 2*Y
    return X, Y, Z, coeffs"""