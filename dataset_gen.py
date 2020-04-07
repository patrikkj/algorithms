import numpy as np

from sklearn import datasets


def polynomial_fit(m=100, p=2):
    X = np.linspace(-3, 3, m).reshape(m, 1)
    coeffs = np.random.randint(-5, 5, p+1).reshape(1, p+1)
    poly_features = np.power(X, np.arange(0, p+1))
    y = np.dot(poly_features, coeffs.T)
    noise = np.random.randn(m).reshape(m, 1) * (y.max() - y.min()) / 20
    y = y + noise
    return X, y, coeffs

def multivariate_fit(m=500, n_features=2):
    bias = np.random.randint(0, 100)
    return datasets.make_regression(n_samples=m, n_features=n_features, 
        n_informative=n_features, noise=0.1, bias=bias, coef=True, random_state=2) + (bias, )
    
