import numpy as np

from sklearn import datasets, model_selection


def polynomial_fit(m=100, p=2):
    X = np.linspace(-3, 3, m).reshape(m, 1)
    coeffs = np.random.randint(-5, 5, (p+1, 1))
    poly_features = np.power(X, np.arange(0, p+1))
    y = np.dot(poly_features, coeffs)
    noise = np.random.randn(m, 1) * (y.max() - y.min()) / 20
    y = y + noise
    return X, y, coeffs[:-1], coeffs[-1]


def multivariate_fit(m=100, n_features=2):
    bias = np.random.randint(0, 100, (1, 1))
    X, y, weights = datasets.make_regression(n_samples=m, n_features=n_features,
                                             n_informative=n_features, noise=0.1, 
                                             bias=bias, coef=True, random_state=2)
    return X, y.reshape(-1, 1), weights, bias


def binary_classification(m=500):
    X, y = datasets.make_classification(n_samples=m, n_features=2,
                                        n_redundant=0, n_informative=2)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.8)
    return X_train, X_test, y_train.reshape(-1, 1), y_test.reshape(-1, 1)


def multinomial_classification(m=500, n_blobs=3):
    X, y = datasets.make_blobs(n_samples=m, n_features=2, centers=n_blobs)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.8)
    return X_train, X_test, y_train.reshape(-1, 1), y_test.reshape(-1, 1)
