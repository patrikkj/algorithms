import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import dataset_gen

from .regression import (linear_regression, multivariate_regression,
                         polynomial_regression)


def demo_lin_reg():
    # Generate dataset
    m = 100
    X, y, coeffs = dataset_gen.polynomial_fit(m, 1)

    # Regression
    epochs = 1000
    theta, rX = linear_regression(X, y, alpha=0.005, k=8, epochs=epochs)
    print("Simple linear regression")
    print(f"  Number of examples: {m}")
    print(f"  Epochs: {epochs}")
    print(f"  Actual coeffs: {coeffs}")
    print(f"  Regression coeffs: {theta}\n")
    rX = np.hstack((np.ones(X.shape), X))
    ry = np.dot(rX, theta)

    # Plot data
    fig = plt.figure()
    plt.plot(X, y, "x", X, ry, "-")
    plt.savefig('demos/demo-lin-reg.png')
    plt.close(fig)

def demo_mul_reg():
    # Generate dataset
    m = 100
    X, y, coeffs, bias = dataset_gen.multivariate_fit(m)
    y = y[..., None]

    # Regression
    epochs = 1000
    theta, rX = multivariate_regression(X, y, alpha=0.005, k=8, epochs=epochs)
    print("Multivariate regression")
    print(f"  Number of examples: {m}")
    print(f"  Epochs: {epochs}")
    print(f"  Actual coeffs: {coeffs}")
    print(f"  Actual bias: {bias}")
    print(f"  Regression coeffs: {theta}\n")

    # Plot data
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], y)

    fC = np.ones(100)
    fX = X[:, 0].ravel()
    fX = np.linspace(fX.min(), fX.max(), 100)
    fY = X[:, 1].ravel()
    fY = np.linspace(fY.min(), fY.max(), 100)
    fZ = np.dot(np.vstack((fC, fX, fY)).T, theta)

    ax.scatter(fX, fY, fZ)
    plt.savefig('demos/demo-mul-reg.png')
    plt.close(fig)

def demo_poly_reg():
    # Generate dataset
    m, p = 100, 3
    X, y, coeffs = dataset_gen.polynomial_fit(m, p)

    # Regression
    epochs = 1000
    theta, rX = polynomial_regression(X, y, p=p, alpha=0.005, k=8, epochs=epochs)
    print(f"Polynomial regression (p = {p})")
    print(f"  Number of examples: {m}")
    print(f"  Epochs: {epochs}")
    print(f"  Actual coeffs: {coeffs}")
    print(f"  Regression coeffs: {theta}\n")
    rX = np.power(X, np.arange(0, p + 1))
    ry = np.dot(rX, theta)

    # Plot data
    fig = plt.figure()
    plt.plot(X, y, "x", X, ry, "-")
    plt.savefig('demos/demo-poly-reg.png')
    plt.close(fig)


demo_lin_reg()
demo_mul_reg()
demo_poly_reg()
