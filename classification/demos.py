import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import examples
from .regression import linear_regression, polynomial_regression, multivariate_regression


def demo_lin_reg():
    # Generate dataset
    X, y, coeffs = examples.polynomial_fit(50, 1)

    # Regression
    theta, rX = linear_regression(X, y, alpha=0.005, epochs=10000)
    print("Simple linear regression")
    print(f"  Actual coeffs: {coeffs}")
    print(f"  Regression coeffs: {theta}\n")
    rX = np.hstack((np.ones(X.shape), X))
    ry = np.dot(rX, theta.T)

    # Plot data
    fig = plt.figure()
    plt.plot(X, y, "x", X, ry, "-")
    plt.savefig('demos/demo-lin-reg.png')
    plt.close(fig)

def demo_mul_reg():
    # Generate dataset
    m = 700
    X, y, coeffs = examples.multivariate_fit(m)
    y = y[..., None]

    # Regression
    theta, rX = multivariate_regression(X, y, alpha=0.005, epochs=10000)
    print("Multivariate regression")
    print(f"  Actual coeffs: {coeffs}")
    print(f"  Regression coeffs: {theta}\n")

    # Plot data
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], y)

    fX = X[:, 0].ravel()
    fY = X[:, 1].ravel()
    #fZ = ry.ravel()
    #idx = np.argsort(fX)

    fX = np.linspace(fX.min(), fX.max(), 100)
    fY = np.linspace(fY.min(), fY.max(), 100)
    fZ = np.dot(np.vstack((fX, fY)).T, theta.T)

    ax.scatter(fX, fY, fZ)

    #ax.plot(fX[idx], fY[idx], fZ[idx], "--")
    plt.savefig('demos/demo-mul-reg.png')
    plt.close(fig)

def demo_poly_reg():
    # Generate dataset
    m, p = 50, 3
    X, y, coeffs = examples.polynomial_fit(m, p)

    # Regression
    theta, rX = polynomial_regression(X, y, p=p, alpha=0.005, epochs=10000)
    print(f"Polynomial regression (p = {p})")
    print(f"  Actual coeffs: {coeffs}")
    print(f"  Regression coeffs: {theta}\n")
    rX = np.power(X, np.arange(0, p + 1))
    ry = np.dot(rX, theta.T)

    # Plot data
    fig = plt.figure()
    plt.plot(X, y, "x", X, ry, "-")
    plt.savefig('demos/demo-poly-reg.png')
    plt.close(fig)


demo_lin_reg()
demo_mul_reg()
demo_poly_reg()
