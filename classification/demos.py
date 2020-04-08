import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

import dataset_gen

from .regression import (linear_regression, multivariate_regression,
                         polynomial_regression)
from .classification import (binary_classifier, multinomial_classifier,
                             knn_classifier)


def demo_lin_reg():
    # Generate dataset
    m = 100
    X, y, W_true, b_true = dataset_gen.polynomial_fit(m, 1)

    # Regression
    epochs = 1000
    W, b = linear_regression(X, y, alpha=0.005, k=8, epochs=epochs)
    print("Simple linear regression")
    print(f"  Number of examples: {m}")
    print(f"  Epochs: {epochs}")
    print(f"  Actual weight: {W_true.ravel()}")
    print(f"  Actual bias: {b_true.ravel()}")
    print(f"  Regression coeff: {W.ravel()}")
    print(f"  Regression bias: {b.ravel()}\n")
    reg_y = np.dot(X, W) + b

    # Plot data
    fig = plt.figure()
    plt.plot(X, y, "x", X, reg_y, "-")
    plt.savefig('demos/demo-lin-reg.png')
    plt.close(fig)


def demo_mul_reg():
    # Generate dataset
    m = 100
    X, y, W_true, b_true = dataset_gen.multivariate_fit(m)

    # Regression
    epochs = 1000
    W, b = multivariate_regression(X, y, alpha=0.005, k=8, epochs=epochs)
    print("Multivariate regression")
    print(f"  Number of examples: {m}")
    print(f"  Epochs: {epochs}")
    print(f"  Actual weights: {W_true.ravel()}")
    print(f"  Actual bias: {b_true.ravel()}")
    print(f"  Regression weights: {W.ravel()}")
    print(f"  Regression bias: {b.ravel()}\n")

    # Plot data
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], y)

    fX = X[:, 0].ravel()
    fX = np.linspace(fX.min(), fX.max(), 100)
    fY = X[:, 1].ravel()
    fY = np.linspace(fY.min(), fY.max(), 100)
    fZ = np.dot(np.vstack((fX, fY)).T, W) + b

    ax.scatter(fX, fY, fZ)
    plt.savefig('demos/demo-mul-reg.png')
    plt.close(fig)


def demo_poly_reg():
    # Generate dataset
    m, p = 100, 3
    X, y, W_true, b_true = dataset_gen.polynomial_fit(m, p)

    # Regression
    epochs = 1000
    W, b = polynomial_regression(X, y, p=p, alpha=0.005, k=8, epochs=epochs)
    print(f"Polynomial regression (p = {p})")
    print(f"  Number of examples: {m}")
    print(f"  Epochs: {epochs}")
    print(f"  Actual weights: {W_true.ravel()}")
    print(f"  Actual bias: {b_true.ravel()}")
    print(f"  Regression weights: {W.ravel()}")
    print(f"  Regression bias: {b.ravel()}\n")
    rX = np.power(X, np.arange(1, p+1))
    ry = np.dot(rX, W) + b

    # Plot data
    fig = plt.figure()
    plt.plot(X, y, "x", X, ry, "-")
    plt.savefig('demos/demo-poly-reg.png')
    plt.close(fig)


def demo_binary_classifier():
    # Generate dataset
    m = 500
    X_train, X_test, y_train, y_test = dataset_gen.binary_classification(m)

    # Regression
    epochs = 1000
    model = binary_classifier(X_train, y_train, threshold=0.5, epochs=epochs)
    y_hat = model(X_test)
    print("Binary classification")
    print(f"  Number of examples: {m}")
    print(f"  Epochs: {epochs}")
    print(f"  Accuracy: {np.mean(y_hat == y_test)}\n")

    # Create color mesh
    h = 0.02
    x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
    y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    cmap = ListedColormap(plt.cm.tab10.colors[:2])

    # Use model to predict and plot mesh values
    fig = plt.figure()
    z = model(np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))).reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=cmap, alpha=.8)

    # Plot data
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(),
                cmap=cmap, s=25, alpha=0.8, edgecolors='k')
    plt.savefig('demos/demo-bin-classifier.png')
    plt.close(fig)


def demo_multi_classifier():
    # Generate dataset
    m = 500
    n_blobs = 3
    X_train, X_test, y_train, y_test = dataset_gen.multinomial_classification(
        m, n_blobs)

    # Regression
    epochs = 1000
    model = multinomial_classifier(X_train, y_train, epochs=epochs)
    y_hat = model(X_test)
    print("Multinomial classification")
    print(f"  Number of examples: {m}")
    print(f"  Epochs: {epochs}")
    print(f"  Accuracy: {np.mean(y_hat == y_test)}\n")

    # Create color mesh
    h = 0.02
    x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
    y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    cmap = ListedColormap(plt.cm.tab10.colors[:n_blobs])

    # Use model to predict and plot mesh values
    fig = plt.figure()
    z = model(np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))).reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=cmap, alpha=.8)

    # Plot data
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(),
                cmap=cmap, s=25, alpha=0.8, edgecolors='k')
    plt.savefig('demos/demo-multi-classifier.png')
    plt.close(fig)


def demo_knn_classifier():
    # Generate dataset
    m = 500
    n_blobs = 5
    X_train, X_test, y_train, y_test = dataset_gen.multinomial_classification(
        m, n_blobs)

    # Regression
    epochs = 1000
    model = knn_classifier(X_train, y_train, default_k=n_blobs)
    y_hat = model(X_test)
    print("K-nearest neighbours classification")
    print(f"  Number of examples: {m}")
    print(f"  Epochs: {epochs}")
    print(f"  Accuracy: {np.mean(y_hat == y_test)}\n")

    # Create color mesh
    h = 0.1
    x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
    y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    cmap = ListedColormap(plt.cm.tab10.colors[:n_blobs])

    # Use model to predict and plot mesh values
    fig = plt.figure()
    z = model(np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))).reshape(xx.shape)
    plt.pcolormesh(xx, yy, z, cmap=cmap, alpha=.8)

    # Plot data
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(),
                cmap=cmap, s=25, alpha=0.8, edgecolors='k')
    plt.savefig('demos/demo-knn-classifier.png')
    plt.close(fig)


demo_lin_reg()
demo_mul_reg()
demo_poly_reg()
demo_binary_classifier()
demo_multi_classifier()
demo_knn_classifier()
