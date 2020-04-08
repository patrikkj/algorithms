import numpy as np

from common.activations import sigmoid

from .regression import logistic_regression


def binary_classifier(X, y, threshold=0.5, **kwargs):
    """Trains a binary classifier from the dataset provided using logistic regression.

    In order to classify new data, use:
        y_hat = model(X)

    Args:
        X (ndarray[m, n]):              ground features
        y (ndarray[m, 1]):              ground labels
        threshold (float, optional):    default classification threshold, can be 
                                        overridden if passed as argument to the model
                                        (defaults to 0.5)
        **kwargs (...):                 logistic regression arguments, see documentation

    Returns:
        model (X -> y_hat):             callable binary classifier
    """
    classes = np.unique(y)
    W, b = logistic_regression(X, y, **kwargs)
    def model(X_pred, threshold=threshold):
        indices = (sigmoid(np.dot(X_pred, W) + b) >= threshold).astype(int)
        return classes[indices].reshape(-1, 1)
    return model


def multinomial_classifier(X, y, **kwargs):
    """Trains a multi-class classifier from the dataset 
    provided using a set of logistic regression classifiers.

    In order to classify new data, use:
        y_hat = model(X)

    Args:
        X (ndarray[m, n]):              ground features
        y (ndarray[m, 1]):              ground labels
        **kwargs (...):                 logistic regression arguments, see documentation

    Returns:
        model (X -> y_hat):             callable multi-class classifier
    """
    classes = np.unique(y)

    # Build logistic regression models
    lr_params = []
    for cls in classes:
        y_model = (y == cls).astype(int).reshape(-1, 1)
        W, b = logistic_regression(X, y_model, **kwargs)
        lr_params.append((W, b))

    def model(X_pred):
        """Evaluates each logistic regression model, picking the label
        supplied by the most confident model for each input.
        """
        activations = [sigmoid(np.dot(X_pred, W) + b) for W, b in lr_params]
        indices = np.argmax(np.hstack(activations), axis=1)
        return classes[indices].reshape(-1, 1)
    return model


def knn_classifier(X, y, default_k=3):
    """Builds a K-nearest neighbour classifier from the dataset provided.

    In order to classify new data, use:
        y_hat = model(X)

    Args:
        X (ndarray[m, n]):      ground features
        y (ndarray[m, 1]):      ground labels
        k (int, optional):      number of neighbours to consider (defaults to 3)
    
    Returns:
        model (X -> y_hat):     callable knn classifier
    """
    classes = np.unique(y)

    def model(X_pred, k=default_k):
        # Euclidean distances from every point to every labeled point
        distances = np.linalg.norm(X - X_pred[:, np.newaxis, :], axis=-1)

        # Indices of k nearest neighbours (np.argpartition yields indices
        # s.t. first k elements along axis are ordered)
        indices = np.argpartition(distances, k, axis=-1)[..., :k]

        # Classes of k nearest neighbours
        k_nearest = np.take(y, indices)

        # Find class index of most frequent label among k nearest neighbours
        mode_indices = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, k_nearest)
        return classes[mode_indices].reshape(-1, 1)
    return model


def decision_tree_classifier():
    pass
