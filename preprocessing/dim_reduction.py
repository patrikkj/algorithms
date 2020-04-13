import numpy as np


def pca(X, var_retained=0.9):
    """Implements Principal Component Analysis (PCA) using the covariance method.
    
    Args:
        X (ndarray[m, n]):                  input data
        var_retained (float, optional):     minimal variance retained by output components (defaults to 0.9)
    
    Returns:
        [type]: [description]
    """
    # Find the covariance matrix
    B = X - np.mean(X, axis=0)
    C = 1/(X.shape[0] - 1) * np.dot(B.T, B)

    # Find cov. matrix eigenvectors and eigenvalues
    w, V = np.linalg.eig(C)

    # Sort eigenvectors
    indices = np.argsort(w)[::-1]
    w = w[indices]
    V = V[..., indices]
    cumvar = np.cumsum(w / w.sum())
    out_dims = np.searchsorted(cumvar, var_retained) + 1

    # Trunctate dataset using subset of eigenvectors
    W = V[..., :out_dims]
    return np.dot(X, W)

