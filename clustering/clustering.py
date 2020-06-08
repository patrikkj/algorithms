import numpy as np
# from common.utils import timeit


def k_means(X, centroids=None, k=3):
    """Implementation of K-means clustering.
    
    Args:
        X (ndarray[m, n]):              input dataset
        centroids ([type], optional):   initial centroids (defaults to random)
        k (int, optional):              number of clusters
    
    Returns:
        centroids (ndarray[k, n]):      cluster centers
        labels (ndarray[m]):            labels of each point
    """
    # Initialize randomly if no centroids are provided
    if centroids is None:
        centroids = X[np.random.choice(X.shape[0], size=k, replace=False)]
    k = centroids.shape[0]

    prev_centroids = None
    while not np.array_equal(centroids, prev_centroids):
        prev_centroids = np.copy(centroids)

        # Assign cluster membership to every point
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=-1)
        labels = np.argmin(distances, axis=1)

        # Move centroids
        centroids = np.array([np.mean(X[labels==i], axis=0) for i in range(k)])
    return centroids, labels


def k_means_vec(X, centroids=None, k=3):
    """Fully vectorized implementation of K-means clustering.
    
    Args:
        X (ndarray[m, n]):              input dataset
        centroids ([type], optional):   initial centroids (defaults to random)
        k (int, optional):              number of clusters
    
    Returns:
        centroids (ndarray[k, n]):      cluster centers
        labels (ndarray[m]):            labels of each point
    """
    # Initialize randomly if no centroids are provided
    if centroids is None:
        centroids = X[np.random.choice(X.shape[0], size=k, replace=False)]
    k = centroids.shape[0]

    prev_centroids = None
    while not np.array_equal(centroids, prev_centroids):
        prev_centroids = np.copy(centroids)

        # Assign cluster membership to every point
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=-1)
        labels = np.argmin(distances, axis=1)

        # Create masks for vectorization
        mask_2d = (labels == np.arange(k)[:, np.newaxis]).astype(float)
        mask_2d[mask_2d == 0] = np.nan
        mask_3d = np.broadcast_to(np.atleast_3d(mask_2d), (k, *X.shape))

        # Move centroids
        clusters = X * mask_3d
        centroids = np.nanmean(clusters, axis=1)
    return centroids, labels


def labels_to_clusters(X, labels):
    """Utility function for converting labels to clusters.
    
    Args:
        X (ndarray[m, n]):              input dataset
        labels (ndarray[m]):            labels of each point
    
    Returns:
        clusters (list, ndarray[i, n]): list of clusters (i denotes the size of each cluster)
    """
    return [X[labels==i] for i in range(labels.max() + 1)]


def mst(X):
    n = X.shape[0]

    # Proximity matrix D[i,j] -> sim(i, j)
    D = np.linalg.norm(X[:, np.newaxis, :] - X, axis=-1)

    # State containers
    link = np.full((n,), -1)
    dist = np.full((n,), np.inf)

    for i in range(n - 1):



def hac(X, linkage='single'):
    D = np.linalg.norm(X[:, np.newaxis, :] - X, axis=-1)
    print(D)

X = np.random.rand(10, 2)
hac(X)

