import numpy as np
import common.utils as utils


def k_means(X, centroids=None, k=3, norm=2):
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
    i = 1
    while not np.array_equal(centroids, prev_centroids):
        print('\n')
        print('—'*50)
        print(f"{'Iteration ' + str(i):^50}")
        print('—'*50)
        i += 1
        prev_centroids = np.copy(centroids)

        # Assign cluster membership to every point
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=-1, ord=norm)
        print("Distance to centroids")
        print(distances)

        labels = np.argmin(distances, axis=1)
        print("\nCluster membership")
        print(np.array2string(labels.reshape(-1, 1), formatter={'all': lambda e: f'C{e+1:<2}'}))

        # Print clusters
        print("\nNew clusters")
        clusters = utils.groupby(enumerate(sorted(labels)), key=lambda tup: tup[1])
        for cluster_index, points in clusters.items():
            cluster_str = f"C{str(cluster_index+1) + ':':<3}"
            members_str = '[' + ', '.join(f'P{p[0]+1}' for p in points) + ']'
            print(cluster_str, members_str)

        # Move centroids
        centroids = np.array([np.mean(X[labels==i], axis=0) for i in range(k)])
        print("\nNew centroids")
        print(centroids)
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