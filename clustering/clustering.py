import numpy as np

X = np.random.rand(10, 2)

def k_means(X, centroids=None, k=3):
    # Initialize randomly if no centroids are provided
    if centroids is None:
        centroids = X[np.random.choice(X.shape[0], size=k, replace=False)]

    prev_centroids = None
    while not np.array_equal(centroids, prev_centroids):
        prev_centroids = np.copy(centroids)

        # Assign cluster membership to every point
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=-1)
        indices = np.argmin(distances, axis=1).reshape(-1, 1)
        one_hot = (indices == np.arange(k))[..., np.newaxis]
        one_hotb = np.repeat(one_hot, X.shape[-1], axis=-1)
        print("onehot\n\n", one_hot.shape)
        print("onehotb\n\n", one_hotb.shape)

        # Move centroids
        clusters = X[:, None, :] * one_hot
        clusters[~one_hotb] = np.nan
        centroids = np.nanmean(clusters, axis=0)
        print(clusters)
        print(centroids)

        clusters = [X[indices==i] for i in range(centroids.shape[0])]
        centroids = np.array([np.average(points, 0) for points in clusters])
    return centroids

k_means(X)

def k_means_fully_vectorized(X, centroids=None, k=3):
    # Initialize randomly if no centroids are provided
    if centroids is None:
        centroids = X[np.random.choice(X.shape[0], size=k, replace=False)]

    prev_centroids = None
    while not np.array_equal(centroids, prev_centroids):
        prev_centroids = np.copy(centroids)

        # Assign cluster membership to every point
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=-1)
        indices = np.argmin(distances, axis=1).reshape(-1, 1)
        one_hot = (indices == np.arange(k))[..., np.newaxis]
        mask = np.repeat(one_hot, X.shape[-1], axis=-1)

        # Move centroids
        clusters = np.ma.array(X)
        clusters = X[:, None, :] * one_hot
        clusters[~one_hotb] = np.nan
        centroids = np.nanmean(clusters, axis=0)
    return centroids