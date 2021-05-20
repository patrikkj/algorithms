import numpy as np

clusters = [
    np.array([[1, 1], [2, 1]]),
    np.array([[2, 4], [2, 5]]),
    np.array([[4, 1], [5, 1]]),
]

test_data = np.array([
    [66.24345364, 57.31053969],
    [43.88243586, 39.69929645],
    [44.71828248, 48.38791398],
    [39.27031378, 48.07972823],
    [58.65407629, 55.66884721],
    [26.98461303, 44.50054366],
    [67.44811764, 49.13785896],
    [42.38793099, 45.61070791],
    [53.19039096, 50.21106873],
    [47.50629625, 52.91407607],
    [2.29566576, 20.15837474],
    [18.01306597, 22.22272531],
    [16.31113504, 20.1897911 ],
    [13.51746037, 19.08356051],
    [16.30599164, 20.30127708],
    [5.21390499, 24.91134781],
    [9.13976842, 17.17882756],
    [3.44961396, 26.64090988],
    [8.12478344, 36.61861524],
    [13.71248827, 30.19430912],
    [74.04082224, 23.0017032 ],
    [70.56185518, 16.47750154],
    [71.26420853, 8.57481802],
    [83.46227301, 16.50657278],
    [75.25403877, 17.91105767],
    [71.81502177, 25.86623191],
    [75.95457742, 28.38983414],
    [85.50127568, 29.31102081],
    [75.60079476, 22.85587325],
    [78.08601555, 28.85141164]
])
test_centroids = np.array([
    [25, 50],
    [50, 50],
    [75, 50]
])



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



def assign_to_nearest(X, centroids):
    """Assign cluster membership to every point."""
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=-1)
    labels = np.argmin(distances, axis=1)
    return labels

def labels_to_clusters(X, labels):
    """Utility function for converting labels to clusters.
    
    Args:
        X (ndarray[m, n]):              input dataset
        labels (ndarray[m]):            labels of each point
    
    Returns:
        clusters (list, ndarray[i, n]): list of clusters (i denotes the size of each cluster)
    """
    return [X[labels==i] for i in range(labels.max() + 1)]





def average_distances(cluster, other, norm=2):
    return np.mean(np.linalg.norm(cluster[:, None] - other, axis=2, ord=norm), axis=1)

def silhouette(clusters, norm=2):
    cluster_scores = []
    for i, cluster in enumerate(clusters):
        neighbours = clusters[:i] + clusters[i+1:]
        a = average_distances(cluster, cluster, norm=norm)
        a *= cluster.shape[0] / (cluster.shape[0] - 1) # scale by (n / n-1) to factor out distance to self
        b = np.min([average_distances(cluster, neighbour, norm=norm) for neighbour in neighbours], axis=0)

        scores = (b - a) / np.maximum(a, b)
        cluster_scores.append(scores)
    return cluster_scores

#print(silhouette(clusters, norm=1))
#print(silhouette(clusters, norm=2))

# Assignment 3 - 2021
new_centroids, labels = k_means(X=test_data, centroids=test_centroids)
print(new_centroids)
#labels = assign_to_nearest(X=test_data, centroids=test_centroids)
clusters = labels_to_clusters(X=test_data, labels=labels)
print(clusters)
clusters_scores = silhouette(clusters, norm=2)
print(sum([np.sum(scores) for scores in clusters_scores]) / test_data.shape[0])
print("MEAN", np.mean([np.mean(scores) for scores in clusters_scores]))


def silhouette_score(data, centroids, norm=2):
    """
    Function implementing the k-means clustering.

    :param data
        data
    :param centroids
        centroids
    :return
        mean Silhouette Coefficient of all samples
    """
    ### START CODE HERE ###
    prev_centroids = None
    while not np.array_equal(centroids, prev_centroids):
        prev_centroids = np.copy(centroids)

        # Calcluates the euclidean distance for every point to the centroids given.
        distances = np.apply_along_axis(euclidean_distance, 1, data, centroids)
        min_indexes = np.argmin(distances, axis=1)

        centroids_list = []
        clusters = []
        for i in range(centroids.shape[0]):
            # Find points associated with given centroid
            indices = np.nonzero(min_indexes == i)
            points = data[tuple(indices)]
            clusters.append(points)

            # Find new centroid
            centroid = np.average(points, axis=0)
            centroids_list.append(centroid)
        centroids = np.array(centroids_list)
    clusters = np.array(clusters)
    
    # Find score
    cluster_scores = []
    for i, cluster in enumerate(clusters):
        neighbours = clusters[:i] + clusters[i+1:]
        a = average_distances(cluster, cluster, norm=norm)
        a *= cluster.shape[0] / (cluster.shape[0] - 1) # scale by (n / n-1) to factor out distance to self
        b = np.min([average_distances(cluster, neighbour, norm=norm) for neighbour in neighbours], axis=0)

        scores = (b - a) / np.maximum(a, b)
        cluster_scores.append(scores)

    cluster_sums = [np.sum(scores) for scores in cluster_scores]
    return sum(cluster_sums) / data.shape[0]

#print(silhouette_score(test_data, test_centroids))
