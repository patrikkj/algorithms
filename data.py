import numpy as np

clusters = [
    np.array([[1, 1], [2, 1]]),
    np.array([[2, 4], [2, 5]]),
    np.array([[4, 1], [5, 1]]),
]

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

print(silhouette(clusters, norm=1))
print(silhouette(clusters, norm=2))