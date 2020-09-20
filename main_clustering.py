import numpy as np

import clustering.dbscan as dbscan
from classification.classification import knn_classifier
from clustering.clustering import k_means, labels_to_clusters
from clustering.hac import *

np.set_printoptions(linewidth=200)

# DBSCAN
def run_dbscan():
    X = np.array([
        [4, 8],
        [4, 9],
        [4, 10],
        [4, 13],
        [4, 14],
        [5, 3],
        [5, 7],
        [5, 14],
        [6, 15],
        [6, 16],
        [6, 19],
        [7, 11],
        [7, 16],
        [7, 17],
        [7, 18],
        [7, 19]
    ])
    min_pts = 4
    eps = 3

    core, border, noise = dbscan.DBSCAN(X, eps, min_pts, norm=1)
    #print(core, border, noise)
    dbscan.plot_clusters(X, core, border, noise)

# HAC
def run_hac():
    points = (
        Point("P1", 2, 3),
        Point("P2", 4, 5),
        Point("P3", 6, 4),
        Point("P4", 6, 5),
        Point("P5", 7, 5),
        Point("P6", 7, 12),
        Point("P7", 8, 2),
        Point("P8", 8, 10)
    )
    
    hac(points, heuristic_func='min', distance_func='manhattan')









# K-means
def run_k_means():
    X = np.array([
        [4, 8], 
        [4, 10],    
        [4, 13],    
        [5, 3], 
        [5, 7], 
        [7, 11],    
    ])
    init_centroids = np.array([
        [4, 4],
        [5, 8],
        [5, 11],
    ])

    centroids, labels = k_means(X, centroids=init_centroids, k=3, norm=1)
    #print(centroids)
    #print(labels)
    #print(labels_to_clusters(X, labels))


# KNN (Classification)
def run_knn():
    X = np.array([
        [4, 8],
        [8, 8],
        [8, 4],
        [6, 7],
        [1, 10],
        [3, 6],
        [2, 4],
        [1, 7],
        [6, 4],
        [6, 2],
        [6, 3],
        [4, 3],
        [4, 4],
    ])
    y = np.array([
        [1],
        [1],
        [1],
        [1],
        [2],
        [2],
        [2],
        [2],
        [3],
        [3],
        [3],
        [3],
        [3],
    ])
    X_pred = np.array([
        [6, 6],
        [4, 6],
        [4, 5],
        [2, 6],
    ])

    # Build model
    model = knn_classifier(X, y, default_k=3, norm=1)

    # Classify new data
    y_hat = model(X_pred)


# run_dbscan()
run_hac()
# run_k_means()
# run_knn()
