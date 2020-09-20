import matplotlib.pyplot as plt
import numpy as np
import common.utils as utils


def distance_matix(X, norm=2):
    return np.linalg.norm(X[:, np.newaxis, :] - X, axis=-1, ord=norm)


def DBSCAN(X, eps, min_pts, norm=2):
    core, border, noise = set(), set(), set()

    # Euclidean distances from every point to every other point
    distances = distance_matix(X, norm=norm)
    print("\nDistance matrix")
    print(distances)

    # Number of neighbours within a maximum distance of 'eps'
    where = np.where(distances <= eps, 1, 0)
    print("\nDistance matrix filtered on 'eps'")
    print(where)

    # Mapping from point index to index of nearby points
    argwhere = np.argwhere(where)
    groups = utils.groupby(argwhere, key=lambda tup: tup[0])
    neighbour_mapping = {k: [tup[1] for tup in v if tup[1] != k] for k, v in groups.items()}
    print("\nMapping from point to nearby points")
    for k, v in neighbour_mapping.items():
        k_str = f"P{str(k+1) + ':':3}"
        v_str = '[' + ', '.join(f'P{i+1}' for i in v) + ']'
        print(k_str, v_str)

    # Traverse points by decreasing number of neighbours,
    # this makes sure that core points are classified first
    sort_func = lambda tup: len(tup[1])
    for point, neighbours in sorted(neighbour_mapping.items(), key=sort_func, reverse=True):
        if len(neighbours) >= (min_pts - 1):
            core.add(point)
            continue
        
        # Classify as border if any of its' neighbours are core points
        for neighbour in neighbours:
            if neighbour in core:
                border.add(point)
                break
        else: # If no core point is found, classify as noise
            noise.add(point)
    
    # Print results
    print("\n\nCore points:")
    print([p+1 for p in sorted(core)])

    print("\nBorder points:")
    print([p+1 for p in sorted(border)])

    print("\nNoise points:")
    print([p+1 for p in sorted(noise)])

    return core, border, noise


def plot_clusters(X, core, border, noise):

    labels = np.zeros(X.shape[0])
    for result_set, value in ((core, 0), (border, 1), (noise, -1)):
        for i in result_set:
            labels[i] = value

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col, markersize in zip([0, 1,-1], colors, [14, 8, 8]):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=markersize)

    plt.title('DBSCAN')
    plt.show()
