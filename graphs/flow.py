from copy import deepcopy

from .search.traversal_matrix import find_shortest_path_mat


# Maximum flow networks (Using linked nodes representation)
def ford_fulkerson(C, s, t, augmenting_path, **kwargs):
    """
    Finds the maximum flow through the network, restricted by C.
        C                   Capacity matrix
        s                   Source node (ID)
        t                   Sink node (ID)
        augmenting_path     Method yielding an augmenting path
    ->  Flow matrix F, representing maximum flow.
    """
    # Assign every edge a flow of 0
    F = [[0]*n for _ in range(n)]

    # Initialize residual capacity matrix
    C_f = deepcopy(C)

    while path := augmenting_path(C_f, s, t, **kwargs):
        # Find path bottleneck
        edges = zip(path, path[1:])
        residual_capacities = (C_f[u][v] for u, v in edges)
        min_capacity = min(residual_capacities)

        # Update flow for every edge in augmenting path
        for u, v in edges:
            F[u][v] += min_capacity
            F[v][u] -= min_capacity
            C_f[u][v] -= min_capacity
            C_f[v][u] += min_capacity

    return F, sum(F[s])


def edmonds_karp(C, s, t, **kwargs):
    return ford_fulkerson(C, s, t, find_shortest_path_mat, **kwargs)
