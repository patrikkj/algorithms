from copy import deepcopy

from ..utils import Node, encapsulate
from .single_source import bellman_ford, dijkstra_w, djikstra

#################################################################
#                        DISCLAIMER                             #
#    I know that implicitly declaring functions by mutating     #
#       global namespace is considered very bad practice.       #
# The purpose of doing it this way was to see if I was able to  #
#  use higher-order decorators to write working psuedocode. :)  #
#################################################################

g = globals()


def print_all_pairs_shortest_path(PI, i, j):
    """
    PI  Predecessor matrix
        PI[i][j] == None    <=>     i == j or w(i, j) = 'inf'
        PI[i][j] == i       <=>     i != j or w(i, j) < 'inf'
    """
    if i == j:
        print(i)
    elif PI[i][j] is None:
        print(f"No path from {i} to {j} exists!")
    else:
        print_all_pairs_shortest_path(PI, i, PI[i][j])
        print(j)


def floyd_warshall(W):
    """
    W   Weighted adjacency matrix
        W[i][j] = w(i, j)
    """
    n = len(W)

    # Table for storing temporary lowest weights
    D = deepcopy(W)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i][j] = min(D[i][j], D[i][k] + D[k][j])
    return D


@encapsulate('neighbours', namespace=g)
def transitive_closure(G, **kwargs):
    """
    The reachability matrix of G, denoted by the transitive closure.
        G   List representation of graph
    """
    n = len(G)

    # Initialize reachability matrix
    T = [[False] * n for _ in range(n)]
    for i, u in enumerate(G):
        for v, _ in get_neighbours(u):
            T[i][G.index(v)] = True

    # Update reachability matrix
    # (Here we can use Floyd Warshall with f = 'a or b', g = 'a and b')
    for k in range(n):
        for i in range(n):
            for j in range(n):
                T[i][j] = T[i][j] or (T[i][k] and T[k][j])
    return T


@encapsulate('key', 'neighbours', namespace=g)
def johnson(G, **kwargs):
    n = len(G)

    # Connect a temporary node 's' to every vertex in G
    # such that w(s, u) == 0 for every node 'u' in G
    s = Node()
    set_neighbours(s, [(u, 0) for u in G])
    G.append(s)

    # Use 'Bellman-Ford' for creating a transformation
    # such that every edge has a non-negative weight
    if not bellman_ford(G, s, **kwargs):
        print("The input graph contains a negative-weighted cycle.")
        return

    # Remove 's', we don't need it after 'Bellman-Ford'
    G.remove(s)

    # Mapping from node to transformation value
    h = {node: get_key(node) for node in G}

    # Dictionary for holding transformed edge weights
    w_ = {}
    for u in G:
        for v, w in get_neighbours(u):
            w_[(u, v)] = w + h[u] - h[v]

    # Distance matrix
    D = [[None] * n for _ in range(n)]
    for u in G:
        dijkstra_w(G, u, w_, **kwargs)
        for v in G:
            D[G.index(u)][G.index(v)] = get_key(v) + h[v] - h[u]
    return D


@encapsulate('key', 'neighbours', namespace=g)
def johnson_short(G, **kwargs):
    # Connect a temporary node 's' to every vertex in G
    # such that w(s, u) == 0 for every node 'u' in G
    s = Node()
    set_neighbours(s, [(u, 0) for u in G])

    # Use 'Bellman-Ford', assuming there are no negative cycles
    bellman_ford(G + [s], s, **kwargs)

    # Mapping from node to transformation value
    h = {node: get_key(node) for node in G}

    # Distance matrix
    n = len(G)
    D = [[None] * n for _ in range(n)]
    for u in G:
        dijkstra(G, u, **kwargs)
        for v in G:
            D[G.index(u)][G.index(v)] = get_key(v) + h[v] - h[u]
    return D
