from datastructures.min_heap import MinHeap
from datastructures.queue import Queue
from datastructures.stack import Stack
from functools import reduce
from copy import deepcopy
from .utils import encapsulate, Node


#################################################################
#                        DISCLAIMER                             #
#    I know that implicitly declaring functions by mutating     #
#       global namespace is considered very bad practice.       #
# The purpose of doing it this way was to see if I was able to  #
#  use higher-order decorators to write working psuedocode. :)  #
#################################################################

g = globals()


# Graph traversal algorithms (Linked node representation)
@encapsulate('color', 'd', 'pi', 'neighbours', namespace=g)
def bfs(graph, start, end=None, **kwargs):
    # Initialize nodes
    for u in graph:
        set_color(u, "white")
        set_d(u, float('inf'))
        set_pi(u, None)

    # Initialize starting node
    set_color(start, "gray")
    set_d(start, 0)
    set_pi(start, None)

    # While there are nodes to process
    q = Queue([start])
    while q:
        u = q.dequeue()

        # Check if node is the one we're looking for, if any
        if u == end:
            return u

        # Update each neighbour of current node
        for v, w in get_neighbours(u):
            # Upon encontering a new node, update node
            if get_color(v) == "white":
                set_color(v, "gray")
                set_d(v, get_d(u) + 1)
                set_pi(v, u)

                # Enqueue neighbor
                q.enqueue(v)

        # Mark node as processed
        set_color(u, "black")

@encapsulate('pi', namespace=g)
def create_path(node, **kwargs):
    # Base case
    if get_pi(node) is None:
        return [node]
    return create_path(get_pi(node)) + [node]

@encapsulate('color', 'd', 'pi', 'neighbours', namespace=g)
def find_shortest_path(graph, start, end, **kwargs):
    # BFS returns end node, with predecessors defining
    # the shortest path to start node if such path exists.
    end_node = bfs(graph, start, end, **kwargs)

    if end_node:
        return create_path(end_node, **kwargs)


# Single source shortest path
@encapsulate('key', 'pi', namespace=g)
def initialize_single_source(graph, start):
    for node in graph:
        set_key(node, float('inf'))
        set_pi(node, None)
    set_key(start, 0)

@encapsulate('key', 'pi', namespace=g)
def relax(u, v, w):
    if get_key(v) > get_key(u) + w:
        set_key(v, get_key(u) + w)
        set_pi(v, u)

@encapsulate('key', 'pi', namespace=g)
def relax_dijkstra(u, v, w, decrease_key):
    if get_key(v) > get_key(u) + w:
        decrease_key(v, get_key(u) + w)
        set_pi(v, u)

@encapsulate('neighbours', namespace=g)
def dijkstra(graph, start, **kwargs):
    # Initialize nodes
    initialize_single_source(graph, start)

    # Initialize min-heap using 'key' as priority
    q = MinHeap(graph, key_attr=key_attr)

    # Greedy traversal node by node
    while q:
        u = q.extract_min()
        for v, _ in get_neighbours(u):
            # Relax with reference to decrease-key function, lookup using node instead of index
            relax_dijkstra(u, v, w, q.decrease_key_noderef)

@encapsulate('neighbours', namespace=g)
def dijkstra_w(graph, start, w, **kwargs):
    """
    Same as 'dijkstra' above, but using a dictionary 'w' of the form (u, v) -> w(u, v)
    """
    # Initialize nodes
    initialize_single_source(graph, start)

    # Initialize min-heap using 'risk' as priority key
    q = MinHeap(graph, key_attr=key_attr)

    # Greedy traversal node by node
    while q:
        u = q.extract_min()
        for v, _ in get_neighbours(u):
            relax_dijkstra(u, v, w[(u, v)], q.decrease_key_noderef)

@encapsulate('key', 'neighbours', namespace=g)
def bellman_ford(graph, start, **kwargs):
    # Initialize nodes
    initialize_single_source(graph, start)

    # Graph traversal
    for _ in range(len(graph) - 1):         # O(V)
        # Relax every edge in graph         # O(E)
        for u in graph:
            for v, w in get_neighbours(u):
                relax(u, v, w)

    # Final iteration for detecting negative cycles
    for u in graph:
        for v, w in get_neighbours(u):
            if get_key(v) > get_key(u) + w:
                return False
    return True

@encapsulate('neighbours', namespace=g)
def dag_shortest_path(graph, start, **kwargs):
    # Topological sorting of vertices
    graph.sort(key=get_key)

    # Initialize nodes
    initialize_single_source(graph, start)

    # Topological graph traversal
    for u in graph:
        for v, w in get_neighbours(u):
            relax(u, v, w)


# All-pairs shortest paths
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

@encapsulate('key', 'neighbours', 'pi', namespace=g)
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




def main():
    n1 = Node(ip=1)
    n2 = Node(ip=2)
    n3 = Node(ip=3)
    n4 = Node(ip=4)

    n1.neighs = [(n2, 6), (n3, 3)]
    n2.neighs = [(n4, 3)]
    n3.neighs = [(n4, 4)]
    n4.neighs = []

    n1.prob = 1.0
    n2.prob = 0.5
    n3.prob = 0.75
    n4.prob = 0.80

    graph = [n1, n2, n3, n4]

    # Initial state
    for node in graph:
        print(node)
    print()

    end = bfs(graph, n1, end=n4, neighbours_attr='neighs')
    path = create_path(end)
    for node in path:
        print(node)

    # path = find_shortest_path(graph, n1, n4, neighbours_attr='neighs')
    # print("Shortest path is:", [node.ip for node in path])

    # print(floyd_warshall(
    # [
    #     [0, 7, 2],
    #     [float('inf'), 0, float('inf')],
    #     [float('inf'), 4, 0]
    # ]))


if __name__ == '__main__':
    main()
