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


# Single source shortest path
@encapsulate('key', 'pi', namespace=g)
def initialize_single_source(graph, start):
    """Initializes node attributes for various single-source shortest path algorithms."""
    for node in graph:
        set_key(node, float('inf'))
        set_pi(node, None)
    set_key(start, 0)


@encapsulate('key', 'pi', namespace=g)
def relax(u, v, w):
    """Tests whether the best-known way from u to v intercepts w, updates path if true."""
    if get_key(v) > get_key(u) + w:
        set_key(v, get_key(u) + w)
        set_pi(v, u)


@encapsulate('key', 'pi', namespace=g)
def relax_dijkstra(u, v, w, decrease_key):
    """Tests whether the best-known way from u to v intercepts w, updates path if true."""
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
