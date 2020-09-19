from datastructures.disjoint_set import DisjointSet
from datastructures.min_heap import MinHeap

from .utils import encapsulate

#################################################################
#                        DISCLAIMER                             #
#    I know that implicitly declaring functions by mutating     #
#       global namespace is considered very bad practice.       #
# The purpose of doing it this way was to see if I was able to  #
#  use higher-order decorators to write working psuedocode. :)  #
#################################################################

g = globals()


def kruskal(graph, weights):
    """Grows a minimal spanning tree using Kruskal's algorithm.

    For custom node classes, **kwargs can be used to redirect encapsulated attributes, e.g.:
        (args, ..., 'neighbours_attr'='neighs') => get_neighbours(node) = node.neighs 

    Args:
        graph (list, Node):         list of all nodes in graph
        weights (list, tuple):      list of tuples of the form (u, v, weight)
    """
    A = set()

    # Create a singleton disjoint set for every node
    ds = DisjointSet(*graph)

    for u, v, _ in sorted(weights, key=lambda tup: tup[2]):
        if ds.find_set(u) != ds.find_set(v):
            A.add((u, v))
            ds.union(u, v)
    return A


@encapsulate('key', 'pi', namespace=g)
def prim(graph, start, **kwargs):
    """Grows a minimal spanning tree using Prim's algorithm.

    For custom node classes, **kwargs can be used to redirect encapsulated attributes, e.g.:
        (args, ..., 'neighbours_attr'='neighs') => get_neighbours(node) = node.neighs 

    Args:
        graph (list, Node):         list of all nodes in graph
        start (Node):               starting node
        **kwargs (...):             used to redirect encapsulation attributes
    """
    for u in graph:
        set_key(u, float('inf'))
        set_pi(u, None)
    set_key(start, 0)

    # Initialize min-heap using 'key' as priority
    key_attr = kwargs.get('key_attr', 'key')
    q = MinHeap(graph, key_attr=key_attr)

    while q:
        u = q.extract_min()
        for v, w in get_neighbours(u):
            if v in q and w < get_key(v):
                set_key(v, w)
                set_pi(v, u)
