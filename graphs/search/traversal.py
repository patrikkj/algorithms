from datastructures.queue import Queue
from datastructures.stack import Stack

from ..utils import Node, encapsulate


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


@encapsulate('color', 'd', 'pi', 'neighbours', namespace=g)
def dfs(graph, start, end=None, **kwargs):
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
    s = Stack([start])
    while s:
        u = s.pop()

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
                s.push(v)

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
