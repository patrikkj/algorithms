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


@encapsulate('color', 'd', 'pi', 'neighbours', namespace=g)
def dfs(graph, start, end=None, **kwargs):
    pass