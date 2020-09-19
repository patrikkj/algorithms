from datastructures.queue import Queue


# Graph traversal algorithms (Matrix representation)
def bfs_mat(W, start_id, end_id=None, **kwargs):
    # Initialize nodes using lists
    n = len(W)
    C = ["white"] * n       # Color list
    D = [float('inf')] * n  # Distance list
    PI = [None] * n         # Predecessor list

    # Initialize starting node
    C[start_id] = "gray"
    D[start_id] = 0
    PI[start_id] = None

    # While there are nodes to process
    q = Queue([start_id])
    while q:
        u_id = q.dequeue()

        # Check if node is the one we're looking for, if specified
        if u_id == end_id:
            return C, D, PI, True

        # Update each neighbour of current node
        for v_id, w in enumerate(W[u_id]):
            # Do not process nonexisting edges nor self-loop
            if not w or v_id == u_id:
                break

            # Upon encontering a new node, update node
            if C[v_id] == "white":
                C[v_id] = "gray"
                D[v_id] = D[u_id] + 1
                PI[v_id] = u_id

                # Enqueue neighbor
                q.enqueue(v_id)

        # Mark node as processed
        C[u_id] = "black"
    return C, D, PI, False

def create_path_mat(node_id, predecessors, **kwargs):
    pi_id = predecessors[node_id]
    # Base case
    if pi_id is None or pi_id is node_id:
        return [node_id]
    return create_path_mat(pi_id) + [node_id]

def find_shortest_path_mat(W, start_id, end_id, **kwargs):
    # If end node is specified, 4th return argument is set to 'True' if a path from 'start' to 'end' was found.
    _, _, predecessors, has_path = bfs_mat(W, start_id, end_id, **kwargs)

    # If such path exists, generate it from list of predecessors.
    if has_path:
        return create_path_mat(end, predecessors, **kwargs)

