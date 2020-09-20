
def flatten_tree(root, successor_attr):
    """
    Preorder tree flattening.
    """
    if not (successors := getattr(root, successor_attr)):
        return [root]
    return [root] + [n for child in successors 
                       for n in flatten_tree(child, successor_attr)]