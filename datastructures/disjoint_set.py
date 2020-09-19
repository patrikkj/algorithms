class DisjointSetNode:
    def __init__(self, parent, rank):
        self.parent = self if parent is None else parent
        self.rank = rank


class DisjointSet:
    def __init__(self, *nodes):
        """Creates a singleton disjoint set for every node provided."""
        self.mapping = {}
        for node in nodes:
            self.make_set(node) 

    def make_set(self, x):
        """Creates a singleton disjoint set."""
        self.mapping[x] = DisjointSetNode(None, 0)

    def union(self, x, y):
        """Unites the two dynamic sets 'x' and 'y'.
        The two sets are assumed to be disjoint prior to the operation.
        """
        x, y = self.mapping[x], self.mapping[y]
        self.link(self.find_set(x), self.find_set(y))
    
    def link(self, x, y):
        """Implemements the 'union by rank' heuristic.
        'x' and 'y' are assumed to be instances of DisjointSetNode.
        """
        if x.rank > y.rank:
            y.parent = x
        else:
            x.parent = y
            if x.rank == y.rank:
                y.rank += 1
    
    def find_set(self, x):
        """Returns a pointer to the representative of the set containing 'x'.
        Implements the 'path compression' heuristic.
        """
        if not isinstance(x, DisjointSetNode):
            x = self.mapping[x]
        if x != x.parent:
            x.parent = self.find_set(x.parent)
        return x.parent
