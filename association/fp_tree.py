from .transactions import Transactions
from common import visualize, traversal, utils


class Node():
    def __init__(self, value, count=1, parent=None):
        self.value = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []

    def has_child(self, value):
        return any(c.value == value for c in self.children)

    def get_child(self, value, default=None):
        return next((c for c in self.children if c.value == value), default)
    
    def add_child(self, value, count=1):
        child = Node(value, count=count, parent=self)
        self.children.append(child)
        return child
    
    def get_path(self):
        if self.parent.value is None:
            return (self.value, )
        return self.parent.get_path() + (self.value, )
    
    def has_single_path(self):
        if (degree := len(self.children)) == 0:
            return True
        elif degree == 1:
            return self.children[0].has_single_path()
        else:
            return False
        
    def __repr__(self):
        return str(self.value)


class FPTree():
    def __init__(self, transactions, threshold, parent_tree=None, value=None, pattern_db=None):
        if isinstance(transactions, dict):
            self.transactions = Transactions(transactions)
        else:
            self.transactions = transactions
        self.threshold = threshold

        # Assert that subtrees are initialized properly
        if parent_tree is not None and value is None:
            raise ValueError("Subtrees must be assigned a 'value' and 'count'.")
        self.parent_tree = parent_tree
        self.value = value
        self.suffix = tuple() if parent_tree is None else tuple([value]) + parent_tree.suffix
        self.history = []
        self.pattern_db = pattern_db

        # Later initialization
        self.headers = {}
        self.freq_items = {}
        self.freq_patterns = {}
        self.root = Node(None, count=None, parent=None)
        self.build_tree()


    def build_tree(self):
        # Fetch frequent itemsets (k=1)
        self.freq_items, _ = self.transactions.find_frequent_items(self.threshold)

        # Filter infrequent items and sort remaining transactions
        ordering = self._create_base_ordering()
        filtered_transactions = self.transactions.filter_infrequent(self.freq_items, 
                                                                    ordering=ordering)
        # print()
        # for t, c in filtered_transactions:
        #     print(t, c)
        # Build tree from the set of all associated transactions.
        for transaction, count in filtered_transactions:
            # Start traversal from the root
            node = self.root

            for item in transaction:
                if (child := node.get_child(item, None)) is not None:
                    child.count += count
                else:
                    child = node.add_child(item, count=count)
                node = child

        # Link headers by preorder traversal
        self.create_linkage()

        # Visualize tree
        visualize.visualize_graph(self.root, file_prefix=str(self.suffix), successor_attr='children', label_func=lambda n: f'{n}:{n.count}') 


    def create_linkage(self):
        """
        Link nodes by order in a preorder traversal.
        """
        cache = {}
        for node in traversal.flatten_tree(self.root, 'children')[1:]:
            if (prev := cache.get(node.value, None)) is not None:
                prev.link = node
            else:
                self.headers[node.value] = node
            cache[node.value] = node

    def build_prefix_paths(self, value):
        prefix_paths = {}
        node = self.headers[value]
        while node is not None:
            if (path := node.get_path()[:-1]):
                prefix_paths[path] = node.count
            node = node.link
        return prefix_paths

    def build_conditional_tree(self, prefix_paths, value):
        return FPTree(prefix_paths, self.threshold, parent_tree=self, value=value)

    def mine_tree(self):
        for item in self._create_base_ordering()[::-1]:
            # print(f'Mining: {tuple(item) + self.suffix}')
            
            # Iterations at base level yield empty iterations
            if self.parent_tree is None:
                patterns = {(item, ): self.freq_items[item]}
                self.freq_patterns.update(patterns)
                record = [self.suffix, {}, {}, patterns]
                self._get_base().history.append(record)

            # Build prefix path
            prefix_paths = self.build_prefix_paths(item)
            # print(prefix_paths, '\n')

            # Build conditional FP Tree
            cond_tree = self.build_conditional_tree(prefix_paths, item)
            # print(cond_tree, '\n')
        
            if (is_single := cond_tree.root.has_single_path()):
                # Extract maximal frequent itemsets
                cond_mfi = cond_tree.freq_items
                cond_tree.generate_patterns(utils.generate_subsets(cond_tree.freq_items.keys()))
                # print(cond_tree.freq_patterns, '\n')
            else:
                # Extract maximal frequent itemsets
                temp_tree = FPTree(cond_tree.transactions, self.threshold)
                F = structure_itemsets(temp_tree.mine_tree())

                cond_mfi = maximal_frequent_itemset(F)
                cond_tree.generate_patterns((prefix, ) for prefix in cond_tree.freq_items.keys())
                # print(cond_tree.freq_patterns, '\n')
            
            # Create history record
            record = [tuple(item) + self.suffix, prefix_paths, cond_mfi, cond_tree.freq_patterns]
            base = self._get_base()
            base.history.append(record)
            base.freq_patterns.update(cond_tree.freq_patterns)


            # Recursively mine subtree
            if not is_single:
                cond_tree.mine_tree()

        return self.freq_patterns

    def generate_patterns(self, prefixes):
        for prefix in prefixes:
            pattern = tuple(prefix + self.suffix)
            count = min(self.freq_items[e] for e in prefix)
            self.freq_patterns[pattern] = self.freq_patterns.get(pattern, 0) + count

    def _create_base_ordering(self):
        candidates = self.freq_items.keys()
        base_ordering = self._get_base().freq_items.keys()
        return tuple(item for item in base_ordering if item in candidates)

    def _get_base(self):
        tree = self
        while tree.parent_tree is not None:
            tree = tree.parent_tree
        return tree
    



##################################
# Compact itemset representation #
##################################
def structure_itemsets(itemsets):
    F = {}
    for itemset, count in itemsets.items():
        F.setdefault(len(itemset), {})[tuple(sorted(itemset))] = count
    return {k: v for k, v in sorted(F.items())}

def maximal_frequent_itemset(F):
    rejected = set()
    mfis = {}
    for k in range(len(F), 0, -1):
        for bigger, count in F[k].items():
            if bigger not in rejected:
                mfis[bigger] = count
            if k == 1:
                continue
            for smaller in F[k-1]:
                if set(smaller).issubset(set(bigger)):
                    rejected.add(smaller)
    mfi_lists = []
    for mfi, count in mfis.items():
        remaining = set(mfi)
        mfi_tuples = []
        for _ in range(len(mfi)):
            item_maxfreqs = {}
            for item in remaining:
                subset = tuple(sorted(set(remaining) - set(item)))
                if subset:
                    subset_freq = F[len(subset)][subset]
                else:
                    item_maxfreqs[item] = F[1][(item, )]
                for sub_item in subset:
                    item_maxfreqs[sub_item] = max(item_maxfreqs.get(sub_item, 0), subset_freq)
            lowest, count = next((k, v) for k, v in sorted(item_maxfreqs.items(), key=lambda tup: tup[1]))
            mfi_tuples.append((lowest, count))
            remaining.remove(lowest)
        mfi_lists.append(sorted(mfi_tuples, key=lambda tup: tup[1], reverse=True))
    return mfi_lists

def main():
    T = ['ACFGH', 'BCDEG', 'BCEFH', 'ABCG', 'CDEH', 'ABCGH', 'ABCDGH', 'BCEG', 'ABCFGH', 'ABCDEGH']
    transactions = Transactions(T, break_ties='alphabetical-inv')
    fp_tree = FPTree(transactions, 5)
    #visualize.visualize_graph(fp_tree.root, file_prefix='custom', successor_attr='children', label_func=lambda n: f'{n}:{n.count}')


#main()
