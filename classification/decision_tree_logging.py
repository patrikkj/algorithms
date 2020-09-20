import numpy as np
from common import visualize

import common.utils as utils


def _gini(values):
    """Measure of entropy among a set of values."""
    bins = np.bincount(values)
    return 1 - np.sum(np.square(bins / np.sum(bins)))

def _entropy(values):
    bins = np.bincount(values)
    p = bins / np.sum(bins)
    p = p[p != 0]
    return - np.sum(p * np.log2(p))

def _error(values):
    bins = np.bincount(values)
    p = bins / np.sum(bins)
    p = p[p != 0]
    return 1 - np.max(p)

def _gain(parent, children, impurity_func, parent_attr, attr_values, history):
    """Measure of information gain between two successive levels of the decition tree."""
    N = len(parent)
    weights = [len(c) / N for c in children]
    scores = [impurity_func(c) for c in children]
    weighted_scores = [w*s for w, s in zip(weights, scores)]

    impurity_parent = impurity_func(parent)
    impurity_child = sum(weighted_scores)
    gain = impurity_parent - impurity_child
    
    # Create logging record
    for i, child_details in enumerate(zip(attr_values, weights, scores, weighted_scores)):
        child, weight, imp, weighted_imp = child_details
        if i == 0:
            record = (parent_attr, (parent_attr, child), weight, imp, weighted_imp, impurity_parent, impurity_child, gain)
        else:
            record = (None, (parent_attr, child), weight, imp, weighted_imp, None, None, None)
        history.append(record)
    return gain


def decision_tree(X, y, impurity_func='gini', attrs=None, history=None, split_attr=None, split_value=None):
    """Builds a traversable decision tree.
    
    Args:
        X (ndarray[m, n]):          ground features
        y (ndarray[m, 1]):          ground labels
        attrs (tuple, optional):    tuple of attributes to branch on (defaults to all)
    
    Returns:
        root_attr (int):            index of root attribute
        mapping (dict):             decision tree
    """
    # First run
    if attrs is None:
        attrs = set(range(X.shape[1]))

    if impurity_func == 'gini':
        impurity_func = _gini
    elif impurity_func == 'entropy':
        impurity_func = _entropy

    #return decision_tree(X, y, impurity_func=impurity_func, attrs=attrs)

    # Base case: If all samples belong to the same class
    if np.unique(y).size == 1:
        return y[0]
    elif not attrs:
        print(f"INFO: Discovered ambigous mapping for X = {X}")
        return list(y)

    # Log history for printing
    if history is None:
        history = []
        output = decision_tree(X, y, impurity_func, attrs, history, split_attr, split_value)
        return output, history

    
    # Find best split attribute within current subtree
    local_history = []
    best_gain = -float('inf')
    best_candidate = None
    for attr in attrs:
        # Format: 'group_i' -> [(X0, y0), (X1, y1), ... (Xi, yi)]
        groups = utils.groupby(zip(X, y), key=lambda tup: tup[0][attr])

        # Evaluate candidate
        attr_values, buckets = groups.keys(), groups.values()
        transpose_buckets = [list(zip(*b)) for b in buckets]
        x_buckets, y_buckets = list(zip(*transpose_buckets))
        attr_gain = _gain(y, y_buckets, impurity_func=impurity_func, parent_attr=attr, attr_values=attr_values, history=local_history)

        # Compare candidate and update split attributre
        if attr_gain > best_gain:
            best_gain = attr_gain
            best_candidate = (attr, attr_values, x_buckets, y_buckets)

    # Append to histories
    history.append((split_attr, split_value, local_history))

    # Generate tree
    mapping = {}
    best_attr, attr_values, x_buckets, y_buckets = best_candidate
    for value, next_x, next_y in zip(attr_values, x_buckets, y_buckets):
        next_x = np.array(next_x)
        next_attr = attrs - {best_attr}
        _split_attr = (best_attr, ) if split_attr is None else split_attr + (best_attr, )
        mapping[value] = decision_tree(next_x, next_y, impurity_func=impurity_func, attrs=next_attr, history=history, split_attr=_split_attr, split_value=value)
    return (best_attr, mapping) # RETURNS TUPLE -> SUBTREE


def traverse(instance, decision_tree):
    """Traverses the input decision tree to yield a prediction."""
    root_attr, mapping = decision_tree
    output = mapping[instance[root_attr]]
    if isinstance(output, np.int64):
        return output
    return traverse(instance, output)

def print_summary(summary, header, categories):
    for local_summary in summary:
        split_attr, split_value, local_history = local_summary
        split_attr_str = ' → '.join(header[i] for i in split_attr) if split_attr is not None else 'root'
        split_value_str = categories[split_attr[-1]][split_value] if split_value is not None else 'root'
        print(f'\n\nSplit attribute: {split_attr_str}    Split value: {split_value_str}')
        c1_header = 'Parent'
        c2_header = 'Child'
        c3_header = 'Weight'
        c4_header = 'Impurity'
        c5_header = 'Weighted Imp.'
        c6_header = 'Parent Imp.'
        c7_header = 'Child Imp.'
        c8_header = 'Gain'
        utils.print_table(local_history, 
            model={
                c1_header: lambda v: header[v] if v is not None else '',
                c2_header: lambda tup: categories[tup[0]][tup[1]] if tup is not None else '',
                c3_header: lambda v: str(round(v, 3)) if v is not None else '',
                c4_header: lambda v: str(round(v, 3)) if v is not None else '',
                c5_header: lambda v: str(round(v, 3)) if v is not None else '',
                c6_header: lambda v: str(round(v, 3)) if v is not None else '',
                c7_header: lambda v: str(round(v, 3)) if v is not None else '',
                c8_header: lambda v: str(round(v, 3)) if v is not None else ''
            }, 
            sep_func=lambda i, row: bool(row[0]), 
            padding=2
        )


class Node():
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children is not None else {}

    def add_child(self, key, value):
        self.children[key] = value

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return str(self)


def create_node(model, headers, categories):
    if isinstance(model, tuple):
        root_attr, mapping = model
        node = Node(headers[root_attr])

        for attr_value, new_model in mapping.items():
            value = categories[root_attr][attr_value]
            node.add_child(value, create_node(new_model, headers, categories))

    elif isinstance(model, list):
        node = Node(str(model))
    else: #regular value
        node = Node(str(model))
    return node

def export_tree(model, headers, categories, show=True):
    root = create_node(model, headers, categories)
    print()
    edge_label_func = lambda node, successor: next(k for k, v in node.children.items() if v == successor)
    visualize.visualize_graph(
        root, 
        directory='exported_trees',
        file_prefix='tree', 
        successor_attr='children', 
        label_func=lambda n: n.value, 
        edge_label_func=edge_label_func,
        root_to_none=False,
        show=show
    ) 
