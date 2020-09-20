import numpy as np

import common.utils as utils


def _gini(values):
    """Measure of entropy among a set of values."""
    bins = np.bincount(values)
    return 1 - np.sum(np.square(bins / np.sum(bins)))

def _entropy(values):
    return True

def _gain(parent, children, impurity_func):
    """Measure of information gain between two successive levels of the decition tree."""
    weights = map(len, children)
    scores = map(impurity_func, children)
    weighted_scores = [w*s for w, s in zip(weights, scores)]
    return impurity_func(parent) - sum(weighted_scores) / len(parent)


def decision_tree(X, y, attrs=None):
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
        return decision_tree(X, y, attrs=attrs)

    # Base case: If all samples belong to the same class
    if np.unique(y).size == 1:
        return y[0]

    # Find best split attribute within current subtree
    best_gain = 0
    best_candidate = None
    for attr in attrs:
        # Format: 'group_i' -> [(X0, y0), (X1, y1), ... (Xi, yi)]
        groups = utils.groupby(zip(X, y), key=lambda tup: tup[0][attr])

        # Evaluate candidate
        attr_values, buckets = groups.keys(), groups.values()
        transpose_buckets = [list(zip(*b)) for b in buckets]
        x_buckets, y_buckets = list(zip(*transpose_buckets))
        attr_gain = _gain(y, y_buckets, impurity_func=_gini)

        # Compare candidate and update split attributre
        if attr_gain > best_gain:
            best_gain = attr_gain
            best_candidate = (attr, attr_values, x_buckets, y_buckets)

    # Generate tree
    mapping = {}
    best_attr, attr_values, x_buckets, y_buckets = best_candidate
    for value, next_x, next_y in zip(attr_values, x_buckets, y_buckets):
        next_x = np.array(next_x)
        next_attr = attrs - {best_attr}
        mapping[value] = decision_tree(next_x, next_y, attrs=next_attr)
    return (best_attr, mapping)


def traverse(instance, decision_tree):
    """Traverses the input decision tree to yield a prediction."""
    root_attr, mapping = decision_tree
    output = mapping[instance[root_attr]]
    if isinstance(output, np.int64):
        return output
    return traverse(instance, output)
