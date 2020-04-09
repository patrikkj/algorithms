from collections import defaultdict


def groupby(iterable, key=None):
    if key is None:
        key = lambda x: x
    groups = defaultdict(list)
    for element in iterable:
        lookup = key(element)
        groups[lookup].append(element)
    return groups
