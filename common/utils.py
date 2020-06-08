import time 
from collections import defaultdict


def groupby(iterable, key=None):
    if key is None:
        key = lambda x: x
    groups = defaultdict(list)
    for element in iterable:
        lookup = key(element)
        groups[lookup].append(element)
    return groups


def timeit(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        diff = te - ts
        print(f"{method.__name__}: {diff:.8f} s")
        return result
    return timed
