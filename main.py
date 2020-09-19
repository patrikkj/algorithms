# import classification.demos
# import clustering.clustering
# import graphs.search

# graphs.search.main()

import graphs.mst as mst

graph = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
weights = [
    ('a', 'b', 4),
    ('a', 'h', 8),
    ('b', 'c', 8),
    ('b', 'h', 11),
    ('c', 'd', 7),
    ('c', 'f', 4),
    ('c', 'i', 2),
    ('d', 'e', 9),
    ('d', 'f', 14),
    ('e', 'f', 10),
    ('f', 'g', 2),
    ('g', 'h', 1),
    ('g', 'i', 6),
    ('h', 'i', 4),
]

print(mst.kruskal(graph, weights))