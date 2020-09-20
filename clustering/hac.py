import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum(np.power(p1 - p2, 2)))

def manhattan_distance(p1, p2):
    return np.sum(abs(p1 - p2))



class Point:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.array = np.array((x, y))

    def __str__(self):
        return f"{self.name}({self.x}, {self.y})"


class Cluster:
    def __init__(self, point):
        self.points = {point}

    def merge(self, other):
        self.points = self.points.union(other.points)

    def centroid(self):
        return np.mean(tuple(p.array for p in self.points), axis=0)

    def names(self):
        return ", ".join(point.name for point in self.points)

    def __str__(self):
        return '{' + self.names() + '}'

    def __lt__(self, other):
        if len(self.points) > len(other.points):
            return True

        if self.names() < other.names():
            return True
        return False


def min_linkage(c1, c2, distance_func):
    current_min = float('inf')

    for p1 in c1.points:
        for p2 in c2.points:
            d = distance_func(p1.array, p2.array)
            if d < current_min:
                current_min = d
    return current_min

    
def average_linkage(c1, c2, distance_func):
    return distance_func(c1.centroid(), c2.centroid())

    
def max_linkage(c1, c2, distance_func):
    current_max = -float('inf')

    for p1 in c1.points:
        for p2 in c2.points:
            d = distance_func(p1.array, p2.array)
            if d > current_max:
                current_max = d
    return current_max


def hac(points, heuristic_func='average', distance_func='euclidean', padding=1):
    if distance_func == 'euclidean':
        distance_func = euclidean_distance
    elif distance_func == 'manhattan':
        distance_func = manhattan_distance
    else:
        raise ValueError('Unknown distance metric.')

    if heuristic_func == 'min':
        heuristic_func = min_linkage
    elif heuristic_func == 'average':
        heuristic_func = average_linkage
    elif heuristic_func == 'max':
        heuristic_func = max_linkage
    else:
        raise ValueError('Unknown heuristic function.')

    clusters = [Cluster(point) for point in points]
    while len(clusters) > 1:
        D = []
        headers = [''] + [f"{{{c.names()}}}" for c in clusters]
        D.append(headers)

        min_pair, min_distance = None, float("inf")
        for c1 in sorted(clusters):
            distances = []
            distances.append(f"{{{c1.names()}}}")

            _min_cluster, _min_distance = None, float("inf")
            for c2 in sorted(clusters):
                if c2 == c1:
                    # line += " & |"
                    distances.append('—')
                    continue
                
                distance = float(heuristic_func(c1, c2, distance_func))

                distance_str = str(int(distance)) if distance.is_integer() else str(format(distance, '.2f'))
                distances.append(distance_str)

                if distance < _min_distance:
                    _min_cluster = c2
                    _min_distance = distance           

            if _min_distance < min_distance:
                min_pair = (c1, _min_cluster)
                min_distance = _min_distance   
            D.append(distances)
        
        # Calculate column widths and create dynaimcally sized templates
        col_sizes = [max(len(e) for e in column) + 2*padding for column in zip(*D)]
        meta_template = ''.join(["{{:^{}}}" for _ in range(len(headers))])
        template = meta_template.format(*col_sizes)

        for record in D:
            print(template.format(*record))

        # Merge closest clusters
        c1, c2 = min_pair
        print(f"\nMerging cluster {c1} and {c2}.\n")
        print()
        clusters.remove(c2)
        c1.merge(c2)



# print(euclidean_distance(np.array((1, 2)), np.array((3, 4))))
