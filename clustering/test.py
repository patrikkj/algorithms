import numpy as np

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt



def euclidean_distance(p1, p2):
    return np.sqrt(np.sum(np.power(p1 - p2, 2)))

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
        names = (point.name for point in self.points)
        return ", ".join(sorted(names))

    def __str__(self):
        return '{' + self.names() + '}'

    def __lt__(self, other):
        # if int(self.names()) < int(other.names()):
        if self.names() < other.names():
            return True
        if len(self.points) > len(other.points):
            return True

        return False

    def __gt__(self, other):
        if self.names() > other.names():
            return True
        if len(self.points) < len(other.points):
            return True

        return False

    def __cmp__(self,other):
        if len(self.points) > len(other.points):
            return 1
        elif len(self.points) < len(other.points):
            return -1
        else:
            return self.names() > other.names()


def closest_points(c1, c2):
    p1, p2 = None, None
    distance = float("inf")

    points_1 = list(c1.points)
    points_2 = list(c2.points)

    for _p1 in points_1:
        for _p2 in points_2:
            _d = euclidean_distance(_p1.array, _p2.array)
            if _d < distance:
                p1 = _p1
                p2 = _p2
                distance = _d
    return p1, p2, distance


def furthest_points(c1, c2):
    p1, p2 = None, None
    distance = float("-inf")

    points_1 = list(c1.points)
    points_2 = list(c2.points)

    for _p1 in points_1:
        for _p2 in points_2:
            _d = euclidean_distance(_p1.array, _p2.array)
            if _d > distance:
                p1 = _p1
                p2 = _p2
                distance = _d
    return p1, p2, distance


def hac(points, heuristic_func):
    clusters = [Cluster(point) for point in points]

    while len(clusters) > 1:
        sorted_clusters = tuple(sorted(clusters))
        # Print header
        header = ""
        for c in sorted_clusters:
            # header += ' & \\{' + c.names() + '\\}'
            header += ' & ' + r"P\textsubscript{" + c.names() + "}"
        header += " \\\\"
        print(header[1:])
        print(r"\midrule")

        min_pair, min_distance = None, float("inf")
        for c1 in sorted_clusters:
            line = r"P\textsubscript{" + c1.names() + "}"

            _min_cluster, _min_distance = None, float("inf")
            for c2 in sorted_clusters:
                if c2 == c1:
                    line += " & |"
                    continue
                
                p1, p2, distance = heuristic_func(c1, c2)
                _parsed_distance = str(int(distance)) if distance.is_integer() else format(distance, '.2f')
                if distance <= 2:
                    _parsed_distance = r"\textcolor{red}{\textbf{" + _parsed_distance + r"}}"
                line += f" & {_parsed_distance}"
                # line += f" & {int(distance) if distance.is_integer() else format(distance, '.2f')}"
                # print(f"   {str(c1):10}, {str(c2):10}: {round(distance, 2)}")
                if distance < _min_distance:
                    _min_cluster = c2
                    _min_distance = distance           

            if _min_distance < min_distance:
                min_pair = (c1, _min_cluster)
                min_distance = _min_distance   
            line += " \\\\"
            print(line)
        
        # Merge closest clusters
        c1, c2 = min_pair
        print(f"Merging cluster {c1} and {c2}.")
        print()
        clusters.remove(c2)
        c1.merge(c2)

points = (
    Point("A", 4, 3),
    Point("B", 5, 8),
    Point("C", 5, 7),
    Point("D", 9, 3),
    Point("E", 11, 6),
    Point("F", 13, 8)
)

point_dbscan = (
    Point("1", 1,1), 
    Point("2", 14,8), 
    Point("3", 6,12), 
    Point("4", 3,1), 
    Point("5", 5,11),
    Point("6", 13,6), 
    Point("7", 4,12), 
    Point("8", 12,8), 
    Point("9", 1,3), 
    Point("10", 8,1), 
    Point("11", 5,9), 
    Point("12", 10,12),
    Point("13", 14,5), 
    Point("14", 2,4), 
    Point("15", 8,6), 
    Point("16", 4,3), 
    Point("17", 12,5), 
    Point("18", 14,14),
)
points_arr = np.array([p.array for p in points])
hac(point_dbscan, furthest_points)



# ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
#                    400., 754., 564., 138., 219., 869., 669.])

# hierarchy.set_link_color_palette(['g'])
Z = hierarchy.linkage(points_arr, 'complete')
plt.figure()
dn = hierarchy.dendrogram(Z, labels="ABCDEF", link_color_func=lambda k: '0')

# hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
# fig, axes = plt.subplots(1, 2, figsize=(8, 3))
# dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y',
#                            orientation='top')
# hierarchy.set_link_color_palette(None)  # reset to default after use
plt.show()


# print(euclidean_distance(np.array((1, 2)), np.array((3, 4))))
