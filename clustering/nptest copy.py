import numpy as np

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
        return ", ".join(point.name for point in self.points)

    def __str__(self):
        return '{' + self.names() + '}'

    def __lt__(self, other):
        if len(self.points) > len(other.points):
            return True

        if self.names() < other.names():
            return True
        return False

    # def __cmp__(self,other):
    #     if len(self.points) > len(other.points):
    #         1
    #     elif len(self.points) < len(other.points):
    #         return -1
    #     else:
    #         return self.names() > other.names()


def hac(points, heuristic_func):
    clusters = [Cluster(point) for point in points]

    while len(clusters) > 1:
        # Print header
        header = ""
        for c in clusters:
            header += ' & \\{' + c.names() + '\\}'
        header += " \\\\"
        print(header[1:])
        print(r"\midrule")

        min_pair, min_distance = None, float("inf")
        for c1 in sorted(clusters):
            line = '\\{' + c1.names() + '\\}'

            _min_cluster, _min_distance = None, float("inf")
            for c2 in sorted(clusters):
                if c2 == c1:
                    line += " & |"
                    continue
                
                distance = heuristic_func(c1.centroid(), c2.centroid())
                line += f" & {int(distance) if distance.is_integer() else format(distance, '.2f')}"
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
        # print(f"Merging cluster {c1} and {c2}.")
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

hac(points, euclidean_distance)

# print(euclidean_distance(np.array((1, 2)), np.array((3, 4))))
