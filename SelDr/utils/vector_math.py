import math


class VectorMath:
    @staticmethod
    def euclidean_distance(point1, point2):
        if hasattr(point1, 'x'):
            point1 = (point1.x, point1.y)
        if hasattr(point2, 'x'):
            point2 = (point2.x, point2.y)

        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)