import numpy as np
from shapely.geometry import LineString


def interpolate_points(points, nb_out_points=None, segment_length=None):
    """
    Interpolate points in equal intervals over a line string defined with input points.
    Exactly one out of nr_segments and interval must not be not.
    :param points: Input points, 2D array [[x1, y1], [x2, y2], ...]. Number of points must be greater than 0.
    :param nb_out_points: desired number of interpolated points
    :param segment_length: distance between points
    :return: 2D array of interpolated points points, nr of points is nr of segments + 1
    """
    if nb_out_points is not None:
        if nb_out_points == 1:
            return points[0]
        elif nb_out_points < 1:
            raise ValueError("nb_out_points must be grater than 0")
        nr_segments = nb_out_points - 1
    else:
        nr_segments = None

    if len(points) == 0:
        raise ValueError("Point array is empty! Nothing to interpolate.")
    if len(points) < 2:
        return np.array([points[0]])
    line = LineString(points)
    length = line.length

    if bool(nr_segments) and not bool(segment_length):
        segment_length = length / nr_segments
    elif not bool(nr_segments) and bool(segment_length):
        nr_segments = int(length // segment_length)
    else:
        raise ValueError("Exactly one out of nr_segments and interval must not be None.")

    new_points = []
    for i in range(nr_segments + 1):
        pt_length = i * segment_length
        if pt_length > length + 1e-6:
            break
        pt = line.interpolate(i*segment_length)
        new_points.append(pt.coords[0])
    new_points = np.array(new_points)
    return new_points


def point_dist(p0, p1):
    """
    Calculate distance between two points in 2D.
    :param p0: first point, array like coordinates
    :param p1: second point, array like coordinates
    :return: distance, float
    """
    return ((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)**(1/2)


def test_point_dist():
    import random
    a = [random.random() for _ in range(10)]
    b = [random.random() for _ in range(10)]
    for p in zip(a, b):
        r = (0, 1)
        p = np.array(p)
        r = np.array(r)
        assert point_dist(p, r) == np.linalg.norm(p-r)


def sort_points(points, origin=(0, 0)):
    """
    Sort points in a track line sequence starting from origin.
    :param points: points to sort, array like
    :param origin: origin, starting point
    :return: sorted points, array
    """
    origin = np.array(origin)
    points = np.array(points)
    sorted = np.empty((0, 2), dtype=np.float32)

    # Find point nearest to origin
    nearest_idx = 0
    for i, pt in enumerate(points):
        if point_dist(pt, origin) < point_dist(points[nearest_idx], origin):
            nearest_idx = i
    sorted = np.append(sorted, [points[nearest_idx]], axis=0)
    points = np.delete(points, nearest_idx, axis=0)

    # Find next points
    while len(points) > 0:
        next_idx = 0
        for i, pt in enumerate(points):
            if point_dist(pt, sorted[-1]) < point_dist(points[next_idx], sorted[-1]):
                next_idx = i

        # Check continuity
        if point_dist(points[next_idx], sorted[-1]) > 30e-3:
            break

        sorted = np.append(sorted, [points[next_idx]], axis=0)
        points = np.delete(points, next_idx, axis=0)

    return np.array(sorted)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    points = np.array([(316, 369), (319, 269), (323, 191), (345, 147), (405, 116), (457, 115), (499, 112), (574, 108)])
    new_points = interpolate_points(points, segment_length=400)
    print(len(new_points))

    plt.gca().invert_yaxis()
    plt.grid()
    plt.plot(points[:, 0], points[:, 1], "-*")
    plt.plot(new_points[:, 0], new_points[:, 1], "*")

    # tck, u = interpolate.splprep(new_points.transpose(), s=0)
    # unew = np.arange(0, 1.01, 0.01)
    # out = interpolate.splev(unew, tck)
    # plt.plot(*out, "--")

    plt.show()
