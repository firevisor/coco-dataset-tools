import numpy as np


def rotate(dataset_path, degrees):
    """Rotates the dataset clockwise by the given number of degrees.

    This assumes that the annotations in the dataset are all of the "Object
    Detection" kind. Also assumed that `iscrowd` is 0, which means that polygon
    segmentation is used.

    Performs the rotation by obtaining the image size (via `image_id`) and
    rotating each point in `segmentation` and `bbox` according to it.
    """

    raise NotImplementedError()


def __rotate_point(point, origin, degrees):
    """Rotates the point around the origin by the given number of degrees, in
    the counter-clockwise direction.
    """

    x, y = point[0] - origin[0], point[1] - origin[1]

    radians = np.deg2rad(degrees)
    cos = np.cos(radians)
    sin = np.sin(radians)

    rotation = np.matrix([[cos, sin], [-sin, cos]])
    result = np.dot(rotation, [x, y])

    return [round(float(result.T[0])) + origin[0], round(float(result.T[1])) + origin[1]]
