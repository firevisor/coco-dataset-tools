import json
import numpy as np
from pathlib import Path
from coco_tools.error import COCOToolsError


def rotate(dataset_path, degrees):
    """Rotates the dataset counter-clockwise by the given number of degrees.

    This assumes that the annotations in the dataset are all of the "Object
    Detection" kind. Also assumed that `iscrowd` is 0, which means that polygon
    segmentation is used.

    Performs the rotation by obtaining the image size (via `image_id`) and
    rotating each point in `segmentation` and `bbox` according to it.
    """

    dataset_path = Path(dataset_path)

    raw_data = None
    try:
        with open(str(dataset_path), "r") as dataset_file:
            raw_data = json.load(dataset_file)
    except FileNotFoundError:
        raise COCOToolsError(f"file \"{dataset_path}\" not found")

    with open(__derive_path(dataset_path), "w") as output_file:
        json.dump(raw_data, output_file)

    raise NotImplementedError()


def __derive_path(dataset_path):
    """Derives the output path given `dataset_path`.
    """

    output_filename = Path(f"{str(dataset_path.stem)}_rotated.json")
    output_path = dataset_path.parent / output_filename
    return output_path


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
