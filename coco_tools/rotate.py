import math
import json
import numpy as np
import pandas as pd
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

    # Extract `images` and `annotations`.
    images = raw_data["images"]
    annotations = raw_data.pop("annotations")

    annotations = __rotate_annotations(degrees, images, annotations)

    # Set `annotations` on new data.
    new_data = raw_data
    new_data["annotations"] = annotations

    with open(__derive_path(dataset_path), "w") as output_file:
        json.dump(new_data, output_file)


def __rotate_annotations(degrees, images, annotations):
    """Rotates the annotations by the given degrees counter-clockwise.

    Performs a join with the given images to determine width and height.
    """

    # Create data frames.
    images = pd.DataFrame(images)[["id", "width", "height"]]
    annotations = pd.DataFrame(annotations)

    # Join the two data frames on `id` and `image_id`.
    annotations = annotations.join(images.set_index("id"), on="image_id")

    # Rotate each row.
    annotations = annotations.apply(
        lambda ann: __rotate_annotation(degrees, ann), axis=1)

    # Remove the `width` and `height` columns.
    annotations = annotations.drop(["width", "height"], axis=1)

    return annotations.to_dict("records")


def __rotate_annotation(degrees, annotation):
    """Rotates the annotation by the given degrees counter-clockwise.
    """

    # Calculate the `origin`, which will be used to rotate everything.
    width = annotation["width"]
    height = annotation["height"]
    origin = [width / 2, height / 2]

    # Caculate the points of the bbox.
    bbox = annotation["bbox"]
    bbox_1 = [bbox[0], bbox[1]]
    bbox_2 = [bbox[0] + bbox[2], bbox[1] + bbox[3]]

    # Rotate the points of the bbox.
    bbox_1 = __rotate_point(bbox_1, origin, degrees)
    bbox_2 = __rotate_point(bbox_2, origin, degrees)

    # Set the bbox back. Have to check the min, max points again.
    new_bbox_1 = [
        min(bbox_1[0], bbox_2[0]),
        min(bbox_1[1], bbox_2[1]),
    ]
    new_bbox_2 = [
        max(bbox_1[0], bbox_2[0]) - new_bbox_1[0],
        max(bbox_1[1], bbox_2[1]) - new_bbox_1[1],
    ]
    annotation["bbox"] = [new_bbox_1[0],
                          new_bbox_1[1], new_bbox_2[0], new_bbox_2[1]]

    # Group up `segmentation`.
    segmentation = np.array(annotation["segmentation"][0])
    segmentation = segmentation.reshape((len(segmentation) // 2, 2))

    # Rotate each point of the segmentation.
    segmentation = [__rotate_point(point, origin, degrees)
                    for point in segmentation]

    # Set the segmentation back.
    annotation["segmentation"] = segmentation

    return annotation


def __rotate_point(point, origin, degrees):
    """Rotates the point around the origin by the given number of degrees, in
    the counter-clockwise direction.
    """

    x, y = point[0] - origin[0], point[1] - origin[1]

    radians = np.deg2rad(int(degrees))
    cos = np.cos(radians)
    sin = np.sin(radians)

    rotation = np.matrix([[cos, sin], [-sin, cos]])
    result = np.dot(rotation, [x, y])

    return [round(float(result.T[0]) + origin[0]), round(float(result.T[1]) + origin[1])]


def __derive_path(dataset_path):
    """Derives the output path given `dataset_path`.
    """

    output_filename = Path(f"{str(dataset_path.stem)}_rotated.json")
    output_path = dataset_path.parent / output_filename
    return output_path
