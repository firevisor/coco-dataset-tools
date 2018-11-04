def rotate(dataset_path, degrees):
    """Rotates the dataset clockwise by the given number of degrees.

    This assumes that the annotations in the dataset are all of the "Object
    Detection" kind. Also assumed that `iscrowd` is 0, which means that polygon
    segmentation is used.

    Performs the rotation by obtaining the image size (via `image_id`) and
    rotating each point in `segmentation` and `bbox` according to it.
    """

    raise NotImplementedError()
