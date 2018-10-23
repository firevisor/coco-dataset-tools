import json
from coco_tools.error import COCOToolsError


def split(dataset, ratio):
    """Splits the dataset into multiple parts based on the given ratio.

    Within the dataset, one image can have multiple annotations. `split` splits
    the dataset by the number of images, and splits the annotations based on the
    images that they belong to.

    For example, in a dataset of 1000 images, a ratio of `70:20:10` would split
    the dataset into three datasets containing `700`, `200` and `100`
    respectively.
    """

    # Normalize the ratio.
    ratio = __extract_ratio(ratio)

    # Load the dataset.
    data = None
    try:
        with open(dataset, "r") as dataset_file:
            data = json.load(dataset_file)
    except FileNotFoundError:
        raise COCOToolsError(f"file \"{dataset}\" not found")

    # Extract `images` and `annotations`.
    images = data.pop("images")
    annotations = data.pop("annotations")

    # Generate an index.
    index = __generate_index(annotations)
    print(index)

    # Initialize the new datas.
    new_datas = []
    for _ in ratio:
        new_datas.append(data.copy())

    return new_datas


def __generate_index(annotations):
    """Generates an index which maps the image_id to the index of the annotation
    within the list.
    """

    index = {}

    for (i, annotation) in enumerate(annotations):
        image_id = annotation["image_id"]
        id = annotation["id"]

        try:
            index[image_id]
        except KeyError:
            index[image_id] = []

        index[image_id].append(id)

    return index


def __extract_ratio(ratio):
    """Splits, verifies and normalizes the ratio.

    For example, a ratio of `70:20:30` will become `[0.58, 0.17, 0.25]`. The
    total does not need to add up to `100`.
    """

    # Split and strip.
    ratio = ratio.split(":")
    for ration in ratio:
        ration.strip()

    # Parse, and hence, verify.
    for (i, ration) in enumerate(ratio):
        try:
            ration = int(ration)
        except ValueError:
            raise COCOToolsError(f'ratio {ration} should be an integer')
        ratio[i] = ration

    # Normalize based on sum.
    total = sum(ratio)
    for (i, ration) in enumerate(ratio):
        ration /= total
        ratio[i] = ration

    return ratio
