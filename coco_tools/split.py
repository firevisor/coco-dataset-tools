import json
import pandas as pd
import numpy as np
from pathlib import Path
from coco_tools.error import COCOToolsError


def split(dataset, output_path, ratio):
    """Splits the dataset into multiple parts based on the given ratio.

    Within the dataset, one image can have multiple annotations. `split` splits
    the dataset by the number of images, and splits the annotations based on the
    images that they belong to.

    For example, in a dataset of 1000 images, a ratio of `70:20:10` would split
    the dataset into three datasets containing `700`, `200` and `100`
    respectively.
    """

    # Create the output path.
    output_path = Path(output_path)

    # Normalize the ratio.
    ratio = __extract_ratio(ratio)

    # Load the dataset.
    raw_data = None
    try:
        with open(dataset, "r") as dataset_file:
            raw_data = json.load(dataset_file)
    except FileNotFoundError:
        raise COCOToolsError(f"file \"{dataset}\" not found")

    # Extract `images` and `annotations`.
    images = raw_data.pop("images")
    annotations = raw_data.pop("annotations")

    # Initialize the new datas and datasets (the filenames).
    new_datas = []
    for _ in ratio:
        new_datas.append(raw_data.copy())

    # Split the data.
    __split_data(new_datas, ratio, images, annotations)

    # Output the results to the corresponding files.
    for (i, new_data) in enumerate(new_datas):
        with open(str(output_path / str(i)), "w") as output_file:
            json.dump(new_data, output_file)


def __split_data(datas, ratio, images, annotations):
    """Sets `images` and `annotations` on the `datas` based on `ratio`.

    Take note that this method mutates `datas`. It is done this way because
    `datas` should contain the additional data as part of a COCO dataset.

    `pandas` is used here to perform the splitting/partitioning.
    """

    # Create data frames.
    images = pd.DataFrame(images)
    annotations = pd.DataFrame(annotations)

    # Create the base mask
    base_mask = np.random.rand(len(images))

    # Track the current sum of ratios. This is used when finding the range to
    # compare to.
    ratio_sum = 0

    # Iterate through each ratio and split the data.
    for (i, ration) in enumerate(ratio):
        data = datas[i]

        # Create the mask.
        mask = (base_mask >= ratio_sum) & (base_mask < ratio_sum + ration)
        ratio_sum += ration

        # Set the images on the data.
        data["images"] = images[mask].to_dict("records")

        # Set the annotations on the data.
        common = images[mask].merge(
            annotations, left_on="id", right_on="image_id", how="inner")
        data["annotations"] = annotations[annotations.image_id.isin(
            common.image_id)].to_dict("records")


def __extract_ratio(ratio):
    """Splits, verifies and normalizes the ratio.

    For example, a ratio of `70: 20: 30` will become `[0.58, 0.17, 0.25]`. The
    total does not need to add up to `100`.
    """

    # Split and strip.
    ratio = ratio.split(":")
    for ration in ratio:
        ration.strip()

    # Parse, and hence, verify.
    for (i, ration) in enumerate(ratio):
        try:
            ration = float(ration)
        except ValueError:
            raise COCOToolsError(f'ratio {ration} should be a float')
        ratio[i] = ration

    # Normalize based on sum.
    total = sum(ratio)
    for (i, ration) in enumerate(ratio):
        ration /= total
        ratio[i] = ration

    return ratio
