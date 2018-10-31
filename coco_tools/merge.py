import json
from pathlib import Path
from coco_tools.error import COCOToolsError


def merge(dataset_paths, name):
    """Merges the datasets at the given paths.

    Will merge `images`, `annotations`, `licenses` and `categories`. For `images`
    and `annotations`, any duplicates in the `id` will be logged as warnings.
    """

    # Extract and validate inputs
    dataset_paths = list(map(Path, dataset_paths))
    name = name.strip()

    # Perform basic validation.
    if len(dataset_paths) < 2:
        raise COCOToolsError("no point merging less than two datasets")

    # Load the datasets from `dataset_paths`.
    raw_datas = []
    for dataset_path in dataset_paths:
        try:
            with open(str(dataset_path), "r") as dataset_file:
                raw_datas.append(json.load(dataset_file))
        except FileNotFoundError:
            raise COCOToolsError(f"file \"{dataset_path}\" not found")

    # Extract the various properties.
    info = __extract_info(raw_datas)
    licenses = __extract_licenses(raw_datas)

    raise NotImplementedError()


def __extract_info(datas):
    """Extract the `info` from the first dataset and removes `info` from all
    the datasets.
    """

    info = datas[0]["info"]

    for data in datas:
        data.pop("info")

    return info


def __extract_licenses(datas):
    """Combine all the `licenses` from each dataset, ignoring duplicates.
    """

    licenses = set()

    for data in datas:
        new_licenses = data.pop("licenses")
        licenses = licenses.union(set(new_licenses))

    return list(licenses)
