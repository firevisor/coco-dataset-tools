import json
import pandas as pd
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
    categories = __extract_categories(raw_datas)
    images = __extract_images(raw_datas)

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
    """Merge all the `licenses` from each dataset, ignoring duplicates.
    """

    licenses = set()

    for data in datas:
        new_licenses = data.pop("licenses")
        licenses = licenses.union(set(new_licenses))

    return list(licenses)


def __extract_categories(datas):
    """Merge all the `categories` from each dataset.

    Duplicates are ignored if the categories are identical. However, if there
    are inconsistencies in the `id` for each category (which are identified by
    their `name`), an error will be raised.
    """

    categories = []

    for data in datas:
        new_categories = data.pop("categories")

        for new_category in new_categories:
            found = False

            # Verify that `new_category` is unique/consistent.
            for category in categories:
                if category["name"] != new_category["name"]:
                    continue

                found = True

                if category != new_category:
                    raise COCOToolsError("inconsistent category found")

            if not found:
                categories.append(new_category)

    return categories


def __extract_images(datas):
    """Merge all the `images` from each dataset.

    Images are identified by `id`. Warnings are logged on each duplicate.
    """

    images = pd.DataFrame()

    for data in datas:
        new_images = pd.DataFrame(data.pop("images"))
        new_images = new_images.set_index("id")

        duplicate_ids = images.index.intersection(new_images.index)

        for id in duplicate_ids.values:
            print(f"[WARN] duplicate id found: {id}")

        images = pd.concat([images, new_images], ignore_index=False)
        images = images.drop_duplicates()

    return images.to_dict("records")
