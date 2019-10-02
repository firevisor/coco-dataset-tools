import json
import pandas as pd
import numpy as np
from pathlib import Path
from coco_tools.error import COCOToolsError


def split(orig_anno_file, out_anno_files, ratio):
    """Splits the dataset into multiple parts based on the given ratio.

    Within the dataset, one image can have multiple annotations. `split` splits
    the dataset by the number of images, and splits the annotations based on the
    images that they belong to.

    For example, in a dataset of 1000 images, a ratio of `70:20:10` would split
    the dataset into three datasets containing `700`, `200` and `100`
    respectively.
    """

    # Extract and validate the inputs.
    ratios = __extract_ratio(ratio)
    # Some additional input validation.
    if len(ratios) != len(out_anno_files):
        raise COCOToolsError("ratio and names should be of same length")

    # Load the dataset from `dataset_path`.
    raw_data = None
    try:
        with open(str(orig_anno_file), "r") as dataset_file:
            raw_data = json.load(dataset_file)
    except FileNotFoundError:
        raise COCOToolsError(f"file \"{dataset_path}\" not found")

    # Extract `images` and `annotations`.
    images = pd.DataFrame(raw_data.pop("images"))
    annotations = pd.DataFrame(raw_data.pop("annotations"))
    
    # Split annotation data
    cumsum_ratios = np.cumsum(ratios)
    print(cumsum_ratios) 
    subsets = np.split(images.sample(frac=1),
                    [int(cumsum_ratios[i]*len(images)) for i in range(len(cumsum_ratios) - 1)])
    print([len(s) for s in subsets])
    all_subset_annos = [annotations[annotations['image_id'].isin(subset['id'])]
                    for subset in subsets]

    all_subset_data = []
    for subset, subset_annos in zip(subsets, all_subset_annos):
        new_data = {'images': subset.to_dict('records'),
                    'categories': raw_data['categories'],
                    'annotations': subset_annos.to_dict('records')}
        all_subset_data.append(new_data)

    
    for subset_data, anno_file in zip(all_subset_data, out_anno_files):
        with open(anno_file, 'w') as fo:
            json.dump(subset_data, fo)


def __extract_ratio(ratio):
    """Splits, verifies and normalizes the ratio.

    For example, a ratio of `70: 20: 30` will become `[0.58, 0.17, 0.25]`. The
    total does not need to add up to `100`.
    """

    # Split and strip.
    ratio = [ration.strip() for ration in ratio.split(":")]

    # Verify length of ratio.
    if len(ratio) != 3:
        raise COCOToolsError("ratio should have length 3")

    # Parse, and hence, verify.
    try:
        ratio = list(map(float, ratio))
    except ValueError:
        raise COCOToolsError(f'ratio should be a floats')

    # Normalize based on sum.
    return list(map(lambda ration: ration / sum(ratio), ratio))
