from coco_tools.error import COCOToolsError


def split(dataset, ratio):
    ratio = __extract_ratio(ratio)
    print(dataset, ratio)


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
