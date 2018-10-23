from argparse import ArgumentParser
from coco_tools import COCOToolsError, split


def main():
    parser = ArgumentParser(description="Useful operations for COCO datasets")
    subparsers = parser.add_subparsers(
        help="Possible operations", dest="command")

    split_parser = subparsers.add_parser("split", help="Splits a dataset")
    split_parser.add_argument("dataset", help="The dataset to split")
    split_parser.add_argument(
        "ratio", help="The ratio to split by (e.g. 70:20:10)")

    merge_parser = subparsers.add_parser("merge", help="Merges datasets")
    merge_parser.add_argument("datasets", nargs="+",
                              help="The datasets to merge")

    args = parser.parse_args()

    try:
        if args.command == "split":
            split(args.dataset, "output", args.ratio)
    except COCOToolsError as e:
        print(f'error: {e}')
        exit(1)


if __name__ == "__main__":
    main()
