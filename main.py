from argparse import ArgumentParser
from coco_tools import COCOToolsError, split, merge


def main():
    parser = ArgumentParser(description="Useful operations for COCO datasets")
    subparsers = parser.add_subparsers(
        help="Possible operations", dest="command")

    split_parser = subparsers.add_parser("split", help="Splits a dataset")
    split_parser.add_argument(
        "-i", "--dataset", help="The dataset to split", default="data.json")
    split_parser.add_argument(
        "-r", "--ratio", help="The ratio to split by (e.g. 70:20:10)", default="70:20:10")
    split_parser.add_argument(
        "-n", "--names", help="The names for each split (e.g. train:validation:test)", default="train:validation:test")

    merge_parser = subparsers.add_parser("merge", help="Merges datasets")
    merge_parser.add_argument(
        "-i", "--datasets", nargs="+", help="The datasets to merge", default=[])
    merge_parser.add_argument(
        "-n", "--name", help="The name for merged dataset", default="merge")

    args = parser.parse_args()

    try:
        if args.command == "split":
            split(args.dataset, args.ratio, args.names)
        elif args.command == "merge":
            merge(args.datasets, args.name)
    except COCOToolsError as e:
        print(f'error: {e}')
        exit(1)


if __name__ == "__main__":
    main()
