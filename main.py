from argparse import ArgumentParser
import coco_tools


def main():
    parser = ArgumentParser(description="Useful operations for COCO datasets")
    subparsers = parser.add_subparsers(
        help="Possible operations", dest="command")

    split_parser = subparsers.add_parser("split", help="Splits a dataset")
    split_parser.add_argument("dataset", help="The dataset to split")

    merge_parser = subparsers.add_parser("merge", help="Merges datasets")
    merge_parser.add_argument("datasets", nargs="+",
                              help="The datasets to merge")

    args = parser.parse_args()

    if args.command == "split":
        coco_tools.split(args.dataset)


if __name__ == "__main__":
    main()
