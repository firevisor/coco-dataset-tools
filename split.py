import argparse
import os
import json

from coco_tools.split import split


from pycocotools.coco import COCO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Path to the dataset')
    parser.add_argument('--name', help='Name of the dataset')
    parser.add_argument('--ratio', help='List of ratios like "0.7:0.2:0.1"')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = args.dataset
    dataset_name = args.name
    anno_file = '{}/annotations/instances_{}.json'.format(dataset_path, dataset_name)

    train_anno_file = '{}/annotations/instances_{}_train.json'.format(dataset_path, dataset_name)
    val_anno_file = '{}/annotations/instances_{}_validate.json'.format(dataset_path, dataset_name)
    test_anno_file = '{}/annotations/instances_{}_test.json'.format(dataset_path, dataset_name)
    
    out_anno_files = [train_anno_file, val_anno_file, test_anno_file]
    
    split(anno_file, out_anno_files, args.ratio) 

    # Sanity checks
    coco = COCO(train_anno_file)
    coco = COCO(val_anno_file)
    coco = COCO(test_anno_file)




if __name__ == '__main__':
    main()

    
