import argparse
import json
from pathlib import Path
from coco_tools.merge import merge_images, merge_anno


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--anno', '-a', nargs='+', required=True)
    parser.add_argument('--imgdir', '-i', nargs='+', required=True)
    parser.add_argument('--outdir', '-o', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    anno_files = args.anno
    img_dirs = args.imgdir
    out_dir = args.outdir
    
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    dataset_name = out_dir.name

    # Merge annotation files
    out_anno_dir = out_dir / Path('annotations')
    out_anno_dir.mkdir(exist_ok=True)

    out_anno_file = out_anno_dir / Path('instances_{}.json'.format(str(dataset_name)))

    merge_anno(anno_files, str(out_anno_file))

    # Merge images from the datasets
    out_img_dir = out_dir / dataset_name
    merge_images(anno_files, img_dirs, str(out_img_dir))

    # Sanity checks
    with open(str(out_anno_file), 'r') as fi:
        anno = json.load(fi)
    imgs = anno['images']
    for img in imgs:
        file_name = Path(img['file_name'])
        file_path = out_img_dir / file_name
        if not file_path.is_file():
            raise FileNotFoundError('{} in annotation file but not in {}'.format(file_name, str(out_img_dir)))

    all_img_ids = set([img['id'] for img in imgs])
    annotations = anno['annotations']
    for a in annotations:
        img_id = a['image_id']
        if not img_id in all_img_ids:
            raise ValueError('Could not match annotation {} with any image'.format(a['id']))


if __name__ == '__main__':
    main()

    
