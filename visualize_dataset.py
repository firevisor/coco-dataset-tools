
import cv2
import numpy as np
import pandas as pd
import glob
import os
import tqdm
import json
import copy
import argparse
import scipy.ndimage.measurements
from tensorpack.utils import logger, viz
from tensorpack.utils.timer import timed_operation
from tensorpack.utils.palette import PALETTE_RGB
import pycocotools.mask as cocomask
from six.moves import zip


class COCODetection(object):
    # handle the weird (but standard) split of train and val

    # Not used
    _INSTANCE_TO_BASEDIR = {
        'valminusminival2014': 'val2014',
        'minival2014': 'val2014',
    }

    COCO_id_to_category_id = {1: 1, 2: 2, 3: 3, 5: 4, 6: 5}
    category_id_to_COCO_id = {v: k for k, v in COCO_id_to_category_id.items()}
    """
    Mapping from the incontinuous COCO category id to an id in [1, #category]
    For your own dataset, this should usually be an identity mapping.
    """

    def __init__(self, imgdir, annofile):
        self._imgdir = os.path.realpath(imgdir)
        self.name = self._imgdir
        assert os.path.isdir(self._imgdir), self._imgdir
        annotation_file = os.path.realpath(annofile)
        print(os.path.isfile(annotation_file))
        assert os.path.isfile(annotation_file), annotation_file
        from pycocotools.coco import COCO
        self.coco = COCO(annotation_file)
        logger.info("Instances loaded from {}.".format(annotation_file))

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    def print_coco_metrics(self, json_file):
        """
        Args:
            json_file (str): path to the results json file in coco format
        Returns:
            dict: the evaluation metrics
        """
        from pycocotools.cocoeval import COCOeval
        ret = {}
        cocoDt = self.coco.loadRes(json_file)
        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        fields = ['IoU=0.5:0.95', 'IoU=0.5',
                  'IoU=0.75', 'small', 'medium', 'large']
        for k in range(6):
            ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

        json_obj = json.load(open(json_file))
        if len(json_obj) > 0 and 'segmentation' in json_obj[0]:
            cocoEval = COCOeval(self.coco, cocoDt, 'segm')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            for k in range(6):
                ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
        return ret

    def load(self, add_gt=True, add_mask=False):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask

        Returns:
            a list of dict, each has keys including:
                'image_id', 'file_name',
                and (if add_gt is True) 'boxes', 'class', 'is_crowd', and optionally
                'segmentation'.
        """
        if add_mask:
            assert add_gt
        with timed_operation('Load Groundtruth Boxes for {}'.format(self.name)):
            img_ids = self.coco.getImgIds()
            img_ids.sort()
            # list of dict, each has keys: height,width,id,file_name
            imgs = self.coco.loadImgs(img_ids)

            for img in tqdm.tqdm(imgs):
                img['image_id'] = img.pop('id')
                self._use_absolute_file_name(img)
                if add_gt:
                    self._add_detection_gt(img, add_mask)
            return imgs

    def _use_absolute_file_name(self, img):
        """
        Change relative filename to abosolute file name.
        """
        img['file_name'] = os.path.join(
            self._imgdir, img['file_name'])
        assert os.path.isfile(img['file_name']), img['file_name']

    def _add_detection_gt(self, img, add_mask):
        """
        Add 'boxes', 'class', 'is_crowd' of this image to the dict, used by detection.
        If add_mask is True, also add 'segmentation' in coco poly format.
        """
        # ann_ids = self.coco.getAnnIds(imgIds=img['image_id'])
        # objs = self.coco.loadAnns(ann_ids)
        # equivalent but faster than the above two lines
        objs = self.coco.imgToAnns[img['image_id']]

        # clean-up boxes
        valid_objs = []
        width = img['width']
        height = img['height']
        for objid, obj in enumerate(objs):
            if obj.get('ignore', 0) == 1:
                continue
            x1, y1, w, h = obj['bbox']
            # bbox is originally in float
            # x1/y1 means upper-left corner and w/h means true w/h. This can be verified by segmentation pixels.
            # But we do make an assumption here that (0.0, 0.0) is upper-left corner of the first pixel

            x1 = np.clip(float(x1), 0, width)
            y1 = np.clip(float(y1), 0, height)
            w = np.clip(float(x1 + w), 0, width) - x1
            h = np.clip(float(y1 + h), 0, height) - y1
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 1 and w > 0 and h > 0 and w * h >= 4:
                obj['bbox'] = [x1, y1, x1 + w, y1 + h]
                valid_objs.append(obj)

                if add_mask:
                    segs = obj['segmentation']
                    if not isinstance(segs, list):
                        assert obj['iscrowd'] == 1
                        obj['segmentation'] = None
                    else:
                        valid_segs = [np.asarray(
                            p).reshape(-1, 2).astype('float32') for p in segs if len(p) >= 6]
                        if len(valid_segs) == 0:
                            logger.error("Object {} in image {} has no valid polygons!".format(
                                objid, img['file_name']))
                        elif len(valid_segs) < len(segs):
                            logger.warn("Object {} in image {} has invalid polygons!".format(
                                objid, img['file_name']))

                        obj['segmentation'] = valid_segs

        # all geometrically-valid boxes are returned
        boxes = np.asarray([obj['bbox']
                            for obj in valid_objs], dtype='float32')  # (n, 4)
        cls = np.asarray([
            self.COCO_id_to_category_id[obj['category_id']]
            for obj in valid_objs], dtype='int32')  # (n,)
        is_crowd = np.asarray([obj['iscrowd']
                               for obj in valid_objs], dtype='int8')

        # add the keys
        img['boxes'] = boxes        # nx4
        img['class'] = cls          # n, always >0
        img['is_crowd'] = is_crowd  # n,
        if add_mask:
            # also required to be float32
            img['segmentation'] = [
                obj['segmentation'] for obj in valid_objs]

    def getClassNameFromSample(self, class_id):
        return self.coco.loadCats(self.category_id_to_COCO_id[int(class_id)])[0]["name"]

    @staticmethod
    def load_many(basedir, names, add_gt=True, add_mask=False):
        """
        Load and merges several instance files together.

        Returns the same format as :meth:`COCODetection.load`.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            coco = COCODetection(basedir, n)
            ret.extend(coco.load(add_gt, add_mask=add_mask))
        return ret


def getClassesFromImg(img):
    return img["class"]


def getMasksFromImg(img):
    is_crowd = img['is_crowd']
    segmentation = copy.deepcopy(img['segmentation'])
    segmentation = [segmentation[k]
                    for k in range(len(segmentation)) if not is_crowd[k]]
    height, width = img['height'], img['width']
    # Apply augmentation on polygon coordinates.
    # And produce one image-sized binary mask per box.
    masks = []
    for polys in segmentation:
        # if not cfg.DATA.ABSOLUTE_COORD:
        #     polys = [p * width_height for p in polys]
        # polys = [aug.augment_coords(p, params) for p in polys]
        masks.append(segmentation_to_mask(polys, height, width,
                                          linear=(img['category_ids'] == [1])))
    masks = np.asarray(masks, dtype='uint8')    # values in {0, 1}
    return masks


def genBoxesFromMasks(masks):
    """Compute bounding boxes from masks.
    mask: [num_instances, height, width]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([masks.shape[0], 4], dtype=np.int32)
    for i in range(masks.shape[0]):
        m = masks[i, :, :]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)


def segmentation_to_mask(polys, height, width, linear=False):
    """
    Convert polygons to binary masks.
    Args:
        polys: a list of nx2 float array. Each array contains many (x, y) coordinates.
        height, width: dimensions of segmentation
        linear: Boolean for erosion of linear cracks
    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    assert len(polys) > 0, "Polygons are empty!"

    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    res_rle = cocomask.decode(rle)
    if linear:
        return cv2.erode(res_rle, np.ones((6, 6), np.uint8))
    else:
        return res_rle


def draw_mask(im, mask, box, label, alpha=0.5, color=None, linear=False):
    """
    Overlay a mask on top of the image.
    Args:
        im: a 3-channel uint8 image in BGR
        mask: a binary 1-channel image of the same size
        color: if None, will choose automatically
    """
    if color is None:
        color = PALETTE_RGB[np.random.choice(len(PALETTE_RGB))][::-1]
    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),
                  im * (1 - alpha) + color * alpha, im)
    im = im.astype('uint8')
    color_tuple = tuple([int(c) for c in color])
    im = viz.draw_boxes(im, box[np.newaxis, :], [label], color=color_tuple)
    cc = 1
    if linear:
        _, cc = scipy.ndimage.measurements.label(
            mask, structure=np.ones((3, 3)))
    return cc == 1, im


def parse_args():
    parser = argparse.ArgumentParser(
        description='Code for Harris corner detector tutorial.')
    parser.add_argument('--imagedir', help='Path to dataset images.')
    parser.add_argument('--jsonfile', help='Path to json file.')
    parser.add_argument('--check', help='Flag to purely check JSON', action='store_true', default = False)
    parser.add_argument('--output', help='Output Directory for images with masks', default='output_dir')
    return parser.parse_args()


def main():
    errant_imgs = set()
    args = parse_args()
    if args.check:
        ds = COCODetection(args.imagedir, args.jsonfile)
        imgs = ds.load(add_gt=True, add_mask=True)
        for img in tqdm.tqdm(imgs):
            masks = getMasksFromImg(img)
            for mask in masks:
                if 1 in img['category_ids'] or 2 in img['category_ids']:
                    _, cc = scipy.ndimage.measurements.label(mask, structure=np.ones((3, 3)))
                    if cc!=1: errant_imgs.add(img['path'])
    else:
        output_dir = args.output
        ds = COCODetection(args.imagedir, args.jsonfile)
        imgs = ds.load(add_gt=True, add_mask=True)
        os.makedirs(output_dir, exist_ok=True)
        for img in tqdm.tqdm(imgs):
            # Get masks from "img" (it's actually the image's meta rather than the image itself)
            # I follow the same naming from the Tensorpack's implementation of COCODetection
            masks = getMasksFromImg(img)
            boxes = genBoxesFromMasks(masks)
            classes = getClassesFromImg(img)  # Class IDs
            classes = [ds.getClassNameFromSample(
                clsId) for clsId in classes]  # Class names
            file_name = img['file_name']
            image_id = img['image_id']
            im = cv2.imread(file_name)
            orig_im = im.copy()
            # Draw masks, boxes and labels
            # For images with a linear crack, erosion is performed
            for i in range(masks.shape[0]):
                connected, im = draw_mask(im, masks[i], boxes[i], str(
                    classes[i]), linear=(img['category_ids'] == [1]))
                if connected == False:
                    errant_imgs.add(img['path'])

            basename = os.path.basename(file_name)
            output_path = os.path.join(output_dir, str(image_id) + '_' + basename)

            # merge original image to the image with labels
            im = np.concatenate([orig_im, im], axis=1)
            cv2.imwrite(output_path, im)

    # Errant Images where mask erosion separated the cracks
    if len(errant_imgs)!=0:
        print(f"Number of Errant Images: {len(errant_imgs)}")
        print("List of Errant Images:")
        for img_err in errant_imgs:
            print(img_err)
    else:
        print("No errant images")

if __name__ == '__main__':
    main()
