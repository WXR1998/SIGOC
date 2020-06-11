import os
import sys
import time
import numpy as np
import imgaug
import shutil
import skimage.io as io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import argparse
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
sys.path.append(ROOT_DIR)
EXP_NAME = os.path.dirname(os.path.realpath(__file__)).split('/')[-1]

import model as modellib

import dataset_tool.dataset as dataset_tool
import dataset_tool.categories as categories
from dataset_tool.settings import *
from s3d import *
import utils
import visualize

COCO_PATH = '/home/xuanrun/Scene/logs/coco/mask_rcnn_coco.h5'

save_visual_path = '/home/xuanrun/Scene/logs/exp00/visual/'
config = None

############################################################
#  Training Tools
############################################################

def train(model):
    dataset_train = S3DDataset()
    dataset_train.load_s3d('train')
    dataset_train.prepare()

    dataset_val = S3DDataset()
    dataset_val.load_s3d('val')
    dataset_val.prepare()

    augmentation = imgaug.augmenters.Fliplr(0.5)

    print("Start training...")
    print("Training network heads...")
    model.train(dataset_train, dataset_val, 
        learning_rate=config.LEARNING_RATE,
        epochs=100,
        layers='heads',
        augmentation=augmentation)

    print("Fine tune Resnet stage 4 and up...")
    model.train(dataset_train, dataset_val, 
        learning_rate=config.LEARNING_RATE,
        epochs=300,
        layers='4+',
        augmentation=augmentation)

    print("Fine tune all layers...")
    model.train(dataset_train, dataset_val, 
        learning_rate=config.LEARNING_RATE / 10,
        epochs=400,
        layers='all',
        augmentation=augmentation)

    model.train(dataset_train, dataset_val, 
        learning_rate=config.LEARNING_RATE / 10,
        epochs=1000,
        layers='all',
        augmentation=augmentation)

############################################################
#  Evaluation Tools
############################################################

def test(model, config, limit = None, savefiledir = None):
    dataset_test = S3DDataset()
    dataset_test.load_s3d('test')
    dataset_test.prepare()

    print("Start inferencing on %d images..." % len(dataset_test.image_info))

    APs1, APs2 = [], []
    subset = {}
    with open(os.path.join(meta_path, 'subset.txt'), 'r') as fin:
        for l in fin:
            subset[int(l)] = 1

    import tqdm
    for i in tqdm.tqdm(range(len(dataset_test.image_info))):
        if limit is not None:
            if i >= int(limit):
                continue
        if i not in subset:
            continue
        image, meta, gt_class_ids, gt_bbox, gt_mask = modellib.load_image_gt(dataset_test, config, i)
        result = model.detect([image], verbose=0)[0]

        bbox = result['rois']
        mask = result['masks']
        class_ids = result['class_ids']
        scores = result['scores']
        second_class_ids = result['second_class_ids']
        second_scores = result['second_scores']
        probs = result['probs'][0]

        # print(class_ids)
        # print(second_class_ids)
        # print(scores)
        # print(second_scores)

        # print(bbox.shape, gt_bbox.shape)
        # print(mask.shape, gt_mask.shape)
        # print(class_ids.shape, gt_class_ids.shape)
        # print(scores.shape)

        def savefig():
            visualize.display_instances(image, gt_bbox, gt_mask, gt_class_ids, 
                [categories.category2name(i) for i in range(categories.cate_cnt)], 
                savefilename=os.path.join(save_visual_path, '%05d_gt.jpg' % i))
            visualize.display_instances(image, bbox, mask, class_ids, 
                [categories.category2name(i) for i in range(categories.cate_cnt)], 
                savefilename=os.path.join(save_visual_path, '%05d_pred.jpg' % i))
        
        # @timer
        def secondClassResults():
            # 基础的结果
            basemAP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_ids, gt_mask, bbox, class_ids, scores, mask)
            delta = 0.0

            # 计入概率次高分类之后的结果
            for i in range(len(class_ids)):
                ori = class_ids[i]
                class_ids[i] = second_class_ids[i]
                mAP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_ids, gt_mask, bbox, class_ids, scores, mask)
                class_ids[i] = ori
                if mAP - basemAP > 0:
                    delta += mAP - basemAP

            if basemAP >= 0 and basemAP <= 1:
                APs2.append(basemAP + delta)
        
        # @timer
        def basicResults():
            basemAP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_ids, gt_mask, bbox, class_ids, scores, mask)
            if basemAP >= 0 and basemAP <= 1:
                APs1.append(basemAP)
            # visualize.display_instances(image, gt_bbox, gt_mask, gt_class_ids, [categories.category2name(i) for i in range(categories.cate_cnt)], savefilename=os.path.join(savefiledir, 'visual', '%05d_A.jpg' % i))
            # visualize.display_instances_second_class(image, bbox, mask, class_ids, second_class_ids, [categories.category2name(i) for i in range(categories.cate_cnt)], scores, second_scores, savefilename=os.path.join(savefiledir, 'visual', '%05d_B.jpg' % i))

        basicResults()

    
    print('%.3f' % np.mean(APs1))

############################################################
#  Main Script
############################################################

def main():
    global config
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
        metavar="<command>", 
        help="'train' or 'evaluate'")
    parser.add_argument("--logs", required=False, 
        metavar="/path/to/logs/", 
        help="Path to store logs and checkpoints.",
        default=os.path.join(ROOT_DIR, "../logs", EXP_NAME))
    parser.add_argument('--weights', required=False,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco' or 'last'",
        default='coco')
    parser.add_argument('--limit', required=False,
        metavar="Testset size",
        help="Testset Size, availible when evaluation.",
        default=None)
    args = parser.parse_args()
    assert args.command in ['train', 'evaluate']

    if not os.path.exists(args.logs):
        os.mkdir(args.logs)

    config = S3DConfig()
    if args.command == 'evaluate':
        config.IMAGES_PER_GPU = 1
        config.BATCH_SIZE = 1
    config.display()

    if args.command == 'train':
        model = modellib.MaskRCNN(mode='training', config=config,
            model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode='inference', config=config,
            model_dir=args.logs)
        
    if args.weights.lower() == 'last':
        weight_path = model.find_last()
    elif args.weights.lower() == 'coco':
        weight_path = COCO_PATH
    else:
        weight_path = args.weights
    
    if args.weights.lower() == 'coco':
        model.load_weights(weight_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weight_path, by_name=True)
        
    if args.command == 'train':
        train(model)
    else:
        test(model, config, args.limit, args.logs)

if __name__ == '__main__':
    main()