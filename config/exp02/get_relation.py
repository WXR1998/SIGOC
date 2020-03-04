import os
import sys
import time
import numpy as np
import imgaug
import shutil
import skimage.io as io
import json

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
sys.path.append(ROOT_DIR)

from config import Config
import utils

import dataset_tool.dataset as dataset_tool
import dataset_tool.categories as categories
from dataset_tool.settings import *

relation_dir_path = os.path.join(ROOT_DIR, '../dataset/Structured3D/meta/relation')
savefile_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'relation.txt')
num_classes = 41

rel_table = np.zeros((num_classes, num_classes), dtype=np.float32)

def distance(x):
    assert len(x) == 3
    return (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) ** 0.5

'''
    二者平均距离越近，其同时出现的可能性越高
    所以，只需要得到类别之间两两的距离均值，比较，即可得到
'''

for roots, dirs, files in os.walk(relation_dir_path):
    for f in files:
        if f.split('.')[-1] == 'json':
            with open(os.path.join(roots, f), 'r') as fin:
                curr_cate = int(f.split('.')[0])
                print('正在处理类别 %d' % curr_cate)
                dic = json.load(fin)
                dist_sum = 0
                for i in range(num_classes):
                    dists = dic[str(i)]
                    dis_list = []
                    for d in dists:
                        dis_list.append(distance(d))

                    rel_table[curr_cate, i] = sum(dis_list) / len(dis_list) if len(dis_list) > 0 and curr_cate != i else 1e6    # 同类则对loss无贡献
                    dist_sum += 1. / rel_table[curr_cate, i]
                for i in range(num_classes):
                    rel_table[curr_cate, i] = (1. / rel_table[curr_cate, i]) / dist_sum

for i in range(num_classes):
    for j in range(i+1, num_classes):
        rel_table[i, j] = rel_table[j, i] = (rel_table[i, j] + rel_table[j, i]) / 2.

with open(savefile_path, 'w') as fout:
    for i in range(num_classes):
        for j in range(num_classes):
            fout.write('%.5f\t' % rel_table[i, j])
        fout.write('\n')