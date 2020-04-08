import os
import os.path as osp
import json
import argparse
import tqdm
import numpy as np
import skimage.io as io
from collections import OrderedDict

try:
    import categories
    from settings import *
except:
    import dataset_tool.categories as categories
    from dataset_tool.settings import *


def load_fileinfo():
    with open(osp.join(meta_path, 'fileinfo.json'), 'r') as f:
        ret = json.load(f)
    di = OrderedDict()
    for scene in sorted([int(t) for t in ret]):
        di[int(scene)] = OrderedDict()
        for room in sorted([int(t) for t in ret[str(scene)]]):
            di[int(scene)][int(room)] = []
            for position in sorted([int(t) for t in ret[str(scene)][str(room)]]):
                di[int(scene)][int(room)].append(position)
    return di

def load_image(scene, room, position):
    full_path = osp.join(dataset_path, 'scene_%05d'%scene, '2D_rendering', str(room), 'perspective', 'full', str(position))
    rgb = io.imread(osp.join(full_path, 'rgb_rawlight.png'))
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    sem = io.imread(osp.join(full_path, 'semantic_category.png'))
    if len(sem.shape) > 2:
        sem = sem[:, :, 0]
    inst = io.imread(osp.join(full_path, 'instance.png'))
    bbox3d_path = osp.join(dataset_path, 'scene_%05d'%scene, 'bbox_3d.json')
    with open(bbox3d_path, 'r') as fin:
        bbox3d = json.load(fin)
    idcatemap_path = osp.join(dataset_path, 'scene_%05d'%scene, 'idcate_map.json')
    with open(idcatemap_path, 'r') as fin:
        idcatemap = json.load(fin)
    with open(osp.join(full_path, 'bbox_2d.json'), 'r') as fin:
        bbox2d = json.load(fin)

    res = {}
    res['rgb'] = rgb
    res['semantic'] = sem
    res['instance'] = inst
    res['bbox3d'] = bbox3d
    res['bbox2d'] = bbox2d
    res['idcatemap'] = idcatemap
    res['full_path'] = full_path
    return res

'''
    由于C++库只能读取位宽为8bit的图像，需要把这个位宽16的inst图转换成位宽为8的inst图。
    其中，R通道为前8个bit，G通道为中间8个bit
'''
def convert_instance(inst):
    assert inst.dtype == np.uint16
    h, w = inst.shape
    new = np.zeros((h, w, 4), dtype=np.uint8)
    new[:, :, 0] = inst / 256
    new[:, :, 1] = inst % 256
    new[:, :, 3] = 255
    return new

'''
    需要从bbox_3d中提取出所有instance的位置信息
    instance:   bbox_2d     √
                category    √
    
    需要信息：
        每个Scene下物体id-category对应关系
        每个Scene中物体对的xyz距离
'''
# if __name__ == '__main__':
#     fileinfo = load_fileinfo()
#     for scene in tqdm.tqdm(fileinfo):
#         for room in fileinfo[scene]:
#             for position in fileinfo[scene][room]:
#                 res = load_image(scene, room, position)
#                 for i in res:
#                     print(i)
#                 exit(0)