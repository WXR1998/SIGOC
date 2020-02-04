'''
    进行数据集的转换工作，把有色的semantic图转换成按照分类的semantic_category图
'''
import os
import os.path as osp
import json
import argparse
import tqdm
import numpy as np
import skimage.io as io
import categories
from dataset import load_fileinfo
from numba import jit

from settings import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Structured3D 3D Instance Picture")
    parser.add_argument("--path", required=False,
                        help="dataset path", metavar="DIR", default=dataset_path)
    parser.add_argument("--meta_path", required=False,
                        help="meta file path", metavar="METADIR", default=meta_path)
    return parser.parse_args()

args = parse_args()

def dump_fileinfo():
    di = {}
    for scene_id in tqdm.tqdm(range(3500)):
        di[scene_id] = {}
        scene_path = osp.join(args.path, 'scene_%05d' % scene_id, '2D_rendering')
        try:
            rooms = os.listdir(scene_path)
        except:
            print('Scene %d does not exist.' % scene_id)
            continue
        for room_id in rooms:
            di[scene_id][int(room_id)] = []
            room_path = osp.join(scene_path, room_id, 'perspective', 'full')
            positions = os.listdir(room_path)
            for position_id in positions:
                di[scene_id][int(room_id)].append(int(position_id))
    with open(osp.join(meta_path, 'fileinfo.json'), 'w') as f:
        json.dump(di, f, indent=4)

def convert_semantics(path):
    print(path)
    os.system('./../utils/semantic %s' % path)

def fail_pic():
    @jit
    def process_fail_pic(height, width, sem):
        for i in range(height):
            for j in range(width):
                r, g, b, a = sem[i, j]
                cate = categories.color2category(r, g, b)
                sem[i, j][0] = int(cate)
                sem[i, j][1] = int(cate)
                sem[i, j][2] = int(cate)
                sem[i, j][3] = 255
        return sem
    with open('fail.txt', 'r') as fin:
        for line in tqdm.tqdm(fin):
            full_path = line.replace('semantic.png\n', '')
            sem = io.imread(osp.join(full_path, 'semantic.png'))
            height, width, channel = sem.shape
            sem = process_fail_pic(height, width, sem)
            io.imsave(osp.join(full_path, 'semantic_category.png'), sem)

'''
    正常的图片被lodepng处理后，放入源文件夹
    如果处理不正常，会将文件名写入fail.txt，留给fail_pic处理
'''
def normal_pic():
    fileinfo = load_fileinfo()
    for scene in fileinfo:
        for room in fileinfo[scene]:
            for position in fileinfo[scene][room]:
                full_path = osp.join(args.path, 'scene_%05d'%scene, '2D_rendering', str(room), 'perspective', 'full', str(position))
                convert_semantics(full_path + '/')

'''
    处理每个Scene中，每个semantic category之间的位置关系
'''
def distance(cenA, cenB, boxA, boxB):
    def segment_distance(x1, x2, y1, y2):
        if y1 >= x1 and y1 <= x2:
            return 0.0
        if y2 >= x1 and y2 <= x2:
            return 0.0
        if y1 >= x2:
            return y1 - x2
        if y2 <= x1:
            return - x1 + y2
        if x1 >= y1 and x1 <= y2:
            return 0.0
        if x2 >= y1 and x2 <= y2:
            return 0.0
    dx = segment_distance(cenA[0] - boxA[0], cenA[0] + boxA[0], cenB[0] - boxB[0], cenB[0] + boxB[0])
    dy = segment_distance(cenA[1] - boxA[1], cenA[1] + boxA[1], cenB[1] - boxB[1], cenB[1] + boxB[1])
    dz = segment_distance(cenA[2] - boxA[2], cenA[2] + boxA[2], cenB[2] - boxB[2], cenB[2] + boxB[2])
    return np.array([dx, dy, dz], dtype=np.float)

def category_coordinate():
    fileinfo = load_fileinfo()
    result = [{str(j): [] for j in range(41)} for i in range(41)]
    for scene in fileinfo:
        print(scene)
        full_path = osp.join(args.path, 'scene_%05d'%scene)
        with open(osp.join(full_path, 'bbox_3d.json'), 'r') as fin:
            bbox_3d = json.load(fin)
        with open(osp.join(full_path, 'idcate_map.json'), 'r') as fin:
            idcate = json.load(fin)
        for i in range(len(bbox_3d)):
            for j in range(len(bbox_3d)):
                if i != j:
                    try:
                        cateA = int(idcate[str(bbox_3d[i]['ID'])])
                        cateB = int(idcate[str(bbox_3d[j]['ID'])])
                        cenA = np.array(bbox_3d[i]['centroid'], dtype=np.float)
                        boxA = np.array(bbox_3d[i]['coeffs'], dtype=np.float)
                        cenB = np.array(bbox_3d[j]['centroid'], dtype=np.float)
                        boxB = np.array(bbox_3d[j]['coeffs'], dtype=np.float)
                        dis = distance(cenA, cenB, boxA, boxB)
                        result[cateA][str(cateB)].append(list(dis))
                    except Exception as e:
                        continue
    for i in range(41):
        with open(osp.join(args.meta_path, 'relation', '%d.json'%i), 'w') as fout:
            json.dump(result[i], fout, indent=4)

if __name__ == '__main__':
    # normal_pic()
    # fail_pic()
    category_coordinate()
