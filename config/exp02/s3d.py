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

def timer(func):
    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapse = time.time() - start
        print("Time for func[%s] is: %.3f" % (func.__name__, elapse))
        return result
    return timed

class S3DConfig(Config):
    NAME = 'S3D'
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 40

class S3DDataset(utils.Dataset):
    def load_s3d(self, subset):
        assert subset in ['train', 'test', 'val']
        file_dict = dataset_tool.load_fileinfo()

        for i in range(1, categories.cate_cnt):
            self.add_class("s3d", i, categories.category2name(i))

        if subset == 'train':
            limit = [0, 3000]
        elif subset == 'val':
            limit = [3000, 3250]
        else:
            limit = [3250, 3500]

        count = 0
        for scene in file_dict:
            if not (scene >= limit[0] and scene < limit[1]):
                continue
            for room in file_dict[scene]:
                for position in file_dict[scene][room]:
                    count += 1
                    full_path = os.path.join(dataset_path, 'scene_%05d'%scene, '2D_rendering', str(room), 'perspective', 'full', str(position))
                    self.add_image(source="s3d", 
                        image_id=count,
                        path=self.full_path(scene, room, position),
                        width=IMAGE_WIDTH, 
                        height=IMAGE_HEIGHT,
                        scene=scene,
                        room=room,
                        position=position)
    
    def full_path(self, *param):
        assert len(param) == 1 or len(param) == 3
        if len(param) == 1:
            return self.image_info[param[0]]['path']
        else:
            scene, room, position = param
            return os.path.join(dataset_path, 'scene_%05d'%scene, '2D_rendering', str(room), 'perspective', 'full', str(position))

    def load_mask(self, image_id):
        """
            Load instance masks for the given image.

            Returns:
            masks: A bool array of shape [height, width, instance count] with 
                one mask per instance.
            class_ids: A 1D array of class IDs of the instance masks.
        """
        with open(os.path.join(self.full_path(image_id), "bbox_2d.json")) as fin:
            instance_bbox2d = json.load(fin)
        with open(os.path.join(self.full_path(image_id), "../../../../../idcate_map.json")) as fin:
            idcate_map = json.load(fin)
        instance_map = io.imread(os.path.join(self.full_path(image_id), "instance.png")) # [height, width]

        h, w = instance_map.shape
        # res = np.zeros((h, w, len(instance_bbox2d)), dtype=np.uint16)
        results = []

        # 筛去classid为0(BG)和为6(sofa)的
        instance_count = 0
        class_ids = []
        for ins in instance_bbox2d:
            res = (instance_map == int(ins))
            # res[:, :, instance_count] = instance_map == int(ins)
            label = int(idcate_map[ins])
            if label != 0 and label != 6:
                instance_count += 1
                class_ids.append(label)
                results.append(res)
            # class_ids.append(int(idcate_map[ins]))
        
        results = np.array(results)
        assert results.shape[0] == instance_count

        if len(results.shape) == 3:
            results = results.transpose(1, 2, 0)
        else:
            results = np.zeros((h, w, 0), dtype=np.uint16)
        return results.astype(np.bool), np.array(class_ids, dtype=np.uint8)

    def load_image(self, image_id):
        """
            Load the specified image and return a [H, W, 3] Numpy array.
        """
        image = io.imread(os.path.join(self.image_info[image_id]['path'], "rgb_rawlight.png"))
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_rel_coefs(self):
        """
            Load the relation coefs.

            Returns:
            coefs:  [41, 41]
        """
        rel = np.zeros((S3DConfig.NUM_CLASSES, S3DConfig.NUM_CLASSES))
        with open(os.path.join(meta_path, 'relation/relation.txt'), 'r') as fin:
            for i, l in enumerate(fin):
                inputs = [float(x) for x in l.split()]
                for j in range(S3DConfig.NUM_CLASSES):
                    rel[i, j] = inputs[j]
        return rel
        
if __name__ == '__main__':
    pass