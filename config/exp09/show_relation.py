import os
import numpy as np
from s3d import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
sys.path.append(ROOT_DIR)

from dataset_tool import categories




data = S3DDataset()

rel = data.load_rel_coefs()
bia = data.load_rel_bias()

fig = plt.figure(figsize=(16, 16))

ax = fig.add_subplot(111)
def show_img(arr):
    N = arr.shape[0]
    # for i in range(N):
    #     for j in range(N):
    #         if (arr[i, j] != 0):
    #             print('%3d%15s - %3d%15s : %d' % (i, categories.category2name(i), j, categories.category2name(j), int(arr[i, j])))
    return ax.imshow(arr, cmap='Blues', interpolation='none', vmin=np.min(arr), vmax=np.max(arr), aspect='equal')
im = show_img(rel)
plt.colorbar(im, shrink=0.5)

plt.savefig('fig.jpg')