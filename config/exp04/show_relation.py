import os
import numpy as np
from s3d import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = S3DDataset()

rel = data.load_rel_coefs()

fig = plt.figure(figsize=(16, 16))

ax = fig.add_subplot(111)
def show_img(arr):
    return ax.imshow(arr, cmap='Blues', interpolation='none', vmin=np.min(arr), vmax=np.max(arr), aspect='equal')
im = show_img(rel)
plt.colorbar(im, shrink=0.5)

plt.savefig('fig.jpg')