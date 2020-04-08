import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

baseline_dir = '/home/xuanrun/Scene/logs/exp00/visual/'
my_dir = '/home/xuanrun/Scene/logs/exp04/visual/'
save_dir = '/home/xuanrun/Scene/logs/exp04/combined/'

for i in range(1000):
    img_gt = plt.imread(osp.join(baseline_dir, '%05d_gt.jpg' % i))[465:-450, 210:-200, :]
    img_bl = plt.imread(osp.join(baseline_dir, '%05d_pred.jpg' % i))[465:-450, 210:-200, :]
    img_my = plt.imread(osp.join(my_dir, '%05d_pred.jpg' % i))[465:-450, 210:-200, :]
    img = np.concatenate([img_gt, img_bl, img_my], axis = 1)
    plt.imsave(osp.join(save_dir, '%05d.jpg' % i), img)