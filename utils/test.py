import skimage.io as io
import numpy as np

a = np.zeros((2, 4), dtype=np.uint16)

a[0, 0] = 65534
a[1, 0] = 233
a[0, 1] = 512
a[1, 1] = 511

io.imsave('test.png', a)

b = io.imread('test.png')
print(b.dtype)