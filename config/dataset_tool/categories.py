'''
    颜色-分类以及分类-名称的转换
'''
import numpy as np
import os

category_names = ['BG']
category_mapping = [
    (0, 0, 0),
    (174, 199, 232),		# wall
    (152, 223, 138),		# floor
    (31, 119, 180), 		# cabinet
    (255, 187, 120),		# bed
    (188, 189, 34), 		# chair
    (140, 86, 75),  		# sofa
    (255, 152, 150),		# table
    (214, 39, 40),  		# door
    (197, 176, 213),		# window
    (148, 103, 189),		# bookshelf
    (196, 156, 148),		# picture
    (23, 190, 207), 		# counter
    (178, 76, 76),
    (247, 182, 210),		# desk
    (66, 188, 102),
    (219, 219, 141),		# curtain
    (140, 57, 197),
    (202, 185, 52),
    (51, 176, 203),
    (200, 54, 131),
    (92, 193, 61),
    (78, 71, 183),
    (172, 114, 82),
    (255, 127, 14), 		# refrigerator
    (91, 163, 138),
    (153, 98, 156),
    (140, 153, 101),
    (158, 218, 229),		# shower curtain
    (100, 125, 154),
    (178, 127, 135),
    (120, 185, 128),
    (146, 111, 194),
    (44, 160, 44),  		# toilet
    (112, 128, 144),		# sink
    (96, 207, 209),
    (227, 119, 194),		# bathtub
    (213, 92, 176),
    (94, 106, 211),
    (82, 84, 163),  		# otherfurn
    (100, 85, 144)
    ]
color_mapping = -np.ones(196)



def category2name(c):
    if type(c) is str:
        c = int(c)
    assert c >= 0 and c < 41
    return category_names[c]

def category2color(c):
    assert c >= 0 and c < 41
    return category_mapping[c]

def color2category(r, g, b):
    h = cate_hash([r, g, b])
    col = color_mapping[h]
    if col == -1:
        raise Exception('Invalid color in semantic.png.')
    return col

def cate_hash(cate, m = 196):
    return (cate[0] * 91 + cate[1] * 23 + cate[2] * 47) % m

with open(os.path.join(os.path.split(__file__)[0], 'cate.txt'), 'r') as f:
    for line in f:
        category_names.append(line[line.find('\t')+1:-1])

cate_cnt = 0
for col in category_mapping:
    h = cate_hash(col)
    color_mapping[h] = cate_cnt
    cate_cnt += 1
