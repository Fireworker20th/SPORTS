from panopticapi.utils import id2rgb, rgb2id
import numpy as np
from PIL import Image
import os
scene_dir = ['000001', '000002', '000006', '000018', '000020', '000009', '000010']
root = "shared_data/EMA_TRACK_UP/"
outroot1 = "shared_data/vknet_for_aug/s1"
outroot2 = "shared_data/vknet_for_aug/s2"
outroot3 = "shared_data/vknet_for_aug/s6"
outroot4 = "shared_data/vknet_for_aug/s18"
outroot5 = "shared_data/vknet_for_aug/s20"
outroot6 = "shared_data/vknet_for_aug/s9"
outroot7 = "shared_data/vknet_for_aug/s10"

if not os.path.exists(outroot1):
    os.makedirs(outroot1)
if not os.path.exists(outroot2):
    os.makedirs(outroot2)
if not os.path.exists(outroot3):
    os.makedirs(outroot3)
if not os.path.exists(outroot4):
    os.makedirs(outroot4)
if not os.path.exists(outroot5):
    os.makedirs(outroot5)
filelist = os.listdir(root)
filelist.sort()
for scene in scene_dir:

    if scene == '000001':
        for a in range(0, 447):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1

            cat_map = cat_map * 10000
            map = cat_map + ins_map
            map = np.uint32(map)
            map = id2rgb(map)

            filename = "0001" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot1, filename))
    elif scene == '000002':
        for a in range(0, 233):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1

            cat_map = cat_map * 10000
            map = cat_map + ins_map
            map = np.uint32(map)
            map = id2rgb(map)

            filename = "0002" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot2, filename))
    elif scene == '000006':
        for a in range(0, 270):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1

            cat_map = cat_map * 10000
            map = cat_map + ins_map
            map = np.uint32(map)
            map = id2rgb(map)

            filename = "0006" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot3, filename))
    elif scene == '000018':
        for a in range(0, 339):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1

            cat_map = cat_map * 10000
            map = cat_map + ins_map
            map = np.uint32(map)
            map = id2rgb(map)

            filename = "0018" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot4, filename))
    elif scene == '000020':
        for a in range(0, 837):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1

            cat_map = cat_map * 10000
            map = cat_map + ins_map
            map = np.uint32(map)
            map = id2rgb(map)

            filename = "0020" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot5, filename))
    '''
    if scene == '000009':
        for a in range(0, 1461):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1

            cat_map = cat_map * 10000
            map = cat_map + ins_map
            map = np.uint32(map)
            map = id2rgb(map)

            filename = "0009" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot6, filename))
    
    if scene == '000010':
        for a in range(0, 1201):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1

            cat_map = cat_map * 10000
            map = cat_map + ins_map
            map = np.uint32(map)
            map = id2rgb(map)

            filename = "0010" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot7, filename))
    '''
print("data change over")