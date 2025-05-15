from panopticapi.utils import id2rgb, rgb2id
import numpy as np
from PIL import Image
import os
scene_dir = ['000001', '000002', '000006', '000018', '000020']
root = "shared_data/panfpn_fusion/"
outroot = "shared_data/vknet_for_augv2/pvo"
if not os.path.exists(outroot):
    os.makedirs(outroot)

filelist = os.listdir(root)
filelist.sort()
for scene in scene_dir:
    if scene == '000001':
        for a in range(0, 63):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a+321)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1
            #cat_map[0,0] = 0

            cat_map = cat_map * 10000
            map = cat_map + ins_map
            map = np.uint32(map)
            map = id2rgb(map)

            filename = "0001" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot, filename))
    elif scene == '000002':
        for a in range(0, 33):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a+167)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1
            #cat_map[0, 0] = 0

            cat_map = cat_map * 10000
            map = cat_map + ins_map
            map = np.uint32(map)
            map = id2rgb(map)

            filename = "0002" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot, filename))
    elif scene == '000006':
        for a in range(0, 38):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a+194)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1
            #cat_map[0, 0] = 0

            cat_map = cat_map * 10000
            map = cat_map + ins_map
            map = np.uint32(map)
            map = id2rgb(map)

            filename = "0006" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot, filename))
    elif scene == '000018':
        for a in range(0, 48):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a+243)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1
            #cat_map[0, 0] = 0

            cat_map = cat_map * 10000
            map = cat_map + ins_map
            map = np.uint32(map)
            map = id2rgb(map)

            filename = "0018" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot, filename))
    elif scene == '000020':
        for a in range(0, 119):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a+599)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1
            #cat_map[0, 0] = 0

            cat_map = cat_map * 10000
            map = cat_map + ins_map
            map = np.uint32(map)
            map = id2rgb(map)

            filename = "0020" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot, filename))

print("data change over")