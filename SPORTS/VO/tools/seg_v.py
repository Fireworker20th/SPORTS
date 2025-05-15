from panopticapi.utils import id2rgb, rgb2id
import numpy as np
from PIL import Image
import os
scene_dir = ['000001', '000002', '000006', '000018', '000020']
root = "shared_data/vknet_eca/"
outroot = "shared_data/vknet_eca_v/"
if not os.path.exists(outroot):
    os.makedirs(outroot)

filelist = os.listdir(root)
filelist.sort()
for scene in scene_dir:
    if scene == '000001':
        #for a in range(0, 63):
        for a in range(0, 321):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1

            cat_map = cat_map * 10000
            mask = ins_map > 0
            ins_map[mask] = (ins_map[mask] + 14)*10000
            ins_map = np.uint32(ins_map)
            ins_map = id2rgb(ins_map)
            mask2 = ins_map > 0

            map = cat_map
            map = np.uint32(map)
            map = id2rgb(map)
            map[mask2] = ins_map[mask2]

            filename = "0001" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot, filename))
    elif scene == '000002':
        #for a in range(0, 33):
        for a in range(0, 167):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1

            cat_map = cat_map * 10000
            mask = ins_map > 0
            ins_map[mask] = (ins_map[mask] + 14) * 10000
            ins_map = np.uint32(ins_map)
            ins_map = id2rgb(ins_map)
            mask2 = ins_map > 0

            map = cat_map
            map = np.uint32(map)
            map = id2rgb(map)
            map[mask2] = ins_map[mask2]

            filename = "0002" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot, filename))
    elif scene == '000006':
        #for a in range(0, 38):
        for a in range(0, 194):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1

            cat_map = cat_map * 10000
            mask = ins_map > 0
            ins_map[mask] = (ins_map[mask] + 14) * 10000
            ins_map = np.uint32(ins_map)
            ins_map = id2rgb(ins_map)
            mask2 = ins_map > 0

            map = cat_map
            map = np.uint32(map)
            map = id2rgb(map)
            map[mask2] = ins_map[mask2]

            filename = "0006" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot, filename))
    elif scene == '000018':
        #for a in range(0, 48):
        for a in range(0, 243):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1

            cat_map = cat_map * 10000
            mask = ins_map > 0
            ins_map[mask] = (ins_map[mask] + 14) * 10000
            ins_map = np.uint32(ins_map)
            ins_map = id2rgb(ins_map)
            mask2 = ins_map > 0

            map = cat_map
            map = np.uint32(map)
            map = id2rgb(map)
            map[mask2] = ins_map[mask2]

            filename = "0018" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot, filename))
    elif scene == '000020':
        #for a in range(0, 119):
        for a in range(0, 599):
            img_id = "{:06d}".format(a)
            img_new_id = "{:05d}".format(a)
            imgfile_cat = os.path.join(root, scene + '_' + img_id + '_' + 'cat.png')
            imgfile_ins = os.path.join(root, scene + '_' + img_id + '_' + 'ins.png')
            cat_map = np.array(Image.open(imgfile_cat))
            ins_map = np.array(Image.open(imgfile_ins))

            cat_map = cat_map + 1

            cat_map = cat_map * 10000
            mask = ins_map > 0
            ins_map[mask] = (ins_map[mask] + 14) * 10000
            ins_map = np.uint32(ins_map)
            ins_map = id2rgb(ins_map)
            mask2 = ins_map > 0

            map = cat_map
            map = np.uint32(map)
            map = id2rgb(map)
            map[mask2] = ins_map[mask2]

            filename = "0020" + "_" + img_new_id + ".png"
            Image.fromarray(map).save(os.path.join(outroot, filename))

print("data change over")