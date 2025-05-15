from panopticapi.utils import id2rgb, rgb2id
import numpy as np
from PIL import Image
import os
import glob
scene_dir = ['0001', '0002', '0006', '0018', '0020']
root = '/media/jydai/C0FED904FED8F39E/download_jyd/project/vps_output/pvo_eca_val/'
outroot = "/media/jydai/C0FED904FED8F39E/download_jyd/project/vps_output/pvo_eca_val_change/"
if not os.path.exists(outroot):
    os.makedirs(outroot)

filelist = os.listdir(root)
filelist.sort()
for scene in scene_dir:
    if scene == '0001':
        s = 's1'
        for a in range(321, 384):
        #for a in range(0, 321):
            img_id = "{:05d}".format(a)
            imgfile = os.path.join(os.path.join(root, s), scene + '_' + img_id + '.png')
            pan_map = np.array(Image.open(imgfile))
            pan_map = rgb2id(pan_map)
            cat_map = pan_map // 10000

            mask_stuff = cat_map < 12
            pan_map[mask_stuff] = 0
            ins_map = pan_map % 10000

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

            filename = "0001" + "_" + img_id + ".png"
            outroot2 = os.path.join(outroot, s)
            if not os.path.exists(outroot2):
                os.makedirs(outroot2)
            Image.fromarray(map).save(os.path.join(outroot2, filename))
    elif scene == '0002':
        s = 's2'
        for a in range(167, 200):
        #for a in range(0, 167):
            img_id = "{:05d}".format(a)
            imgfile = os.path.join(
                os.path.join(root, s),
                scene + '_' + img_id + '.png')
            pan_map = np.array(Image.open(imgfile))
            pan_map = rgb2id(pan_map)
            cat_map = pan_map // 10000

            mask_stuff = cat_map < 12
            pan_map[mask_stuff] = 0
            ins_map = pan_map % 10000

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

            filename = "0002" + "_" + img_id + ".png"
            outroot2 = os.path.join(outroot, s)
            if not os.path.exists(outroot2):
                os.makedirs(outroot2)
            Image.fromarray(map).save(os.path.join(outroot2, filename))
    elif scene == '0006':
        s = 's6'
        for a in range(194, 232):
        #for a in range(0, 194):
            img_id = "{:05d}".format(a)
            imgfile = os.path.join(
                os.path.join(root, s),
                scene + '_' + img_id + '.png')
            pan_map = np.array(Image.open(imgfile))
            pan_map = rgb2id(pan_map)
            cat_map = pan_map // 10000

            mask_stuff = cat_map < 12
            pan_map[mask_stuff] = 0
            ins_map = pan_map % 10000

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

            filename = "0006" + "_" + img_id + ".png"
            outroot2 = os.path.join(outroot, s)
            if not os.path.exists(outroot2):
                os.makedirs(outroot2)
            Image.fromarray(map).save(os.path.join(outroot2, filename))
    elif scene == '0018':
        s = 's18'
        for a in range(243, 291):
        #for a in range(0, 243):
            img_id = "{:05d}".format(a)
            imgfile = os.path.join(
                os.path.join(root, s),
                scene + '_' + img_id + '.png')
            pan_map = np.array(Image.open(imgfile))
            pan_map = rgb2id(pan_map)
            cat_map = pan_map // 10000

            mask_stuff = cat_map < 12
            pan_map[mask_stuff] = 0
            ins_map = pan_map % 10000

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

            filename = "0018" + "_" + img_id + ".png"
            outroot2 = os.path.join(outroot, s)
            if not os.path.exists(outroot2):
                os.makedirs(outroot2)
            Image.fromarray(map).save(os.path.join(outroot2, filename))
    elif scene == '0020':
        s = 's20'
        for a in range(599, 718):
        #for a in range(0, 599):
            img_id = "{:05d}".format(a)
            imgfile = os.path.join(
                os.path.join(root, s),
                scene + '_' + img_id + '.png')
            pan_map = np.array(Image.open(imgfile))
            pan_map = rgb2id(pan_map)
            cat_map = pan_map // 10000

            mask_stuff = cat_map < 12
            pan_map[mask_stuff] = 0
            ins_map = pan_map % 10000

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

            filename = "0020" + "_" + img_id + ".png"
            outroot2 = os.path.join(outroot, s)
            if not os.path.exists(outroot2):
                os.makedirs(outroot2)
            Image.fromarray(map).save(os.path.join(outroot2, filename))

print("data change over")