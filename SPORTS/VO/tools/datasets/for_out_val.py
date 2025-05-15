from panopticapi.utils import id2rgb, rgb2id
import numpy as np
from PIL import Image
import os
root = "shared_data/final_vps_res/"
outroot = "shared_data/change_final_data"
if not os.path.exists(outroot):
    os.makedirs(outroot)
scene_dir = ['s1', 's2', 's6', 's18', 's20']

for scene in scene_dir:
    path = os.path.join(root, scene)
    filelist = os.listdir(path)
    filelist.sort()
    for f in filelist:
        part1 = f.rsplit("_")[0]
        part2 = f.rsplit("_")[1]
        part2 = part2.split(".")
        part2 = part2[0]
        seq_id = part1[-2:]
        img_id = int(part2)

        img_folder = os.path.join(path, f)
        map = np.array(Image.open(img_folder))
        map = np.uint32(map)
        map = rgb2id(map)

        cat_map = map // 10000
        thing_mask = cat_map < 12
        thing_map = map
        thing_map[thing_mask] = 0
        thing_mask1 = cat_map == 12
        thing_mask2 = cat_map == 13
        thing_mask3 = cat_map == 14
        thing_map[thing_mask1] = thing_map[thing_mask1] - 120000
        thing_map[thing_mask2] = thing_map[thing_mask2] - 130000
        thing_map[thing_mask3] = thing_map[thing_mask3] - 140000

        ig = cat_map == 0
        cat_map[ig] = 15
        cat_map = cat_map - 1
        cat_map[0,0] = cat_map[1,0]

        if scene == 's1':
            img_new_id = "{:06d}".format(img_id - 321)
            filename1 = "0000" + seq_id + "_" + img_new_id + "_cat" + ".png"
            filename2 = "0000" + seq_id + "_" + img_new_id + "_ins" + ".png"

            Image.fromarray(cat_map).save(os.path.join(outroot, filename1))
            Image.fromarray(thing_map).save(os.path.join(outroot, filename2))

        if scene == 's2':
            img_new_id = "{:06d}".format(img_id - 167)
            filename1 = "0000" + seq_id + "_" + img_new_id + "_cat" + ".png"
            filename2 = "0000" + seq_id + "_" + img_new_id + "_ins" + ".png"

            Image.fromarray(cat_map).save(os.path.join(outroot, filename1))
            Image.fromarray(thing_map).save(os.path.join(outroot, filename2))

        if scene == 's6':
            img_new_id = "{:06d}".format(img_id - 194)
            filename1 = "0000" + seq_id + "_" + img_new_id + "_cat" + ".png"
            filename2 = "0000" + seq_id + "_" + img_new_id + "_ins" + ".png"

            Image.fromarray(cat_map).save(os.path.join(outroot, filename1))
            Image.fromarray(thing_map).save(os.path.join(outroot, filename2))

        if scene == 's18':
            img_new_id = "{:06d}".format(img_id - 243)
            filename1 = "0000" + seq_id + "_" + img_new_id + "_cat" + ".png"
            filename2 = "0000" + seq_id + "_" + img_new_id + "_ins" + ".png"

            Image.fromarray(cat_map).save(os.path.join(outroot, filename1))
            Image.fromarray(thing_map).save(os.path.join(outroot, filename2))

        if scene == 's20':
            img_new_id = "{:06d}".format(img_id - 599)
            filename1 = "0000" + seq_id + "_" + img_new_id + "_cat" + ".png"
            filename2 = "0000" + seq_id + "_" + img_new_id + "_ins" + ".png"

            Image.fromarray(cat_map).save(os.path.join(outroot, filename1))
            Image.fromarray(thing_map).save(os.path.join(outroot, filename2))

print("data change over")



