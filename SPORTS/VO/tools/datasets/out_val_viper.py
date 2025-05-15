from panopticapi.utils import id2rgb, rgb2id
import numpy as np
from PIL import Image
import os
root = "shared_data/final_vps_res/"
outroot = "shared_data/change_final_data"
if not os.path.exists(outroot):
    os.makedirs(outroot)
scene_dir = ['s1', 's2', 's3', 's4', 's5', 's6']

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
        thing_mask = cat_map < 14
        thing_map = map
        thing_map[thing_mask] = 0
        thing_mask1 = cat_map == 14
        thing_mask2 = cat_map == 15
        thing_mask3 = cat_map == 16
        thing_mask4 = cat_map == 17
        thing_mask5 = cat_map == 18
        thing_mask6 = cat_map == 19
        thing_mask7 = cat_map == 20
        thing_mask8 = cat_map == 21
        thing_mask9 = cat_map == 22
        thing_mask10 = cat_map == 23

        if thing_mask1.all() != False:
            thing_map[thing_mask1] = thing_map[thing_mask1] - 140000
        if thing_mask2.all() != False:
            thing_map[thing_mask2] = thing_map[thing_mask1] - 150000
        if thing_mask3.all() != False:
            thing_map[thing_mask3] = thing_map[thing_mask1] - 160000
        if thing_mask4.all() != False:
            thing_map[thing_mask4] = thing_map[thing_mask1] - 170000
        if thing_mask5.all() != False:
            thing_map[thing_mask5] = thing_map[thing_mask1] - 180000
        if thing_mask6.all() != False:
            thing_map[thing_mask6] = thing_map[thing_mask1] - 190000
        if thing_mask7.all() != False:
            thing_map[thing_mask7] = thing_map[thing_mask1] - 200000
        if thing_mask8.all() != False:
            thing_map[thing_mask8] = thing_map[thing_mask1] - 210000
        if thing_mask9.all() != False:
            thing_map[thing_mask9] = thing_map[thing_mask1] - 220000
        if thing_mask10.all() != False:
            thing_map[thing_mask10] = thing_map[thing_mask1] - 230000

        ig = cat_map == 0
        cat_map[ig] = 24
        cat_map = cat_map - 1
        #cat_map[0,0] = cat_map[1,0]

        img_new_id = "{:06d}".format(img_id)
        filename1 = "0000" + seq_id + "_" + img_new_id + "_cat" + ".png"
        filename2 = "0000" + seq_id + "_" + img_new_id + "_ins" + ".png"

        Image.fromarray(cat_map).save(os.path.join(outroot, filename1))
        Image.fromarray(thing_map).save(os.path.join(outroot, filename2))


print("data change over")