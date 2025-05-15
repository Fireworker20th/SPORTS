from panopticapi.utils import id2rgb, rgb2id
import numpy as np
from PIL import Image
import os
import glob

root1 = "/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/Virtual_KITTI2/ALL_clone/stuff_TrainIds"
image_paths1 = glob.glob(os.path.join('/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/Virtual_KITTI2/ALL_clone/panoptic_gt_id', '*.png'))
outroot = "/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/v_gt_ins4/"
if not os.path.exists(outroot):
    os.makedirs(outroot)

for path in image_paths1:
    part = path.rsplit("/")[11]

    imgfile_ins = os.path.join(root1, part)
    cat_map = Image.open(path)
    cat_map = np.array(cat_map)
    cat_map = rgb2id(cat_map)
    ins_map = np.array(Image.open(imgfile_ins))

    none = cat_map == 255
    cat_map[none] = 0
    none2 = ins_map == 255
    ins_map[none] = 0

    mask_stuff = ins_map > 0

    cat_map[mask_stuff] = 0
    cat_map_stuff = cat_map // 1000
    mask_stuff2 = cat_map_stuff > 0
    ins_map[mask_stuff2] = cat_map_stuff[mask_stuff2]
    ins_map = ins_map*10000

    cat_map_ins = cat_map % 1000
    # cat_map_ins = cat_map // 1000

    mask = cat_map_ins > 0

    cat_map_ins[mask] = (cat_map_ins[mask] + 14)*10000
    # cat_map_ins[mask] = (cat_map_ins[mask]) * 10000

    mask2 = cat_map_ins > 0

    ins_map[mask2] = cat_map_ins[mask2]

    # mask3 = ins_map <= 140000
    # ins_map[mask3] = 0

    # mask3 = cat_map_ins2 == 1
    # ins_map[mask3] = 0

    ins_map = id2rgb(ins_map)

    filename = part
    Image.fromarray(ins_map).save(os.path.join(outroot, filename))