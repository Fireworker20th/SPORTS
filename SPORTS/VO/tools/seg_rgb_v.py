import numpy as np
from PIL import Image
import os
from panopticapi.utils import rgb2id, id2rgb, save_json, IdGenerator
from datasets.CATEGORY import categories
import glob

from datasets.CATEGORY import categories

categories_dict = {el['trainId']: el for el in categories}


image_paths = glob.glob(os.path.join("/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/test", '*.png'))
outroot = "/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/test/change"
if not os.path.exists(outroot):
    os.makedirs(outroot)

for path in image_paths:
    name = path.rsplit("/")[8]
    pan_map = np.array(Image.open(path))
    id_map = rgb2id(pan_map)
    mask12 = id_map > 11*10000
    id_map[mask12] = 0
    id_map = id_map/10000

    mask1 = id_map == 1
    pan_map[mask1] = categories_dict[1]['color']

    mask2 = id_map == 2
    pan_map[mask2] = categories_dict[2]['color']

    mask3 = id_map == 3
    pan_map[mask3] = categories_dict[3]['color']

    mask4 = id_map == 4
    pan_map[mask4] = categories_dict[4]['color']

    mask5 = id_map == 5
    pan_map[mask5] = categories_dict[5]['color']

    mask6 = id_map == 6
    pan_map[mask6] = categories_dict[6]['color']

    mask7 = id_map == 7
    pan_map[mask7] = categories_dict[7]['color']

    mask8 = id_map == 8
    pan_map[mask8] = categories_dict[8]['color']

    mask9 = id_map == 9
    pan_map[mask9] = categories_dict[9]['color']

    mask10 = id_map == 10
    pan_map[mask10] = categories_dict[10]['color']

    mask11 = id_map == 11
    pan_map[mask11] = categories_dict[11]['color']

    mask12 = id_map == 12
    pan_map[mask9] = categories_dict[9]['color']

    mask13 = id_map == 13
    pan_map[mask10] = categories_dict[10]['color']

    mask14 = id_map == 14
    pan_map[mask11] = categories_dict[11]['color']

    Image.fromarray(pan_map).save(os.path.join(outroot, name))
print("data change over")