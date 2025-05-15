from panopticapi.utils import id2rgb, rgb2id
import numpy as np
from PIL import Image
import os
import glob
import cv2



image_paths2 = "/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/Virtual_KITTI2/ALL_15-deg-left/stuff_TrainIds"

root1 = "/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/Virtual_KITTI2/Scene20/15-deg-left/frames/dynamicMask_00/Camera_0"
#root1 = "/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/Virtual_KITTI2/Scene20/clone/frames/dynamicMask_00/Camera_0"
image_paths1 = glob.glob(os.path.join(root1, '*.npy'))
outroot = "/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/nerf_mask/20"
if not os.path.exists(outroot):
    os.makedirs(outroot)

for path in image_paths1:
    part = path.rsplit("/")[14]
    part2 = part.rsplit(".")[0]
    part3 = part2.rsplit("_")[1]
    part4 = "0" + part3 + ".png"

    imgfile_ins = "0020_" + part3 + ".png"
    stuff_path = os.path.join(image_paths2, imgfile_ins)
    stuff_map = np.array(Image.open(stuff_path))
    mask = stuff_map == 2

    cat_map = np.load(path)

    cat_map = cat_map.astype(np.uint8)

    cat_map = cat_map[:, :, 0]*255

    #cat_map[mask] = 255


    rgb_image = np.tile(cat_map[:, :, np.newaxis], (1,1,3))

    filename = part4

    cv2.imwrite(os.path.join(outroot, filename), rgb_image[..., ::-1])
    #Image.fromarray(cat_map).save(os.path.join(outroot, filename))