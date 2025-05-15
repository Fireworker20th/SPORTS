import shutil
import os
import json
import csv
from PIL import Image
import numpy as np
from panopticapi.utils import rgb2id, id2rgb, save_json, IdGenerator
import pycocotools.mask as mask_util
from CATEGORY import categories
import glob

categories_dict = {el['trainId']: el for el in categories}
id_generator = IdGenerator(categories_dict)
seg2cat = {}
for idx in range(len(categories)):
    segment_id, color = id_generator.get_id_and_color(idx+1)
    seg2cat[segment_id] = idx + 1

root = "datasets/Virtual_KITTI2/"
scene_dir = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']

# ================================================
# gather camera-0 into img dir
# ================================================
print("===========\n gather img \n===========")
path_dir = ["clone/frames/rgb/Camera_0" ,
            "15-deg-left/frames/rgb/Camera_0", ]
target_dir = ['/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/imgs/',
              '/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_15-deg-left/imgs/']
folder_paths = '/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/img'
os.makedirs(folder_paths)
for img_path, target_path in zip(path_dir, target_dir):
    target_path = os.path.join(root, target_path)
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    for scene in scene_dir:
        path = os.path.join(scene, img_path)
        path = os.path.join(root, path)
        filelist = os.listdir(path)
        filelist.sort()
        seq_id = scene[-2:]
        cnt = 0
        for f in filelist:
            filename = "00" + seq_id + "_" + f.rsplit("_")[1]
            src = os.path.join(path, f)
            dst = os.path.join(target_path, filename)
            shutil.copyfile(src, dst)

# ================================================
# stuff_TrainIds, panoptic_gt_id
# ================================================
print("===========\n generate stuff train id and panoptic gt id \n===========")
class_path_dir = ["clone/frames/classSegmentation/Camera_0" ,
                  "15-deg-left/frames/classSegmentation/Camera_0", ]
inst_path_dir = ["clone/frames/instanceSegmentation/Camera_0",
                 "15-deg-left/frames/instanceSegmentation/Camera_0", ]
thing_dir = ["/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/labelTrainid",
             "/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_15-deg-left/labelTrainid", ]
pan_dir = ["/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/instanceTrainids",
           "/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_15-deg-left/instanceTrainids", ]
folder_path = '/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/labelTrainids'
os.makedirs(folder_path)

for (stuff_path, pan_path, cls_path, inst_path) in zip(thing_dir, pan_dir, class_path_dir, inst_path_dir):
    stuff_path = os.path.join(root, stuff_path)
    pan_path = os.path.join(root, pan_path)
    if not os.path.isdir(stuff_path):
        os.makedirs(stuff_path)
    if not os.path.isdir(pan_path):
        os.makedirs(pan_path)
    print(stuff_path)
    for scene in scene_dir:
        cur_cls_path = os.path.join(scene, cls_path)
        cur_cls_path = os.path.join(root, cur_cls_path)
        cls_filelist = os.listdir(cur_cls_path)
        cls_filelist.sort()

        cur_inst_path = os.path.join(scene, inst_path)
        cur_inst_path = os.path.join(root, cur_inst_path)
        inst_filelist = os.listdir(cur_inst_path)
        inst_filelist.sort()

        seq_id = scene[-2:]
        cnt = 0
        for (cls_f, inst_f) in zip(cls_filelist, inst_filelist):
            cat_img = rgb2id(np.array(Image.open(os.path.join(cur_cls_path, cls_f))))
            inst_img = np.array(Image.open(os.path.join(cur_inst_path, inst_f)))
            cat_list = np.unique(cat_img)

            cls_img = np.zeros_like(cat_img)
            for item in cat_list:
                mask = cat_img == item
                if item == 0 :
                    cls_img[mask] = 255
                else:
                    cls_img[mask] = seg2cat[item]
            inst_mask = (inst_img > 0) * (cls_img > 11)
            stuff_mask = cls_img > 11
            inst_map = cls_img * inst_mask
            inst_img = inst_img * inst_mask
            stuff_map = np.array(cls_img, copy=True)
            stuff_map[stuff_mask] = 0
            pan_map = stuff_map + inst_map * 1000 + inst_img
            mask = pan_map == 0
            pan_map[mask] = 255
            pan_map = pan_map - 1
            change = pan_map == 254
            pan_map[change] = 255
            thing_map = np.array(cls_img, copy=True)
            mask1 = thing_map ==0
            thing_map[mask1] = 255
            thing_map[mask] = 255
            thing_map = thing_map - 1
            change1 = thing_map == 254
            thing_map[change1] = 255

            filename = "00" + seq_id + "_" + cls_f.rsplit("_")[1]
            Image.fromarray(thing_map).save(os.path.join(stuff_path, filename))   # thing_labelTrainIds
            Image.fromarray(pan_map).save(os.path.join(pan_path, filename)) # panoptic_gt_id


