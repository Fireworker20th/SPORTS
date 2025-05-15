import shutil
import os
import json
import csv
from PIL import Image
import numpy as np
from panopticapi.utils import rgb2id, id2rgb, save_json, IdGenerator
import pycocotools.mask as mask_util

root = "datasets/Virtual_KITTI2/"
scene_dir = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']

# ================================================
# gather camera-0 into img dir
# ================================================
print("===========\n gather img \n===========")
path_dir = ["clone/frames/forwardFlow/Camera_0" ,
            "15-deg-left/frames/forwardFlow/Camera_0", ]
target_dir = ['ALL_clone/forwardFlow/',
              'ALL_15-deg-left/forwardFlow/']
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


print("===========\n gather img \n===========")
path_dir = ["clone/frames/depth/Camera_0" ,
            "15-deg-left/frames/depth/Camera_0", ]
target_dir = ['ALL_clone/depth/',
              'ALL_15-deg-left/depth/']
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

print("gather over")


print("===========\n split flow into img \n===========")

img_files = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/Virtual_KITTI2/ALL_clone/forwardFlow/'
img_file = '/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/video/'
train_folder = os.path.join(img_file, "train")
val_folder = os.path.join(img_file, "val")
test_folder = os.path.join(img_file, "test")

filelist = os.listdir(img_files)
filelist.sort()
for f in filelist:
    img_folder = os.path.join(img_files,f)
    part1 = f.rsplit("_")[0]
    part2 = f.rsplit("_")[1]
    part2 = part2.split(".")
    part2 = part2[0]
    seq_id = part1[-2:]
    img_id = int(part2)
    part1 = "00" + part1
    part2 = "0" + part2
    if seq_id == "01":
        if img_id <= 320:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_flow.png"
            target_file = os.path.join(target_folder,filename)
            shutil.copyfile(img_folder,target_file)
        elif img_id <= 383:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 321)
            filename = part1 + "_" + img_new_id + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 384)
            filename = part1 + "_" + img_new_id + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
    elif seq_id == "02":
        if img_id <= 166:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_flow.png"
            target_file = os.path.join(target_folder,filename)
            shutil.copyfile(img_folder,target_file)
        elif img_id <= 199:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 167)
            filename = part1 + "_" + img_new_id + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 200)
            filename = part1 + "_" + img_new_id + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
    elif seq_id == "06":
        if img_id <= 193:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        elif img_id <= 231:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 194)
            filename = part1 + "_" + img_new_id + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 232)
            filename = part1 + "_" + img_new_id + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
    elif seq_id == "18":
        if img_id <= 242:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        elif img_id <= 290:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 243)
            filename = part1 + "_" + img_new_id + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 291)
            filename = part1 + "_" + img_new_id + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
    elif seq_id == "20":
        if img_id <= 598:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        elif img_id <= 717:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 599)
            filename = part1 + "_" + img_new_id + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 718)
            filename = part1 + "_" + img_new_id + "_flow.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
print("flow split over")




print("===========\n split depth into img \n===========")

img_files = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/Virtual_KITTI2/ALL_clone/depth/'
img_file = '/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/video/'
train_folder = os.path.join(img_file, "train")
val_folder = os.path.join(img_file, "val")
test_folder = os.path.join(img_file, "test")

filelist = os.listdir(img_files)
filelist.sort()
for f in filelist:
    img_folder = os.path.join(img_files,f)
    part1 = f.rsplit("_")[0]
    part2 = f.rsplit("_")[1]
    part2 = part2.split(".")
    part2 = part2[0]
    seq_id = part1[-2:]
    img_id = int(part2)
    part1 = "00" + part1
    part2 = "0" + part2
    if seq_id == "01":
        if img_id <= 320:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_depth.png"
            target_file = os.path.join(target_folder,filename)
            shutil.copyfile(img_folder,target_file)
        elif img_id <= 383:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 321)
            filename = part1 + "_" + img_new_id + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 384)
            filename = part1 + "_" + img_new_id + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
    elif seq_id == "02":
        if img_id <= 166:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_depth.png"
            target_file = os.path.join(target_folder,filename)
            shutil.copyfile(img_folder,target_file)
        elif img_id <= 199:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 167)
            filename = part1 + "_" + img_new_id + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 200)
            filename = part1 + "_" + img_new_id + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
    elif seq_id == "06":
        if img_id <= 193:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        elif img_id <= 231:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 194)
            filename = part1 + "_" + img_new_id + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 232)
            filename = part1 + "_" + img_new_id + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
    elif seq_id == "18":
        if img_id <= 242:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        elif img_id <= 290:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 243)
            filename = part1 + "_" + img_new_id + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 291)
            filename = part1 + "_" + img_new_id + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
    elif seq_id == "20":
        if img_id <= 598:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        elif img_id <= 717:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 599)
            filename = part1 + "_" + img_new_id + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 718)
            filename = part1 + "_" + img_new_id + "_depth.png"
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
print("depth split over")
