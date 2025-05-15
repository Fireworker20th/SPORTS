import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy import stats

depth_paths = glob.glob(os.path.join('shared_data/depth/', '*.npy'))
output_dir = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/aug_depth/'
root = "shared_data/vknet_v1/"
os.makedirs(output_dir, exist_ok=True)

# 循环读取每个图片并将其转换为NumPy数组并保存为单独的.npy文件
for depth_path in depth_paths:

   # 读取图片
   depths = np.load(depth_path)
   file = depth_path.rsplit("/")[2]
   part1 = file.rsplit("_")[0]
   part2 = file.rsplit("_")[1]
   part2 = part2.split(".")
   part2 = part2[0]
   imgfile_cat = os.path.join(root, '00' + part1 + '_' + '0' + part2 + '_' + 'cat.png')
   imgfile_ins = os.path.join(root, '00' + part1 + '_' + '0' + part2 + '_' + 'ins.png')
   cat_map = np.array(Image.open(imgfile_cat))
   ins_map = np.array(Image.open(imgfile_ins))

   depths = cv2.resize(depths, (1242, 375))
   depths[depths <= 0.00001526] = 0.00001526

   mask1 = depths <= 0.00002
   depth_mask = depths
   depth_mask[mask1] = 0
   plus_map = cat_map + depth_mask
   plus_map = plus_map % 1
   mask3 = plus_map == 0
   depths[mask3] = 0.35

   depths[cat_map == 1] = 0.00001526
   '''
   unique_labels = np.unique(ins_map)
   unique_labels = unique_labels[unique_labels != 0]
   for i in unique_labels:
      mask_ins = ins_map == i
      region = depths[mask_ins]
      region1 = region[region >= 0.01]
      mean_value = np.mean(region1)
      std = np.std(region1)
      if std != 0:
         z_scores = (region - mean_value) / std
         threshold = 3
         outliers = np.abs(z_scores) > threshold
         suit = np.abs(z_scores) <= threshold
         mean = np.mean(region[suit])
         region[outliers] = mean
         region[region <= 0.01] = mean
      else:
         region[region <= 0.01] = mean_value
      depths[mask_ins] = region
   '''


   output_npy_file = os.path.join(output_dir, part1 + '_' + part2 + '.npy')
   np.save(output_npy_file, depths)


print("All images have been saved as .npy files.")