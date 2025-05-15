import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import cv2

npy_folder = '/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/viper/video/aug_depth'
npy_paths = os.path.join(npy_folder, '0001_00001.npy')
output_dir = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/depth_png/'
os.makedirs(output_dir, exist_ok=True)
print(npy_paths)
#for npy_path in npy_paths:
    # 读取图片
depths = np.load(npy_paths)
depths = cv2.resize(depths, (1920, 1080))
print(depths.shape)
#depths = np.stack(depths).astype(np.float32)

disps = np.array(1.0 / depths)
disps = disps*500

scaled_data = ((disps - np.min(disps)) / (np.max(disps) - np.min(disps)))*255
print(disps)
print(depths.min())

plt.imshow(depths, cmap='gray')
plt.show()