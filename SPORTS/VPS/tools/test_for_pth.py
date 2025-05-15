import torch
import mmcv
img_file = "/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/video/train/000001_000000_depth.png"
img = mmcv.imread(img_file)
print(img)