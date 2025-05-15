import os
from PIL import Image
import glob

image_paths1 = glob.glob(os.path.join('/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/labelTrainid', '*.png'))
output_dir1 ='/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/labelTrainids'

for image_path in image_paths1:
    part = os.path.basename(image_path).split('.')[0]
    color_image = Image.open(image_path)
    gray_image = color_image.convert('L')
    output = os.path.join(output_dir1, part + '.png')
    gray_image.save(output)
    color_image.close()
    gray_image.close()
print("change over1")

image_paths = glob.glob(os.path.join('/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_15-deg-left/labelTrainid', '*.png'))
output_dir ='/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_15-deg-left/labelTrainids'

for image_path in image_paths1:
    part = os.path.basename(image_path).split('.')[0]
    color_image = Image.open(image_path)
    gray_image = color_image.convert('L')
    output = os.path.join(output_dir1, part + '.png')
    gray_image.save(output)
    color_image.close()
    gray_image.close()
print("change over2")