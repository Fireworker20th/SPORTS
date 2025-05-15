####################################################
import glob
import cv2
import numpy as np
import os

def flow_read(flow_file):
    bgr = cv2.imread(flow_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    out_flow = 2.0 / (2 ** 16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
    out_flow[..., 0] *= w - 1
    out_flow[..., 1] *= h - 1
    val = (bgr[..., 0] > 0).astype(np.float32)
    return out_flow#, val

dic = { "Scene01":"0001",
        "Scene02":"0002",
        "Scene06":"0006",
        "Scene18":"0018",
        "Scene20":"0020",}

scenes = ['Scene18', 'Scene20' , 'Scene06', 'Scene02', 'Scene01']

for scene in scenes:

    image_folder = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/Virtual_KITTI2/'

    # 获取所有图片文件的路径
    image_paths = glob.glob(os.path.join(image_folder, scene,'clone/frames/forwardFlow/Camera_0/', '*.png'))
    print(os.path.join(image_folder, scene,'frames/forwardFlow/Camera_0/', '*.png'))
    # 创建一个目录来存储.npy文件
    output_dir = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/gt_flow/'
    os.makedirs(output_dir, exist_ok=True)

    # 循环读取每个图片并将其转换为NumPy数组并保存为单独的.npy文件
    for image_path in image_paths:
       # 读取图片
       flow = flow_read(image_path)
       flow_data = np.transpose(flow, (1, 0, 2))
       flow_data = np.array(flow_data)

       if flow_data is not None:
           # 构建输出.npy文件的路径
           part = os.path.basename(image_path).split('.')[0]
           part = part[4:]
           output_npy_file = os.path.join(output_dir, dic.get(scene) + part + '.npy')

           # 将图片数据保存为.npy文件
           np.save(output_npy_file, flow_data)
           print(f"Image saved as {output_npy_file}")
       else:
           print(f"Failed to read image: {image_path}")

print("All images have been saved as .npy files.")


