import glob
import cv2
import numpy as np
import os

def depth_read(depth_file):
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |
                       cv2.IMREAD_ANYDEPTH) / (5 *100)
    depth[depth == np.nan] = 1.0
    depth[depth == np.inf] = 1.0
    depth[depth == 0] = 1.0
    return depth


dic = { "Scene01":"0001",
        "Scene02":"0002",
        "Scene06":"0006",
        "Scene18":"0018",
        "Scene20":"0020",}

scenes = ['Scene18', 'Scene20' , 'Scene06', 'Scene02', 'Scene01']

for scene in scenes:
    # 指定图片文件夹路径
    image_folder = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/Virtual_KITTI2/'

    # 获取所有图片文件的路径
    image_paths = glob.glob(os.path.join(image_folder, scene,'clone/frames/depth/Camera_0/', '*.png'))
    print(os.path.join(image_folder, scene, 'frames/forwardFlow/Camera_0/', '*.png'))
    # 创建一个目录来存储.npy文件
    output_dir = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/gt_depth/'
    os.makedirs(output_dir, exist_ok=True)

    # 循环读取每个图片并将其转换为NumPy数组并保存为单独的.npy文件
    for image_path in image_paths:
        # 读取图片
        depths = depth_read(image_path)
        depths = np.stack(depths).astype(np.float32)
        disps = np.array(1.0 / depths)

        if len(disps[disps > 0.01]) > 0:
            s = disps[disps > 0.01].mean()
            disps = disps / s

        d_data = np.array(disps)
        '''
        # 读取深度数据
        depth_data = d_data

        # 定义目标图像尺寸
        target_height, target_width = 376, 1248

        # 使用双线性插值调整图像尺寸
        depth_data = cv2.resize(depth_data, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        '''

        if d_data is not None:
            # 构建输出.npy文件的路径
            part=os.path.basename(image_path).split('.')[0]
            part=part[6:]
            output_npy_file = os.path.join(output_dir, '00' + dic.get(scene) + '_0' + part + '.npy')

            # 将图片数据保存为.npy文件
            np.save(output_npy_file, d_data)
            print(f"Image saved as {output_npy_file}")
        else:
            print(f"Failed to read image: {image_path}")

print("All images have been saved as .npy files.")



