import numpy as np
import glob
import os
import os.path as osp

dic = { "Scene01":"0001",
        "Scene02":"0002",
        "Scene06":"0006",
        "Scene18":"0018",
        "Scene20":"0020",}

scenes = ['Scene18', 'Scene20' , 'Scene06', 'Scene02', 'Scene01']


for scene in scenes:
    # 指定图片文件夹路径
    txt_folder = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/traj/'

    # 获取所有图片文件的路径
    ex_paths = os.path.join(txt_folder, scene, 'clone/pose.txt')
    # 创建一个目录来存储.npy文件
    output_dir = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/pose_for_val'
    #output_dir = os.path.join(output_dir, scene)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    data = np.loadtxt(ex_paths)

    for i in range(data.shape[0]):
        file = os.path.join(output_dir, "00" + dic.get(scene) + "_" + f"{i:06}" + "_pose" + '.txt')
        file2 = os.path.join(output_dir, dic.get(scene) + "_" + f"{i:05}" + '.txt')
        np.savetxt(file2, data[i:i+1, :])

print("All txt have been saved as .txt files.")
