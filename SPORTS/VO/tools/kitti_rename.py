import os
import glob
import numpy as np
import os.path as osp

'''
root = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/kitti/10/image2_forseg'
root2 = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/kitti/10/image_2_f'
image_paths1 = glob.glob(os.path.join('/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/kitti/10/image2_forseg', '*.png'))
image_paths2 = glob.glob(os.path.join('/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/kitti/10/image_2_f', '*.png'))

for path in image_paths1:
    part1 = path.rsplit("/")[11]
    part2 = part1.rsplit(".")[0]
    id = int(part2)
    if id <= 446:
        img_new_id = "{:05d}".format(id)
        new_name = '0001_' + img_new_id + '.png'
    if id > 446 and id <= 679:
        img_new_id = "{:05d}".format(id - 447)
        new_name = '0002_' + img_new_id + '.png'
    if id > 679 and id <= 949:
        img_new_id = "{:05d}".format(id - 680)
        new_name = '0006_' + img_new_id + '.png'
    if id > 949 and id <= 1288:
        img_new_id = "{:05d}".format(id - 950)
        new_name = '0018_' + img_new_id + '.png'
    if id > 1288:
        img_new_id = "{:05d}".format(id - 1289)
        new_name = '0020_' + img_new_id + '.png'
    old = os.path.join(root, part1)
    new = os.path.join(root, new_name)
    print(new)

    os.rename(old,new)
    
for path in image_paths2:
    part1 = path.rsplit("/")[11]
    part2 = part1.rsplit(".")[0]
    id = int(part2)
    if id < 88:
        img_new_id = "{:05d}".format(id + 251)
        new_name = '0018_' + img_new_id + '.png'
    else:
        img_new_id = "{:05d}".format(id - 88)
        new_name = '0020_' + img_new_id + '.png'
    
    old = os.path.join(root2, part1)
    new = os.path.join(root2, new_name)
    print(new)

    os.rename(old,new)
   
root = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/kitti/10/depth'
root2 = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/kitti/09/image_2_f'
image_paths1 = glob.glob(os.path.join('/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/kitti/10/depth', '*.npy'))
image_paths2 = glob.glob(os.path.join('/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/kitti/09/image_2_f', '*.png'))

for path in image_paths1:
    part1 = path.rsplit("/")[11]
    part2 = part1.rsplit(".")[0]
    part3 = part2.rsplit("_")[1]
    id = int(part3)
    if id <= 446:
        img_new_id = "{:05d}".format(id)
        new_name = '0001_' + img_new_id + '.npy'
    if id > 446 and id <= 679:
        img_new_id = "{:05d}".format(id - 447)
        new_name = '0002_' + img_new_id + '.npy'
    if id > 679 and id <= 949:
        img_new_id = "{:05d}".format(id - 680)
        new_name = '0006_' + img_new_id + '.npy'
    if id > 949 and id <= 1288:
        img_new_id = "{:05d}".format(id - 950)
        new_name = '0018_' + img_new_id + '.npy'
    if id > 1288:
        img_new_id = "{:05d}".format(id - 1289)
        new_name = '0020_' + img_new_id + '.npy'
    old = os.path.join(root, part1)
    new = os.path.join(root, new_name)
    print(new)

    os.rename(old,new)
    


# 指定图片文件夹路径
txt_folder = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/traj/'

# 获取所有图片文件的路径
ex_paths = os.path.join(txt_folder, '10/kitti10_pansegv1/pose.txt')
# 创建一个目录来存储.npy文件
output_dir = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/pose_for_val_kitti10'
#output_dir = os.path.join(output_dir, scene)
if not osp.exists(output_dir):
    os.makedirs(output_dir)

data = np.loadtxt(ex_paths)

for id in range(data.shape[0]):
    if id <= 446:
        img_new_id = "{:05d}".format(id)
        new_name = '0001_' + img_new_id + '.txt'
    if id > 446 and id <= 679:
        img_new_id = "{:05d}".format(id - 447)
        new_name = '0002_' + img_new_id + '.txt'
    if id > 679 and id <= 949:
        img_new_id = "{:05d}".format(id - 680)
        new_name = '0006_' + img_new_id + '.txt'
    if id > 949 and id <= 1288:
        img_new_id = "{:05d}".format(id - 950)
        new_name = '0018_' + img_new_id + '.txt'
    if id > 1288:
        img_new_id = "{:05d}".format(id - 1289)
        new_name = '0020_' + img_new_id + '.txt'

    file = os.path.join(output_dir, new_name)
    np.savetxt(file, data[id:id+1, :])
'''

# 指定图片文件夹路径
txt_folder = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/traj/'

# 获取所有图片文件的路径
ex_paths = os.path.join(txt_folder, '09/kitti09_kv1/pose.txt')
# 创建一个目录来存储.npy文件
output_dir = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/pose_for_val_kitti09'
#output_dir = os.path.join(output_dir, scene)
if not osp.exists(output_dir):
    os.makedirs(output_dir)

data = np.loadtxt(ex_paths)

for id in range(data.shape[0]):
        img_new_id = "{:06d}".format(id+130)
        new_name = '0009_' + img_new_id + '.txt'

        file = os.path.join(output_dir, new_name)
        np.savetxt(file, data[id:id+1, :])

print("All txt have been saved as .txt files.")