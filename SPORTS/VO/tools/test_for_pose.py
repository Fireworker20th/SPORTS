import numpy as np
import glob
import os

dic = { "Scene01":"0001",
        "Scene02":"0002",
        "Scene06":"0006",
        "Scene18":"0018",
        "Scene20":"0020",}

scenes = ['Scene18', 'Scene20' , 'Scene06', 'Scene02', 'Scene01']


for scene in scenes:
    # 指定图片文件夹路径
    txt_folder = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/Virtual_KITTI2/'

    # 获取所有图片文件的路径
    ex_paths = os.path.join(txt_folder, scene, 'extrinsic.txt')
    # 创建一个目录来存储.npy文件
    output_dir = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/datasets/Virtual_KITTI2/ALL_clone/ex/'
    output_dir1 = "/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/video/train/"
    output_dir2 = "/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/video/val/"
    output_dir3 = "/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/video/test/"

    data = np.loadtxt(ex_paths)
    ex = [0]
    z = data.shape
    b = z[0]
    z = int(z[0]/2)
    for i in range(0,z):
       for n in range(0,b):
          if data[n,0] == i:
            ex[0] = data[n-1,2:]
            #ex[1] = data[n+1,2:]
            output_ex_file = os.path.join(output_dir, dic.get(scene) + "_" + f"{i:05}" + '.txt')
            np.savetxt(output_ex_file,ex)

            if scene == "Scene01":
                if i <= 320:
                    target_folder = output_dir1
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{i:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
                elif i <= 383:
                    target_folder = output_dir2
                    a = i - 321
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{a:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
                else:
                    target_folder = output_dir3
                    a = i - 384
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{a:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
            elif scene == "Scene02":
                if i <= 166:
                    target_folder = output_dir1
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{i:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
                elif i <= 199:
                    target_folder = output_dir2
                    a = i - 167
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{a:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
                else:
                    target_folder = output_dir3
                    a = i - 200
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{a:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
            elif scene == "Scene06":
                if i <= 193:
                    target_folder = output_dir1
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{i:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
                elif i <= 231:
                    target_folder = output_dir2
                    a = i - 194
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{a:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
                else:
                    target_folder = output_dir3
                    a = i - 232
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{a:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
            elif scene == "Scene18":
                if i <= 242:
                    target_folder = output_dir1
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{i:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
                elif i <= 290:
                    target_folder = output_dir2
                    a = i - 243
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{a:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
                else:
                    target_folder = output_dir3
                    a = i - 291
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{a:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
            elif scene == "Scene20":
                if i <= 598:
                    target_folder = output_dir1
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{i:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
                elif i <= 717:
                    target_folder = output_dir2
                    a = i - 599
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{a:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)
                else:
                    target_folder = output_dir3
                    a = i - 718
                    output_ex = os.path.join(target_folder, "00" + dic.get(scene) + "_" + f"{a:06}" + "_pose" + '.txt')
                    np.savetxt(output_ex, ex)




print("All txt have been saved as .txt files.")
