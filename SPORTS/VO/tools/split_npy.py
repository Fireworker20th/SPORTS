import os
import shutil

train = 'train'
val = 'val'
test = 'test'

print("===========\n split videos \n===========")

img_file = '/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/video/'
img_files = '/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/video/gt_depth'
train_folder = os.path.join(img_file, train)
val_folder = os.path.join(img_file, val)
test_folder = os.path.join(img_file, test)



for folder in [train_folder, val_folder, test_folder]:
    if not  os.path.exists(folder):
        os.makedirs(folder)

filelist = os.listdir(img_files)
filelist.sort()
for f in filelist:
    img_folder = os.path.join(img_files,f)
    part1 = f.rsplit("_")[0]
    part2 = f.rsplit(".")[0]
    part2 = part2.rsplit("_")[1]
    part3 = f.rsplit(".")[1]
    seq_id = part1[-2:]
    img_id = int(part2)
    if seq_id == "01":
        if img_id <= 320:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_" + part3
            target_file = os.path.join(target_folder,filename)
            shutil.copyfile(img_folder,target_file)
        elif img_id <= 383:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 321)
            filename = part1 + "_" + img_new_id + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 384)
            filename = part1 + "_" + img_new_id + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
    elif seq_id == "02":
        if img_id <= 166:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_" + part3
            target_file = os.path.join(target_folder,filename)
            shutil.copyfile(img_folder,target_file)
        elif img_id <= 199:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 167)
            filename = part1 + "_" + img_new_id + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 200)
            filename = part1 + "_" + img_new_id + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
    elif seq_id == "06":
        if img_id <= 193:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        elif img_id <= 231:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 194)
            filename = part1 + "_" + img_new_id + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 232)
            filename = part1 + "_" + img_new_id + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
    elif seq_id == "18":
        if img_id <= 242:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        elif img_id <= 290:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 243)
            filename = part1 + "_" + img_new_id + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 291)
            filename = part1 + "_" + img_new_id + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
    elif seq_id == "20":
        if img_id <= 598:
            target_folder = train_folder
            filename = part1 + "_" + part2 + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        elif img_id <= 717:
            target_folder = val_folder
            img_new_id = "{:06d}".format(img_id - 599)
            filename = part1 + "_" + img_new_id + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
        else:
            target_folder = test_folder
            img_new_id = "{:06d}".format(img_id - 718)
            filename = part1 + "_" + img_new_id + "_" + part3
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(img_folder, target_file)
print("videos split over")