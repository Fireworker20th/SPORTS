import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from scipy.spatial.transform import Rotation as R
from lietorch import SE3
from torch.functional import split
from .base_kitti import RGBDDataset
from .stream import RGBDStream

from panopticapi.utils import rgb2id
import PIL.Image as Image


def rmat_to_quad(mat):
    r = R.from_matrix(mat)
    quat = r.as_quat()
    return quat

class Kitti(RGBDDataset):
    DEPTH_SCALE = 5.0
    # scale depths to balance rot & trans
    scenes = ['09', '10']
    print(scenes)

    def __init__(self, scene_id='01', **kwargs):
        self.n_frames = 2
        self.test_split = [x for x in Kitti.scenes if x != scene_id]
        super(Kitti, self).__init__(name='Kitti', **kwargs)

    # @staticmethod
    def is_test_scene(self, scene):
        return any(x in scene for x in self.test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building Kitti dataset")
        scenes = Kitti.scenes
        scene_info = {}
        for scene in tqdm(sorted(scenes)):
            posename = scene + '.txt'
            scene = osp.join(self.root, scene)
            images = sorted(
                glob.glob(osp.join(scene, 'image_2/*.png')))
            depths = sorted(
                glob.glob(osp.join(self.root, 'depth_vkitti2/*.png')))

            raw_mat = np.loadtxt(osp.join(scene, posename))
            values = np.array([[0, 0, 0, 1] for _ in range(raw_mat.shape[0])])
            raw_mat = np.concatenate((raw_mat, values), axis=1)
            poses = np.array(raw_mat)

            poses = poses.reshape(-1, 4, 4)
            r = rmat_to_quad(poses[:, 0:3, 0:3])
            t = poses[:, :3, 3] / Kitti.DEPTH_SCALE
            poses = np.concatenate((t, r), axis=1)

            intrinsics = [Kitti.calib_read()] * len(images)
            scene = '/'.join(scene.split('/'))

            scene_info[scene] = {'images': images, 'depths': depths, 'poses': poses, 'intrinsics': intrinsics, }

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([707.09, 707.09, 601.89, 183.11])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / (Kitti.DEPTH_SCALE * 100)
        depth[depth == np.nan] = 1.0
        depth[depth == np.inf] = 1.0
        depth[depth == 0] = 1.0
        return depth

    @staticmethod
    def flow_read(flow_file):
        bgr = cv2.imread(flow_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        h, w, _c = bgr.shape
        out_flow = 2.0 / (2 ** 16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
        out_flow[..., 0] *= w - 1
        out_flow[..., 1] *= h - 1
        val = (bgr[..., 0] > 0).astype(np.float32)
        return out_flow, val

    @staticmethod
    def dymask_read(mask_file):
        content = np.load(mask_file)
        return content[..., 0], content[..., 1]

    @staticmethod
    def segment_read(segment_file):
        segment = rgb2id(np.array(Image.open(segment_file)))
        return segment


class VKitti2Stream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(VKitti2Stream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/VKitti2'

        scene = osp.join(self.root, self.datapath)
        image_glob = osp.join(scene, 'image_left/*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)


class VKitti2TestStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(VKitti2TestStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/mono'
        image_glob = osp.join(self.root, self.datapath, '*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(self.root, 'mono_gt',
                                    self.datapath + '.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)
