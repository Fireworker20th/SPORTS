import json
import os.path
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.cnn import (ConvModule, bias_init_with_prob,
                      build_activation_layer, build_norm_layer)
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import BaseDetector
from mmdet.models.builder import build_head, build_neck, build_backbone, build_roi_extractor
from mmdet.core import build_assigner, build_sampler
from knet.video.qdtrack.builder import build_tracker
from knet.det.utils import sem2ins_masks, sem2ins_masks_cityscapes, sem2ins_masks_kitti_step
from unitrack.mask import tensor_mask2box
from collections import OrderedDict
from PIL import Image
import cv2
import pickle
import mmcv

@DETECTORS.register_module()
class VideoKNetQuansiEmbedFCJointTrain(BaseDetector):
    """
        Simple Extension of KNet to Video KNet by the implementation of VPSFuse Net.
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 track_head=None,
                 extra_neck=None,
                 track_localization_fpn=None,
                 tracker=None,
                 train_cfg=None,
                 test_cfg=None,
                 track_train_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 detach_mask_emd=False,
                 cityscapes=False,
                 kitti_step=False,
                 cityscapes_short=False,
                 vipseg=False,
                 fusion=False,
                 freeze_detector=False,
                 semantic_filter=True,
                 # linking parameters
                 link_previous=False,
                 bbox_roi_extractor=None,
                 **kwargs):
        super(VideoKNetQuansiEmbedFCJointTrain, self).__init__(init_cfg)

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if extra_neck is not None:
            self.extra_neck = build_neck(extra_neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        if track_head is not None:
            self.track_train_cfg = track_train_cfg
            self.track_head = build_head(track_head)
            self.init_track_assigner_sampler()
            if track_localization_fpn is not None:
                self.track_localization_fpn = build_neck(track_localization_fpn)

            if bbox_roi_extractor is not None:
                self.track_roi_extractor = build_roi_extractor(
                    bbox_roi_extractor)

        if tracker is not None:
            self.tracker_cfg = tracker

        if freeze_detector:
           self._freeze_detector()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.device = torch.device("cuda:0")
        self.num_proposals = self.rpn_head.num_proposals
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        self.ignore_label = ignore_label
        self.cityscapes = cityscapes  # whether to train the cityscape panoptic segmentation
        self.kitti_step = kitti_step  # whether to train the kitti step panoptic segmentation
        self.cityscapes_short = cityscapes_short  # whether to test the cityscape short panoptic segmentation
        self.vipseg = vipseg  # whether to test the vip panoptic segmentation
        self.semantic_filter = semantic_filter
        self.link_previous = link_previous
        self.detach_mask_emd = detach_mask_emd

        self.city = False
        self.flow_is_npy = False  # ******************************* !!!
        self.read_fea_conv = False
        self.alpha = 1.0
        self.fusion = fusion
        if self.fusion:
            # 分割部分 freeze, 只训练 fusion 部分
            self._freeze_detector()

            self.pose_transport = False
            self.flow_transport = False
            self.flow_depth_transport = True  # ***************** !!!
            self.depth_proj_op = self.flow_depth_transport
            self.fusion_conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

            self.fusion_conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

            self.fusion_conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

            self.fusion_conv4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)


            if self.read_fea_conv:
                model_path = '/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/work_dirs/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train/epoch_2.pth'
                state_dict = torch.load(model_path)
                self.new_state_dict = OrderedDict()

                self.fusion_conv1.weight = torch.nn.Parameter(state_dict['state_dict']['fusion_conv1.weight'])
                self.fusion_conv1.bias = torch.nn.Parameter(state_dict['state_dict']['fusion_conv1.bias'])

                self.fusion_conv2.weight = torch.nn.Parameter(state_dict['state_dict']['fusion_conv2.weight'])
                self.fusion_conv2.bias = torch.nn.Parameter(state_dict['state_dict']['fusion_conv2.bias'])

                self.fusion_conv3.weight = torch.nn.Parameter(state_dict['state_dict']['fusion_conv3.weight'])
                self.fusion_conv3.bias = torch.nn.Parameter(state_dict['state_dict']['fusion_conv3.bias'])

                self.fusion_conv4.weight = torch.nn.Parameter(state_dict['state_dict']['fusion_conv4.weight'])
                self.fusion_conv4.bias = torch.nn.Parameter(state_dict['state_dict']['fusion_conv4.bias'])

            self.stage1 = False
            self.l1loss = nn.L1Loss()

            self.fx = 725.0087
            self.fy = 725.0087
            self.cx = 620.5
            self.cy = 187

            self.vid = None
            self.ref_flow = None
            self.ref_image = None
            self.ref_id = None


        num_emb_fcs = 1
        act_cfg = dict(type='ReLU', inplace=True)
        in_channels = 256
        out_channels = 256
        self.embed_fcs = nn.ModuleList()
        for _ in range(num_emb_fcs):
            self.embed_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.embed_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.embed_fcs.append(build_activation_layer(act_cfg))

        self.fc_embed = nn.Linear(in_channels, out_channels)
        for p3 in self.fc_embed.parameters():
            p3.requires_grad = False
        for m in self.embed_fcs:
            for p4 in m.parameters():
                p4.requires_grad = False

    def init_tracker(self):
        self.tracker = build_tracker(self.tracker_cfg)

    def _freeze_detector(self):

        self.detector = [
            self.backbone, self.neck, self.rpn_head, self.roi_head, self.track_head
        ]
        for model in self.detector:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False



    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""

        self.track_roi_assigner = build_assigner(
            self.track_train_cfg.assigner)
        self.track_share_assigner = False

        self.track_roi_sampler = build_sampler(
            self.track_train_cfg.sampler, context=self)
        self.track_share_sampler = False

    def preprocess_gt_masks(self, img_metas, gt_masks, gt_labels, gt_semantic_seg):
        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks_tensor = []
        gt_sem_seg = []
        gt_sem_cls = []
        # batch_input_shape shoud be the same across images
        pad_H, pad_W = img_metas[0]['batch_input_shape']
        assign_H = pad_H // self.mask_assign_stride
        assign_W = pad_W // self.mask_assign_stride

        for i, gt_mask in enumerate(gt_masks):
            mask_tensor = gt_mask.to_tensor(torch.float, gt_labels[0].device)
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)

            if gt_semantic_seg is not None:
                # gt_semantic seg is padded by zero when forming a batch
                # need to convert them from 0 to ignore
                gt_semantic_seg[
                i, :, img_metas[i]['img_shape'][0]:, :] = self.ignore_label
                gt_semantic_seg[
                i, :, :, img_metas[i]['img_shape'][1]:] = self.ignore_label

                if self.cityscapes or self.vipseg:
                    sem_labels, sem_seg = sem2ins_masks_cityscapes(
                        gt_semantic_seg[i],
                        ignore_label=self.ignore_label,
                        label_shift=self.num_thing_classes,
                        thing_label_in_seg=list(range(self.num_stuff_classes,
                                                      self.num_thing_classes + self.num_stuff_classes))
                    )
                elif self.kitti_step:
                    sem_labels, sem_seg = sem2ins_masks_kitti_step(
                        gt_semantic_seg[i],
                        ignore_label=self.ignore_label,
                        label_shift=3,
                        thing_label_in_seg=(11, 12, 13))
                else:
                    sem_labels, sem_seg = sem2ins_masks(
                        gt_semantic_seg[i],
                        ignore_label=self.ignore_label,
                        label_shift=self.num_thing_classes,
                        thing_label_in_seg=self.thing_label_in_seg)

                if sem_seg.shape[0] == 0:
                    gt_sem_seg.append(
                        mask_tensor.new_zeros(
                            (mask_tensor.size(0), assign_H, assign_W)))
                else:
                    gt_sem_seg.append(
                        F.interpolate(
                            sem_seg[None], (assign_H, assign_W),
                            mode='bilinear',
                            align_corners=False)[0])
                gt_sem_cls.append(sem_labels)
            else:
                gt_sem_seg = None
                gt_sem_cls = None

            if mask_tensor.shape[0] == 0:
                gt_masks_tensor.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
            else:
                gt_masks_tensor.append(
                    F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),  # downsample to 1/4 resolution
                        mode='bilinear',
                        align_corners=False)[0])

        return gt_masks_tensor, gt_sem_cls, gt_sem_seg

    def load_depth(self, depth_file):
        f = Image.open(depth_file)
        depth = np.array(f)
        o_h, o_w = 375, 1242
        t_h, t_w = 384, 1248
        pad_img = np.zeros((t_h, t_w), dtype=np.uint16)
        pad_img[:o_h, :o_w] = depth
        f.close()
        depth = torch.as_tensor(pad_img / 100)
        return depth

    def load_pose(self, pose_file):
        pose = np.loadtxt(pose_file)
        return pose

    def load_flow(self, flow_file):
        flow = cv2.imread(flow_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float64)
        o_h, o_w = 375, 1242
        t_h, t_w = 384, 1248
        pad_img = np.zeros((t_h, t_w, 3), dtype=np.float64)
        pad_img[:o_h, :o_w, :] = flow
        flow = torch.as_tensor(pad_img.copy()).permute(2, 0, 1)
        return flow

    def wrap_in_stage2(self, features, flow, depth, pose, model):
        if self.pose_transport:
            features = self.pose_transport_feature(features, depth, pose)
        elif self.flow_transport:
            features = self.flow_transport_feature(features, flow)
        elif self.flow_depth_transport:
            if self.depth_proj_op:
                depths = self.pose_transport_depth(depth, pose)
            features = self.flow_transport_feature_with_depth(features, flow, depths)
            # 组织好的 p2 1x512x96x312, p3 1x512x48x156, p4, p5
        features = self.fusion_module(features, model)
        return features

    def pose_transport_depth(self, ori_depth, pose):
        [rows, cols] = ori_depth.shape[-2:]
        v = np.repeat(np.arange(rows).reshape(rows, 1), cols, axis=1)
        u = np.tile(np.arange(cols), (rows, 1))

        Zc0 = ori_depth.detach().cpu().numpy()
        Xc0 = (u[:, :] - self.cx) / self.fx * Zc0[:, :]
        Yc0 = (v[:, :] - self.cy) / self.fy * Zc0[:, :]
        Ones = np.ones((rows, cols))
        point_camera_0 = np.array([Xc0, Yc0, Zc0, Ones]).reshape(4, -1)
        extrinsics_inv_0 = np.array(pose[0]).reshape((4, 4))
        extrinsics_0 = np.linalg.inv(extrinsics_inv_0)  # cam2wold
        extrinsics_inv_1 = np.array(pose[1]).reshape((4, 4))  # world2cam  # extrinsics_inv Twc: camera to world

        relative_pose = np.matmul(extrinsics_inv_1, extrinsics_0)
        point_camera_1 = np.matmul(relative_pose, point_camera_0)

        Xc1, Yc1, Zc1 = point_camera_1[:3, :]
        depth = torch.as_tensor(Zc1.reshape(rows, cols)).to(self.device)
        return depth

    def fusion_module(self, features, model):
        if model == 0:
            for k,v in features.items():
                x = self.fusion_conv1(v)
                # x = self.fusion_conv2(self.activation(x))
                features[k] = x
        elif model == 1:
            for k,v in features.items():
                x = self.fusion_conv2(v)
                # x = self.fusion_conv2(self.activation(x))
                features[k] = x
        elif model == 2:
            for k,v in features.items():
                x = self.fusion_conv3(v)
                # x = self.fusion_conv2(self.activation(x))
                features[k] = x
        elif model == 3:
            for k,v in features.items():
                x = self.fusion_conv4(v)
                # x = self.fusion_conv2(self.activation(x))
                features[k] = x
        return features

    def resize(self, mask, output_size):
        mask = F.interpolate(mask,
                             size=output_size,
                             mode="bilinear",
                             align_corners=True)
        return mask

    def read_vkitti_png_flow(self, bgr, ori_shape):
        # bgr = cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        _c, h, w = ori_shape
        assert bgr.dtype == np.uint16 and _c == 3
        # b == invalid flow flag == 0 for sky or other invalid flow
        invalid = bgr[:, :, 0] == 0
        # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 – 1]
        out_flow = 2.0 / (2 ** 16 - 1.0) * bgr[:, :, 2:0:-1].astype('f4') - 1
        out_flow[:, :, 0] *= w - 1
        out_flow[:, :, 1] *= h - 1
        out_flow[invalid] = 0  # or another value (e.g., np.nan)
        return out_flow

    def depth_filter(self, u1, v1, u, v, dep_uv):
        encode_uvu1v1 = u * 1e14 + v * 1e10 + u1 * 1e6 + v1 * 1e2
        dic = dict(zip(encode_uvu1v1, dep_uv))
        ndic = np.array(sorted(dic.items(), key=lambda item: item[1], reverse=True))
        # print(ndic.shape)
        if ndic.shape[0] != 0:
            new_encode_uvu1v1 = ndic[:, 0]
        else:
            new_encode_uvu1v1 = encode_uvu1v1

        u = (new_encode_uvu1v1 // 1e14).astype(np.int32)
        v = (new_encode_uvu1v1 % 1e14 // 1e10).astype(np.int32)
        u1 = (new_encode_uvu1v1 % 1e10 // 1e6).astype(np.int32)
        v1 = (new_encode_uvu1v1 % 1e6 // 1e2).astype(np.int32)

        return u1, v1, u, v

    def flow_transport_feature_with_depth(self, features, ori_flow, ori_depth):  # 3xHxW
        # Image.fromarray(np.uint8(ori_flow.permute(1,2,0).detach().cpu().numpy())).save("p0.png")
        for k, feat in features.items():
            # p2 2x256x96x312, p3 2x256x48x156, p4 2x256x24x78, p5 2x256x12x39, p6 2x256x6x20
            #     # 1/4               1/8               1/16             1/32            1/64

            ref_flow = self.resize(ori_flow.unsqueeze(0), feat.shape[-2:]).squeeze(0).permute(1, 2, 0)
            if self.flow_is_npy:
                ref_flow = ref_flow.detach().cpu().numpy().astype(np.uint16)
            else:
                ref_flow = self.read_vkitti_png_flow(ref_flow.detach().cpu().numpy().astype(np.uint16), ori_flow.shape)
            ref_depth = self.resize(ori_depth.unsqueeze(0).unsqueeze(0), feat.shape[-2:]).squeeze(0).squeeze(
                0).cpu().numpy()

            [rows, cols] = feat.shape[-2:]

            mask = torch.zeros(feat.shape[-3:]).to(self.device)  # 256x96x312
            v = np.arange(rows)
            v = v.reshape(rows, 1)
            v = np.repeat(v, cols, axis=1)
            u = np.arange(cols)
            u = np.tile(u, (rows, 1))

            u1 = (u + ref_flow[:, :, 0]).astype(np.int32)  # 1247
            v1 = (v + ref_flow[:, :, 1]).astype(np.int32)  # 374

            u = u.flatten()
            v = v.flatten()
            u1 = u1.flatten()
            v1 = v1.flatten()
            dep_uv = ref_depth.flatten()

            mm = (0 <= u1).__and__(u1 < cols).__and__(0 <= v1).__and__(v1 < rows)
            u1 = u1[mm]
            v1 = v1[mm]
            u = u[mm]
            v = v[mm]
            dep_uv = dep_uv[mm]

            u1, v1, u, v = self.depth_filter(u1, v1, u, v, dep_uv)

            mask[:, v1, u1] = feat[0][:, v, u]
            query_mask = torch.as_tensor(mask)  # 256xHxW
            features[k] = torch.cat((feat[1], self.alpha * query_mask), dim=0).unsqueeze(0)

        return features

    def flow_transport_feature(self, features, ori_flow):  # 3xHxW
        # Image.fromarray(np.uint8(ori_flow.permute(1,2,0).detach().cpu().numpy())).save("p0.png")
        for k, feat in features.items():
            # p2 2x256x96x312, p3 2x256x48x156, p4 2x256x24x78, p5 2x256x12x39, p6 2x256x6x20
            #     # 1/4               1/8               1/16             1/32            1/64

            ref_flow = self.resize(ori_flow.unsqueeze(0), feat.shape[-2:]).squeeze(0).permute(1, 2, 0)
            if self.flow_is_npy:
                ref_flow = ref_flow.detach().cpu().numpy().astype(np.uint16)
            else:
                ref_flow = self.read_vkitti_png_flow(ref_flow.detach().cpu().numpy().astype(np.uint16), ori_flow.shape)
            [rows, cols] = feat.shape[-2:]

            mask = torch.zeros(feat.shape[-3:]).to(self.device)  # 256x96x312
            v = np.arange(rows)
            v = v.reshape(rows, 1)
            v = np.repeat(v, cols, axis=1)
            u = np.arange(cols)
            u = np.tile(u, (rows, 1))

            u1 = (u + ref_flow[:, :, 0]).astype(np.int32)  # 1247
            v1 = (v + ref_flow[:, :, 1]).astype(np.int32)  # 374

            u = u.flatten()
            v = v.flatten()
            u1 = u1.flatten()
            v1 = v1.flatten()

            mm = (0 <= u1).__and__(u1 < cols).__and__(0 <= v1).__and__(v1 < rows)
            u1 = u1[mm]
            v1 = v1[mm]
            u = u[mm]
            v = v[mm]

            mask[:, v1, u1] = feat[0][:, v, u]
            query_mask = torch.as_tensor(mask)  # 256xHxW
            features[k] = torch.cat((feat[1], self.alpha * query_mask), dim=0).unsqueeze(0)

        return features

    def data_for_fusion(self, img_meta, ref_img_metas_new):
        what = ref_img_metas_new[0]['for_what']
        img_file = ref_img_metas_new[0]['ori_filename']
        img_file_cur = img_meta[0]['ori_filename']
        root = os.path.join(
            "/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/video/",
            what)
        depth_name = img_file.replace('leftImg8bit.jpg', 'depth.png')
        flow_name = img_file.replace('leftImg8bit.jpg', 'flow.png')
        pose_name = img_file.replace('leftImg8bit.jpg', 'pose.txt')
        pose_cur_name = img_file_cur.replace('leftImg8bit.jpg', 'pose.txt')
        depth_file = os.path.join(root, depth_name)
        flow_file = os.path.join(root, flow_name)
        pose_file = os.path.join(root, pose_name)
        pose_cur_file = os.path.join(root, pose_cur_name)
        pose_ref = self.load_pose(pose_file)
        pose_cur = self.load_pose(pose_cur_file)

        flow = self.load_flow(flow_file)
        depth = self.load_depth(depth_file).to(self.device)
        pose = [pose_ref, pose_cur]

        return flow, depth, pose

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      gt_instance_ids=None,
                      ref_img=None,
                      ref_img_metas=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_labels=None,
                      ref_gt_bboxes=None,
                      ref_gt_masks=None,
                      ref_gt_semantic_seg=None,
                      ref_gt_instance_ids=None,
                      proposals=None,
                      **kwargs):
        """Forward function of SparseR-CNN-like network in train stage.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. But we don't support it in this architecture.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

            # This is for video only:
            ref_img (Tensor): of shape (N, 2, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
                2 denotes there is two reference images for each input image.

            ref_img_metas (list[list[dict]]): The first list only has one
                element. The second list contains reference image information
                dict where each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_gt_bboxes (list[Tensor]): The list only has one Tensor. The
                Tensor contains ground truth bboxes for each reference image
                with shape (num_all_ref_gts, 5) in
                [ref_img_id, tl_x, tl_y, br_x, br_y] format. The ref_img_id
                start from 0, and denotes the id of reference image for each
                key image.

            ref_gt_labels (list[Tensor]): The list only has one Tensor. The
                Tensor contains class indices corresponding to each reference
                box with shape (num_all_ref_gts, 2) in
                [ref_img_id, class_indice].

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        assert proposals is None, 'KNet does not support' \
                                  ' external proposals'
        assert gt_masks is not None
        assert gt_instance_ids is not None
        # preprocess the reference images
        ref_img = ref_img.squeeze(1)  # (b,3,h,w)

        img_h, img_w = batch_input_shape
        ref_masks_gt = []
        for ref_gt_mask in ref_gt_masks:
            ref_masks_gt.append(ref_gt_mask[0])

        ref_labels_gt = []
        for ref_gt_label in ref_gt_labels:
            ref_labels_gt.append(ref_gt_label[:, 1].long())
        ref_gt_labels = ref_labels_gt

        ref_semantic_seg_gt = ref_gt_semantic_seg.squeeze(1)

        ref_gt_instance_id_list = []
        for ref_gt_instance_id in ref_gt_instance_ids:
            ref_gt_instance_id_list.append(ref_gt_instance_id[:,1].long())

        ref_img_metas_new = []
        for ref_img_meta in ref_img_metas:
            ref_img_meta[0]['batch_input_shape'] = batch_input_shape
            ref_img_metas_new.append(ref_img_meta[0])

        # prepare the gt_match_indices
        gt_pids_list = []
        for i in range(len(ref_gt_instance_id_list)):
            ref_ids = ref_gt_instance_id_list[i].cpu().data.numpy().tolist()
            gt_ids = gt_instance_ids[i].cpu().data.numpy().tolist()
            gt_pids = [ref_ids.index(i) if i in ref_ids else -1 for i in gt_ids]
            gt_pids_list.append(torch.LongTensor([gt_pids]).to(img.device)[0])

        gt_match_indices = gt_pids_list

        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks, gt_sem_cls, gt_sem_seg = self.preprocess_gt_masks(img_metas, gt_masks, gt_labels, gt_semantic_seg)
        ref_gt_masks, ref_gt_sem_cls, ref_gt_sem_seg = self.preprocess_gt_masks(ref_img_metas_new,
                                                                    ref_masks_gt, ref_gt_labels, ref_semantic_seg_gt)

        if self.fusion:
            x = self.extract_feat(img)
            x_ref = self.extract_feat(ref_img)
            flow, depth, pose = self.data_for_fusion(img_metas, ref_img_metas_new)

            for model in range(4):
                if model == 0:
                    x_fusion = {'p2': torch.cat((x_ref[0], x[0]), dim=0), 'p3': torch.cat((x_ref[1], x[1]), dim=0),
                                'p4': torch.cat((x_ref[2], x[2]), dim=0), 'p5': torch.cat((x_ref[3], x[3]), dim=0)}
                    x_feats_fusion1 = self.wrap_in_stage2(x_fusion, flow, depth, pose, model)
                    fusion_map1 = [0, 0, 0, 0]
                    fusion_map1[0] = x_feats_fusion1['p2']
                    fusion_map1[1] = x_feats_fusion1['p3']
                    fusion_map1[2] = x_feats_fusion1['p4']
                    fusion_map1[3] = x_feats_fusion1['p5']
                    x_fusion_map1 = fusion_map1
                    rpn_results = self.rpn_head.forward_train(x, x_fusion_map1, img_metas, gt_masks,
                                                              gt_labels, gt_sem_seg,
                                                              gt_sem_cls)
                    (rpn_losses, proposal_feats, x_feats, mask_preds,
                     cls_scores) = rpn_results
                elif model == 1:
                    x_fusion = {'p2': torch.cat((x_ref[0], x[0]), dim=0), 'p3': torch.cat((x_ref[1], x[1]), dim=0),
                                'p4': torch.cat((x_ref[2], x[2]), dim=0), 'p5': torch.cat((x_ref[3], x[3]), dim=0)}
                    x_feats_fusion2 = self.wrap_in_stage2(x_fusion, flow, depth, pose, model)
                    fusion_map2 = [0, 0, 0, 0]
                    fusion_map2[0] = x_feats_fusion2['p2']
                    fusion_map2[1] = x_feats_fusion2['p3']
                    fusion_map2[2] = x_feats_fusion2['p4']
                    fusion_map2[3] = x_feats_fusion2['p5']
                    x_fusion_map2 = fusion_map2
                    rpn_results1 = self.rpn_head.forward_train(x, x_fusion_map2, img_metas, gt_masks,
                                                              gt_labels, gt_sem_seg,
                                                              gt_sem_cls)
                    (rpn_losses1, proposal_feats1, x_feats1, mask_preds1,
                     cls_scores1) = rpn_results1
                elif model == 2:
                    x_fusion = {'p2': torch.cat((x_ref[0], x[0]), dim=0), 'p3': torch.cat((x_ref[1], x[1]), dim=0),
                                'p4': torch.cat((x_ref[2], x[2]), dim=0), 'p5': torch.cat((x_ref[3], x[3]), dim=0)}
                    x_feats_fusion3 = self.wrap_in_stage2(x_fusion, flow, depth, pose, model)
                    fusion_map3 = [0, 0, 0, 0]
                    fusion_map3[0] = x_feats_fusion3['p2']
                    fusion_map3[1] = x_feats_fusion3['p3']
                    fusion_map3[2] = x_feats_fusion3['p4']
                    fusion_map3[3] = x_feats_fusion3['p5']
                    x_fusion_map3 = fusion_map3
                    rpn_results2 = self.rpn_head.forward_train(x, x_fusion_map3, img_metas, gt_masks,
                                                              gt_labels, gt_sem_seg,
                                                              gt_sem_cls)
                    (rpn_losses2, proposal_feats2, x_feats2, mask_preds2,
                     cls_scores2) = rpn_results2
                elif model == 3:
                    x_fusion = {'p2': torch.cat((x_ref[0], x[0]), dim=0), 'p3': torch.cat((x_ref[1], x[1]), dim=0),
                                'p4': torch.cat((x_ref[2], x[2]), dim=0), 'p5': torch.cat((x_ref[3], x[3]), dim=0)}
                    x_feats_fusion4 = self.wrap_in_stage2(x_fusion, flow, depth, pose, model)
                    fusion_map4 = [0, 0, 0, 0]
                    fusion_map4[0] = x_feats_fusion4['p2']
                    fusion_map4[1] = x_feats_fusion4['p3']
                    fusion_map4[2] = x_feats_fusion4['p4']
                    fusion_map4[3] = x_feats_fusion4['p5']
                    x_fusion_map4 = fusion_map4
                    rpn_results3 = self.rpn_head.forward_train(x, x_fusion_map4, img_metas, gt_masks,
                                                              gt_labels, gt_sem_seg,
                                                              gt_sem_cls)
                    (rpn_losses3, proposal_feats3, x_feats3, mask_preds3,
                     cls_scores3) = rpn_results3

            ref_rpn_results = self.rpn_head.forward_train(x_ref, x_ref, ref_img_metas_new, ref_gt_masks,
                                                          ref_labels_gt, ref_gt_sem_seg,
                                                          ref_gt_sem_cls)
            (ref_rpn_losses, ref_proposal_feats, ref_x_feats, ref_mask_preds,
             ref_cls_scores) = ref_rpn_results

            losses_ref, ref_obj_feats, ref_cls_scores, ref_mask_preds, ref_scaled_mask_preds = self.roi_head.forward_train(
                ref_x_feats,
                ref_proposal_feats,
                ref_mask_preds,
                ref_cls_scores,
                ref_img_metas,
                ref_gt_masks,
                ref_gt_labels,
                gt_sem_seg=ref_gt_sem_seg,
                gt_sem_cls=ref_gt_sem_cls,
                imgs_whwh=None)


            if self.link_previous:
                losses, object_feats, cls_scores, mask_preds, scaled_mask_preds, object_feats_track = self.roi_head.forward_train_with_previous(
                    x_feats1,
                    x_feats2,
                    x_feats3,
                    proposal_feats,
                    mask_preds,
                    cls_scores,
                    img_metas,
                    gt_masks,
                    gt_labels,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    gt_bboxes=gt_bboxes,
                    gt_sem_seg=gt_sem_seg,
                    gt_sem_cls=gt_sem_cls,
                    imgs_whwh=None,
                    previous_obj_feats=ref_obj_feats,
                    previous_mask_preds=ref_scaled_mask_preds,
                    previous_x_feats=ref_x_feats,
                )
            else:
                # forward to get the current results
                losses, object_feats, cls_scores, mask_preds, scaled_mask_preds = self.roi_head.forward_train(
                    x_feats,
                    proposal_feats,
                    mask_preds,
                    cls_scores,
                    img_metas,
                    gt_masks,
                    gt_labels,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    gt_bboxes=gt_bboxes,
                    gt_sem_seg=gt_sem_seg,
                    gt_sem_cls=gt_sem_cls,
                    imgs_whwh=None)


            #self.object_feats_track_memory = object_feats_track

        # ===== Tracking Part -==== #
        # assign both key frame and reference frame tracking targets
        key_sampling_results, ref_sampling_results = [], []
        num_imgs = len(img_metas)

        for i in range(num_imgs):
            assign_result = self.track_roi_assigner.assign(
                scaled_mask_preds[i][:self.num_proposals].detach(), cls_scores[i][:self.num_proposals, :self.num_thing_classes].detach(),
                gt_masks[i], gt_labels[i], img_meta=img_metas[i])
            sampling_result = self.track_roi_sampler.sample(
                assign_result,
                mask_preds[i][:self.num_proposals].detach(),
                gt_masks[i])
            key_sampling_results.append(sampling_result)

            ref_assign_result = self.track_roi_assigner.assign(
                ref_scaled_mask_preds[i][:self.num_proposals].detach(), ref_cls_scores[i][:self.num_proposals, :self.num_thing_classes].detach(),
                ref_gt_masks[i], ref_gt_labels[i], img_meta=ref_img_metas_new[i])
            ref_sampling_result = self.track_roi_sampler.sample(
                ref_assign_result,
                ref_mask_preds[i][:self.num_proposals].detach(),
                ref_gt_masks[i])
            ref_sampling_results.append(ref_sampling_result)

        # current is tracking object
        N, num_proposal, _, _, _ = object_feats_track.shape
        emb_feat = object_feats_track.squeeze(-2).squeeze(-1)[:, :self.num_proposals, ]

        for emb_layer in self.embed_fcs:
            emb_feat = emb_layer(emb_feat)
        object_feats_embed = self.fc_embed(emb_feat).view(N, self.num_proposals, -1)


        ref_emb_feat = ref_obj_feats.squeeze(-2).squeeze(-1)[:, :self.num_proposals, ]
        for emb_layer in self.embed_fcs:
            ref_emb_feat = emb_layer(ref_emb_feat)
        ref_object_feats_embed = self.fc_embed(ref_emb_feat).view(N, self.num_proposals, -1)

        # sampling predicted GT mask
        key_emb_indexs = [res.pos_inds for res in key_sampling_results]
        object_feats_embed_list = []
        for i in range(len(key_emb_indexs)):
            object_feats_embed_list.append(object_feats_embed[:, key_emb_indexs[i], :].squeeze(0))

        key_feats = self._track_forward(object_feats_embed_list)

        ref_emb_indexs = [res.pos_inds for res in ref_sampling_results]
        ref_object_feats_embed_list = []
        for i in range(len(ref_emb_indexs)):
            ref_object_feats_embed_list.append(ref_object_feats_embed[:, ref_emb_indexs[i], :].squeeze(0))

        ref_feats = self._track_forward(ref_object_feats_embed_list)

        match_feats = self.track_head.match(key_feats, ref_feats,
                                            key_sampling_results,
                                            ref_sampling_results)

        asso_targets = self.track_head.get_track_targets(
            gt_match_indices, key_sampling_results, ref_sampling_results)
        loss_track = self.track_head.loss(*match_feats, *asso_targets)

        ref_losses = self.add_ref_loss(losses_ref)
        ref_rpn_losses = self.add_ref_rpn_loss(ref_rpn_losses)

        #losses.update(ref_rpn_losses)
        losses.update(rpn_losses)
        #losses.update(ref_losses)
        losses.update(loss_track)

        return losses

    def simple_test(self, img, img_metas, rescale=False, ref_img=None, **kwargs):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """

        # set the dataset type
        if self.cityscapes and not self.kitti_step and not self.cityscapes_short and not self.vipseg:
            iid = img_metas[0]['iid']
            fid = iid % 10000
            is_first = (fid == 1)
        else:
            iid = kwargs['img_id'][0].item()
            fid = iid % 10000
            is_first = (fid == 0)


        # init tracker
        if is_first:
            self.init_tracker()
            self.obj_feats_memory = None
            self.x_feats_memory = None
            self.x_fusion_memory = None
            self.x_fusion_memory1 = None
            self.x_fusion_memory2 = None
            self.x_fusion_memory3 = None
            self.mask_preds_memory = None
            self.image_metas_memory = None
            self.x_ref_memory = None
            print("fid", fid)

        if self.fusion:
            img_metas[0]['for_what'] = 'val'
            if is_first:
                # for current frame
                x = self.extract_feat(img)
                # current frame inference
                x_fusion_map = x
                x_fusion_map1 = x
                x_fusion_map2 = x
                x_fusion_map3 = x
                rpn_results = self.rpn_head.simple_test_rpn(x, x, img_metas)
                (proposal_feats, x_feats, mask_preds, cls_scores,
                 seg_preds) = rpn_results
                x_feats1 = x_feats
                x_feats2 = x_feats
                x_feats3 = x_feats
                x_feats_cur = x_feats
            else:
                flow, depth, pose = self.data_for_fusion(img_metas, self.image_metas_memory)
                x = self.extract_feat(img)
                rpn_results_cur = self.rpn_head.simple_test_rpn(x, x, img_metas)
                (proposal_feats_cur, x_feats_cur, mask_preds_cur, cls_scores_cur,
                 seg_preds_cur) = rpn_results_cur
                for model in range(4):
                    if model == 0:
                        x_ref = self.x_ref_memory
                        x_fusion = {'p2': torch.cat((x_ref[0], x[0]), dim=0), 'p3': torch.cat((x_ref[1], x[1]), dim=0),
                                    'p4': torch.cat((x_ref[2], x[2]), dim=0), 'p5': torch.cat((x_ref[3], x[3]), dim=0)}

                        x_feats_fusion = self.wrap_in_stage2(x_fusion, flow, depth, pose, model)
                        fusion_map = [0, 0, 0, 0]
                        fusion_map[0] = x_feats_fusion['p2']
                        fusion_map[1] = x_feats_fusion['p3']
                        fusion_map[2] = x_feats_fusion['p4']
                        fusion_map[3] = x_feats_fusion['p5']
                        x_fusion_map = fusion_map
                        rpn_results = self.rpn_head.simple_test_rpn(x, x_fusion_map, img_metas)
                        (proposal_feats, x_feats, mask_preds, cls_scores,
                         seg_preds) = rpn_results

                    elif model == 1:
                        x_ref = self.x_ref_memory
                        x_fusion = {'p2': torch.cat((x_ref[0], x[0]), dim=0), 'p3': torch.cat((x_ref[1], x[1]), dim=0),
                                    'p4': torch.cat((x_ref[2], x[2]), dim=0), 'p5': torch.cat((x_ref[3], x[3]), dim=0)}

                        x_feats_fusion = self.wrap_in_stage2(x_fusion, flow, depth, pose, model)
                        fusion_map = [0, 0, 0, 0]
                        fusion_map[0] = x_feats_fusion['p2']
                        fusion_map[1] = x_feats_fusion['p3']
                        fusion_map[2] = x_feats_fusion['p4']
                        fusion_map[3] = x_feats_fusion['p5']
                        x_fusion_map1 = fusion_map
                        rpn_results1 = self.rpn_head.simple_test_rpn(x, x_fusion_map1, img_metas)
                        (proposal_feats1, x_feats1, mask_preds1, cls_scores1,
                         seg_preds1) = rpn_results1

                    elif model == 2:
                        x_ref = self.x_ref_memory
                        x_fusion = {'p2': torch.cat((x_ref[0], x[0]), dim=0), 'p3': torch.cat((x_ref[1], x[1]), dim=0),
                                    'p4': torch.cat((x_ref[2], x[2]), dim=0), 'p5': torch.cat((x_ref[3], x[3]), dim=0)}

                        x_feats_fusion = self.wrap_in_stage2(x_fusion, flow, depth, pose, model)
                        fusion_map = [0, 0, 0, 0]
                        fusion_map[0] = x_feats_fusion['p2']
                        fusion_map[1] = x_feats_fusion['p3']
                        fusion_map[2] = x_feats_fusion['p4']
                        fusion_map[3] = x_feats_fusion['p5']
                        x_fusion_map2 = fusion_map
                        rpn_results2 = self.rpn_head.simple_test_rpn(x, x_fusion_map2, img_metas)
                        (proposal_feats2, x_feats2, mask_preds2, cls_scores2,
                         seg_preds2) = rpn_results2

                    elif model == 3:
                        x_ref = self.x_ref_memory
                        x_fusion = {'p2': torch.cat((x_ref[0], x[0]), dim=0), 'p3': torch.cat((x_ref[1], x[1]), dim=0),
                                    'p4': torch.cat((x_ref[2], x[2]), dim=0), 'p5': torch.cat((x_ref[3], x[3]), dim=0)}

                        x_feats_fusion = self.wrap_in_stage2(x_fusion, flow, depth, pose, model)
                        fusion_map = [0, 0, 0, 0]
                        fusion_map[0] = x_feats_fusion['p2']
                        fusion_map[1] = x_feats_fusion['p3']
                        fusion_map[2] = x_feats_fusion['p4']
                        fusion_map[3] = x_feats_fusion['p5']
                        x_fusion_map3 = fusion_map
                        rpn_results3 = self.rpn_head.simple_test_rpn(x, x_fusion_map3, img_metas)
                        (proposal_feats3, x_feats3, mask_preds3, cls_scores3,
                         seg_preds3) = rpn_results3

            if self.link_previous:
                cur_segm_results, obj_feats, cls_scores, mask_preds, scaled_mask_preds = self.roi_head.simple_test_with_previous(
                    x_feats1,
                    x_feats2,
                    x_feats3,
                    proposal_feats,
                    mask_preds,
                    cls_scores,
                    img_metas,
                    previous_obj_feats=self.obj_feats_memory,
                    previous_mask_preds=self.mask_preds_memory,
                    previous_x_feats=self.x_feats_memory,
                    is_first=is_first
                )
                self.obj_feats_memory = obj_feats
                self.x_feats_memory = x_feats_cur
                self.mask_preds_memory = scaled_mask_preds
                self.image_metas_memory = img_metas
                self.x_fusion_memory = x_fusion_map
                self.x_fusion_memory1 = x_fusion_map1
                self.x_fusion_memory2 = x_fusion_map2
                self.x_fusion_memory3 = x_fusion_map3
                self.x_ref_memory = x
            else:
                cur_segm_results, query_output, cls_scores, mask_preds, scaled_mask_preds = self.roi_head.simple_test(
                    x_feats,
                    proposal_feats,
                    mask_preds,
                    cls_scores,
                    img_metas)

    

        # for tracking part
        _, segm_result, mask_preds, panoptic_result, query_output = cur_segm_results[0]
        panoptic_seg, segments_info = panoptic_result

        # get sorted tracking thing ids, labels, masks, score for tracking
        things_index_for_tracking, things_labels_for_tracking, thing_masks_for_tracking, things_score_for_tracking = \
            self.get_things_id_for_tracking(panoptic_seg, segments_info)
        things_labels_for_tracking = torch.Tensor(things_labels_for_tracking).to(cls_scores.device).long()

        # get the semantic filter
        if self.semantic_filter:
            seg_preds = torch.nn.functional.interpolate(seg_preds, panoptic_seg.shape, mode='bilinear',
                                                        align_corners=False)
            seg_preds = seg_preds.sigmoid()
            seg_out = seg_preds.argmax(1)
            semantic_thing = (seg_out < self.num_thing_classes).to(dtype=torch.float32)
        else:
            semantic_thing = 1.

        if len(things_labels_for_tracking) > 0:
            things_bbox_for_tracking = torch.zeros((len(things_score_for_tracking), 5),
                                                   dtype=torch.float, device=x_feats.device)
            things_bbox_for_tracking[:, 4] = torch.tensor(things_score_for_tracking,
                                                          device=things_bbox_for_tracking.device)

            thing_masks_for_tracking_final = []
            for mask in thing_masks_for_tracking:
                thing_masks_for_tracking_final.append(torch.Tensor(mask).unsqueeze(0).to(
                    x_feats.device).float())
            thing_masks_for_tracking_final = torch.cat(thing_masks_for_tracking_final, 0)
            thing_masks_for_tracking = thing_masks_for_tracking_final
            thing_masks_for_tracking_with_semantic_filter = thing_masks_for_tracking_final * semantic_thing

        if len(things_labels_for_tracking) == 0:
            track_feats = None
        else:
            # tracking embeddings
            N, _, _, _ = query_output.shape
            emb_feat = query_output.squeeze(-2).squeeze(-1).unsqueeze(0)  # (n,d,1,1) -> (1,n,d)

            for emb_layer in self.embed_fcs:
                emb_feat = emb_layer(emb_feat)
            object_feats_embed = self.fc_embed(emb_feat).view(1, N, -1)
            object_feats_embed_for_tracking = object_feats_embed.squeeze(0)
            track_feats = self._track_forward([object_feats_embed_for_tracking])

        if track_feats is not None:
            things_bbox_for_tracking[:, :4] = torch.tensor(tensor_mask2box(thing_masks_for_tracking_with_semantic_filter),
                                                           device=things_bbox_for_tracking.device)
            bboxes, labels, ids = self.tracker.match(
                bboxes=things_bbox_for_tracking,
                labels=things_labels_for_tracking,
                track_feats=track_feats,
                frame_id=fid)

            ids = ids + 1
            ids[ids == -1] = 0

            # print("track feats:", track_feats[0])
            # print("id", ids)

        else:
            ids = []


        track_maps = self.generate_track_id_maps(ids, thing_masks_for_tracking, panoptic_seg)

        semantic_map = self.get_semantic_seg(panoptic_seg, segments_info)

        from scripts.visualizer import trackmap2rgb, cityscapes_cat2rgb, draw_bbox_on_img
        vis_tracker = trackmap2rgb(track_maps)
        vis_sem = cityscapes_cat2rgb(semantic_map)
        if len(things_labels_for_tracking):
            vis_tracker = draw_bbox_on_img(vis_tracker, things_bbox_for_tracking.cpu().numpy())

        # Visualization usage
        return semantic_map, track_maps, None, vis_sem, vis_tracker

    def _track_forward(self, track_feats, x=None, mask_pred=None):
        """Track head forward function used in both training and testing.
        We use mask pooling to get the fine grain features"""
        # if not self.training:
        #     mask_pred = [mask_pred]
        track_feats = torch.cat(track_feats, 0)

        track_feats = self.track_head(track_feats)

        return track_feats

    def forward_dummy(self, img, img_metas=None):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(0, 0, 3)) for _ in range(num_imgs)
        ]
        rpn_results = self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds) = rpn_results
        roi_outs = self.roi_head.simple_test_mask_preds(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,
            dummy_img_metas)
        return roi_outs

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        pass

    def get_things_id_for_tracking(self, panoptic_seg, seg_infos):
        idxs = []
        labels = []
        masks = []
        score = []
        for segment in seg_infos:
            if segment['isthing'] == True:
                thing_mask = panoptic_seg == segment["id"]
                masks.append(thing_mask)
                idxs.append(segment["instance_id"])
                labels.append(segment['category_id'])
                score.append(segment['score'])
        return idxs, labels, masks, score

    def pack_things_object(self, object_feats, ref_object_feats):
        object_feats, ref_object_feats = object_feats.squeeze(-1).squeeze(-1), ref_object_feats.squeeze(-1).squeeze(-1)
        thing_object_feats = torch.split(object_feats, [self.roi_head.num_proposals, self.num_stuff_classes], dim=1)[0]
        ref_thing_object_feats = torch.split(ref_object_feats, [self.roi_head.num_proposals, self.num_stuff_classes], dim=1)[0]
        return thing_object_feats, ref_thing_object_feats

    def pack_things_masks(self, mask_pred, ref_mask_pred):
        thing_mask_pred = torch.split(mask_pred, [self.roi_head.num_proposals, self.num_stuff_classes], dim=1)[0]
        ref_thing_thing_mask_pred = torch.split(ref_mask_pred, [self.roi_head.num_proposals, self.num_stuff_classes], dim=1)[0]
        return thing_mask_pred, ref_thing_thing_mask_pred

    def get_semantic_seg(self, panoptic_seg, segments_info):
        kitti_step2cityscpaes = [11, 12, 13]
        semantic_seg = np.zeros(panoptic_seg.shape)
        for segment in segments_info:
            if segment['isthing'] == True:
                # for things
                if self.kitti_step:
                    cat_cur = kitti_step2cityscpaes[segment["category_id"]]
                    semantic_seg[panoptic_seg == segment["id"]] = cat_cur
                else:   # city and vip_seg
                    semantic_seg[panoptic_seg == segment["id"]] = segment["category_id"] + self.num_stuff_classes
            else:
                # for stuff (0 - n-1)
                if self.kitti_step:
                    cat_cur = segment["category_id"]
                    cat_cur -= 1
                    offset = 0
                    for thing_id in kitti_step2cityscpaes:
                        if cat_cur + offset >= thing_id:
                            offset += 1
                    cat_cur += offset
                    semantic_seg[panoptic_seg == segment["id"]] = cat_cur
                else:   # city and vip_seg
                    semantic_seg[panoptic_seg == segment["id"]] = segment["category_id"] - 1
        return semantic_seg

    def generate_track_id_maps(self, ids, masks, panopitc_seg_maps):

        final_id_maps = np.zeros(panopitc_seg_maps.shape)

        if len(ids) == 0:
            return final_id_maps
        masks = masks.bool()

        for i, id in enumerate(ids):
            mask = masks[i].cpu().numpy()
            final_id_maps[mask] = id

        return final_id_maps

    def add_ref_loss(self, loss_dict):
        track_loss ={}
        for k, v in loss_dict.items():
            track_loss[str(k)+"_ref"] = v
        return track_loss

    def add_ref_rpn_loss(self, loss_dict):
        ref_rpn_loss = {}
        for k, v in loss_dict.items():
            ref_rpn_loss[str(k) +"_ref_rpn"] = v
        return ref_rpn_loss
