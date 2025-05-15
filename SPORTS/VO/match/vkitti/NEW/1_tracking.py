import numpy as np
import cv2
import os
import PIL.Image as Image
from panopticapi.utils import id2rgb, rgb2id
import torch
import torch.nn.functional as F

flow_path = "/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/EMA_TRACK_UP/forwardFlow" # "shared_data/full_flow"
flow_names = os.listdir(flow_path)
flow_names.sort()
depth_path = '/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/depth'
depth_names = os.listdir(depth_path)
depth_names.sort()
segment_dir = "/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/vknet_for_augv2"
seg_list = os.listdir(segment_dir)
seg_list.sort()
data_root =  "/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/tmp"
output_dir = os.path.join(data_root, "vo_fusion_vo_track")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

def read_vkitti_png_flow(bgr, ori_shape):
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

def load_flow(flow_file):
    flow = cv2.imread(flow_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float64)
    o_h, o_w = 375, 1242
    t_h, t_w = 375, 1242
    pad_img = np.zeros((t_h, t_w, 3), dtype=np.float64)
    pad_img[:o_h, :o_w, :] = flow
    flow = torch.as_tensor(pad_img.copy()).permute(2, 0, 1)
    return flow

has_depth = False
cnt = 0
ref_segm = None
seq_id = None
flow_idx = 0
for idx in range(len(seg_list)):
    cnt = cnt + 1
    if has_depth:
        cur_depth_f = depth_names[idx]
    cur_mask_f = seg_list[idx]
    cur_seq_id = seg_list[idx][:4]

    print(cur_mask_f)
    segmentation = np.array(Image.open(os.path.join(segment_dir, cur_mask_f)))
    segmentation = rgb2id(segmentation)
    # depth = np.array(Image.open(os.path.join(depth_path, cur_depth_f))) / 100
    if has_depth:
        depth = np.array(Image.open(os.path.join(depth_path, cur_depth_f))) / 100

    if cur_seq_id != seq_id:
        seq_id = cur_seq_id
        ref_segm = segmentation
        if has_depth:
            ref_depth = depth

        ori_flow = load_flow(os.path.join(flow_path, flow_names[flow_idx]))
        ref_flow = ori_flow.squeeze(0).permute(1, 2, 0)

        ref_flow = read_vkitti_png_flow(ref_flow.detach().cpu().numpy().astype(np.uint16), ori_flow.shape)

        flow_idx = flow_idx + 1
        Image.fromarray(id2rgb(segmentation)).save(os.path.join(output_dir, cur_mask_f))
        continue

    rows, cols = ref_flow.shape[:2]
    mask = np.zeros_like(segmentation)
    dep = np.zeros_like(segmentation)

    v = np.arange(rows)
    v = v.reshape(rows, 1)
    v = np.repeat(v, cols, axis=1)
    u = np.arange(cols)
    u = np.tile(u, (rows, 1))

    u1 = (u + ref_flow[:,:,0]).astype(np.int32) # 1247
    v1 = (v + ref_flow[:,:,1]).astype(np.int32) # 374

    u = u.flatten()
    v = v.flatten()
    u1 = u1.flatten()
    v1 = v1.flatten()

    mm = (0 <= u1).__and__(u1 < cols).__and__(0 <= v1).__and__(v1 < rows)
    u1 = u1[mm]
    v1 = v1[mm]
    u = u[mm]
    v = v[mm]

    if has_depth:
        dep_uv = ref_depth.flatten()
        dep_uv = dep_uv[mm]
        encode_uvu1v1 = u * 1e14 + v * 1e10 + u1 * 1e6 + v1 * 1e2
        dic = dict(zip(encode_uvu1v1, dep_uv))
        ndic = np.array(sorted(dic.items(), key=lambda item:item[1], reverse=True))
        new_encode_uvu1v1 = ndic[:,0]

        u = (new_encode_uvu1v1 // 1e14).astype(np.int32)
        v = (new_encode_uvu1v1 % 1e14 // 1e10).astype(np.int32)
        u1 = (new_encode_uvu1v1 % 1e10 // 1e6).astype(np.int32)
        v1 = (new_encode_uvu1v1 % 1e6 // 1e2).astype(np.int32)

        ref_depth = depth

    mask[v1, u1] = ref_segm[v,u]

    ref_segm = segmentation
    Image.fromarray(id2rgb(mask)).save(os.path.join(output_dir, cur_mask_f))
    if idx < len(flow_names) and cur_seq_id == seg_list[idx + 1][:4]:
        ori_flow = load_flow(os.path.join(flow_path, flow_names[flow_idx]))
        ref_flow = ori_flow.squeeze(0).permute(1, 2, 0)
        ref_flow = read_vkitti_png_flow(ref_flow.detach().cpu().numpy().astype(np.uint16), ori_flow.shape)

        flow_idx = flow_idx + 1
print(cnt)

