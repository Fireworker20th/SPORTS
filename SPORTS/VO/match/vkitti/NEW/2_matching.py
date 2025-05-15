import numpy as np
import cv2
import os
import PIL.Image as Image
from panopticapi.utils import rgb2id, id2rgb
import six
import torch

offset = 2 ** 30
max_ins = 10000  # cat * 10000 + inst_id


def _filter_thing(ps_map):
    cat_mask = ps_map // max_ins
    mask = cat_mask > 14  # 去掉 stuff (id_generator)
    ps_map[mask] = 0
    mask = cat_mask == 0
    ps_map[mask] = 0
    return ps_map


def _ids_to_counts(id_array):
    ids, counts = np.unique(id_array, return_counts=True)
    return dict(six.moves.zip(ids, counts))


segment_dir = "/media/jydai/C0FED904FED8F39E/download_jyd/project/PVO-main/shared_data/vknet_for_augv2"
seg_list = os.listdir(segment_dir)
seg_list.sort()

data_root = "shared_data/tmp"
pred_dir = os.path.join(data_root, "vo_fusion_vo_track")
pred_list = os.listdir(pred_dir)
pred_list.sort()

output_dir = os.path.join(data_root, "vo_fusion_vo_match")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

ref_match = None
seq_id = None
for (seg_f, pred_f) in zip(seg_list, pred_list):
    print(seg_f, pred_f)
    cur_seq_id = seg_f[:4]
    next_mask = rgb2id(np.array(Image.open(os.path.join(segment_dir, seg_f))))
    pred_mask = rgb2id(np.array(Image.open(os.path.join(pred_dir, pred_f))))
    pan_res = np.array(next_mask, copy=True)

    next_map = _filter_thing(next_mask)
    pred_map = _filter_thing(pred_mask)

    rows_list = np.unique(next_map)
    cols_list = np.unique(pred_map)
    rows_id = {v: k for k, v in enumerate(rows_list)}
    cols_id = {v: k for k, v in enumerate(cols_list)}
    rows = len(rows_list)
    cols = len(cols_list)

    if cur_seq_id != seq_id:
        ref_match = None
        seq_id = cur_seq_id

    if ref_match is None:
        ref_match = {}
        for item in np.unique(next_map):
            if item == 0:
                continue
            ref_match[item] = item
        continue

    gt_areas = _ids_to_counts(next_map)
    pred_areas = _ids_to_counts(pred_map)

    int_ids = next_map.astype(np.int64) * offset + pred_map.astype(np.int64)
    int_areas = _ids_to_counts(int_ids)

    match_score = np.zeros((rows, cols))

    for int_id, int_area in six.iteritems(int_areas):
        gt_id = int(int_id // offset)
        gt_cat = int(gt_id // max_ins)
        pred_id = int(int_id % offset)
        pred_cat = int(pred_id // max_ins)
        # 确保两个id // 10000 之后的类别相同
        if gt_cat != pred_cat or gt_id == 0:
            continue

        iou = int_area / pred_areas[pred_id]
        match_score[rows_id[gt_id]][cols_id[pred_id]] = iou

    match_score = torch.tensor(match_score)
    match_likelihood_embed, match_ids = torch.max(match_score, dim=1)
    match_ids = match_ids.cpu().numpy().astype(np.int32).tolist()

    for idx, match_id in enumerate(match_ids):
        ori_id = rows_list[idx]
        mask = next_mask == ori_id
        if ori_id == 0 or match_ids == 0:
            continue

        # 更新 next_mask 的 ID 为 pred_mask 对应的 ID
        if cols_list[match_id] != ori_id:
            new_id = cols_list[match_id]
            pan_res[mask] = new_id
        else:
            pan_res[mask] = ori_id

    Image.fromarray(id2rgb(pan_res)).save(os.path.join(output_dir, pred_f))