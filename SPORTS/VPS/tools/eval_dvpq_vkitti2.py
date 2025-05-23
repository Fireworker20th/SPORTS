import numpy as np
from PIL import Image
import six
import os
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('result_path')
parser.add_argument('--eval_frames', type=int, default=1)
parser.add_argument('--depth_thres', type=float, default=0)
args = parser.parse_args()

eval_frames = args.eval_frames
pred_dir_all = os.path.join(args.result_path, 'panoptic')
depth_dir_all = os.path.join(args.result_path, 'depth')
gt_dir = '/media/jydai/C0FED904FED8F39E/download_jyd/project/Video-K-Net-main/data/vkitti2/ALL_clone/video/val/'
depth_thres = args.depth_thres


def vpq_eval(element):
    pred_ids, gt_ids = element
    max_ins = 2 ** 16
    ign_id = 255
    offset = 2 ** 30
    num_cat = 15

    iou_per_class = np.zeros(num_cat, dtype=np.float64)
    tp_per_class = np.zeros(num_cat, dtype=np.float64)
    fn_per_class = np.zeros(num_cat, dtype=np.float64)
    fp_per_class = np.zeros(num_cat, dtype=np.float64)

    def _ids_to_counts(id_array):
        ids, counts = np.unique(id_array, return_counts=True)
        return dict(six.moves.zip(ids, counts))

    pred_areas = _ids_to_counts(pred_ids)
    gt_areas = _ids_to_counts(gt_ids)

    void_id = ign_id * max_ins
    ign_ids = {
        gt_id for gt_id in six.iterkeys(gt_areas)
        if (gt_id // max_ins) == ign_id
    }

    int_ids = gt_ids.astype(np.int64) * offset + pred_ids.astype(np.int64)
    int_areas = _ids_to_counts(int_ids)

    def prediction_void_overlap(pred_id):
        void_int_id = void_id * offset + pred_id
        return int_areas.get(void_int_id, 0)

    def prediction_ignored_overlap(pred_id):
        total_ignored_overlap = 0
        for _ign_id in ign_ids:
            int_id = _ign_id * offset + pred_id
            total_ignored_overlap += int_areas.get(int_id, 0)
        return total_ignored_overlap

    gt_matched = set()
    pred_matched = set()

    for int_id, int_area in six.iteritems(int_areas):
        gt_id = int(int_id // offset)
        gt_cat = int(gt_id // max_ins)
        pred_id = int(int_id % offset)
        pred_cat = int(pred_id // max_ins)
        if gt_cat != pred_cat:
            continue
        union = (
            gt_areas[gt_id] + pred_areas[pred_id] - int_area -
            prediction_void_overlap(pred_id)
        )
        iou = int_area / union
        if iou > 0.5:
            tp_per_class[gt_cat] += 1
            iou_per_class[gt_cat] += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    for gt_id in six.iterkeys(gt_areas):
        if gt_id in gt_matched:
            continue
        cat_id = gt_id // max_ins
        if cat_id == ign_id:
            continue
        fn_per_class[cat_id] += 1

    for pred_id in six.iterkeys(pred_areas):
        if pred_id in pred_matched:
            continue
        if (prediction_ignored_overlap(pred_id) / pred_areas[pred_id]) > 0.5:
            continue
        cat = pred_id // max_ins
        fp_per_class[cat] += 1

    return (iou_per_class, tp_per_class, fn_per_class, fp_per_class)


def eval(element):
    max_ins = 2 ** 16

    pred_cat, pred_ins, gts, depth_preds, depth_gts = element
    pred_cat = [np.array(Image.open(image)) for image in pred_cat]
    pred_ins = [np.array(Image.open(image)) for image in pred_ins]
    pred_cat = np.concatenate(pred_cat, axis=1)
    pred_ins = np.concatenate(pred_ins, axis=1)
    pred = pred_cat.astype(np.int32) * max_ins + pred_ins.astype(np.int32)

    gts_pan = [np.array(Image.open(image)) for image in gts]
    gts = [gt_pan[..., 0].astype(np.int32) * max_ins +
           gt_pan[..., 1].astype(np.int32) * 256 + gt_pan[..., 2].astype(np.int32)
           for gt_pan in gts_pan]

    abs_rel = 0.
    if depth_thres > 0:
        depth_preds = [np.array(Image.open(name)) for name in depth_preds]
        depth_gts = [np.array(Image.open(name)) for name in depth_gts]
        depth_preds = np.concatenate(depth_preds, axis=1)
        depth_gts = np.concatenate(depth_gts, axis=1)
        depth_mask = depth_gts > 0
        abs_rel = np.mean(
            np.abs(
                depth_preds[depth_mask] -
                depth_gts[depth_mask]) /
            depth_gts[depth_mask])
        pred_in_mask = pred[:, :depth_preds.shape[1]]
        pred_in_depth_mask = pred_in_mask[depth_mask]
        ignored_pred_mask = (
            np.abs(
                depth_preds[depth_mask] -
                depth_gts[depth_mask]) /
            depth_gts[depth_mask]) > depth_thres
        pred_in_depth_mask[ignored_pred_mask] = 14 * max_ins
        pred_in_mask[depth_mask] = pred_in_depth_mask
        pred[:, :depth_preds.shape[1]] = pred_in_mask

    gt = np.concatenate(gts, axis=1)
    result = vpq_eval([pred, gt])

    return result + (abs_rel, )


def main():
    gt_names_all = os.scandir(gt_dir)
    gt_names_all = [name.name for name in gt_names_all if 'panoptic' in name.name]
    gt_names_all = [os.path.join(gt_dir, name) for name in gt_names_all]
    gt_names_all = sorted(gt_names_all)

    if args.depth_thres > 0:
        depth_gt_names_all = os.scandir(gt_dir)
        depth_gt_names_all = [
            name.name for name in depth_gt_names_all if 'depth' in name.name]
        depth_gt_names_all = [os.path.join(gt_dir, name) for name in depth_gt_names_all]
        depth_gt_names_all = sorted(depth_gt_names_all)

    iou_per_class_all = []
    tp_per_class_all = []
    fn_per_class_all = []
    fp_per_class_all = []

    things_index = np.zeros((14,)).astype(bool)
    things_index[11] = True
    things_index[12] = True
    things_index[13] = True

    for i in [1, 2, 6, 18, 20]:
        if args.depth_thres > 0:
            depth_dir = os.path.join(depth_dir_all, str(i))
            depth_pred_names = os.scandir(depth_dir)
            depth_pred_names = [name.name for name in depth_pred_names]
            depth_pred_names = [os.path.join(depth_dir, name)
                                for name in depth_pred_names]
            depth_pred_names = sorted(depth_pred_names)

        pred_dir = os.path.join(pred_dir_all, str(i))
        print(pred_dir)
        pred_names = os.scandir(pred_dir)
        pred_names = [os.path.join(pred_dir, name.name) for name in pred_names]
        cat_pred_names = [name for name in pred_names if name.endswith('cat.png')]
        ins_pred_names = [name for name in pred_names if name.endswith('ins.png')]
        cat_pred_names = sorted(cat_pred_names)
        ins_pred_names = sorted(ins_pred_names)

        all_lst = []
        gt_names = sorted(list(filter(lambda x: os.path.basename(x).startswith('{:06d}'.format(i)), gt_names_all)))
        if args.depth_thres > 0:
            depth_gt_names = sorted(list(filter(lambda x: os.path.basename(x).startswith('{:06d}'.format(i)), depth_gt_names_all)))
        for i in range(len(cat_pred_names) - eval_frames + 1):
            all_lst.append([cat_pred_names[i: i + eval_frames],
                            ins_pred_names[i: i + eval_frames],
                            gt_names[i: i + eval_frames],
                            depth_pred_names[i: i + eval_frames] if args.depth_thres > 0 else None,
                            depth_gt_names[i: i + eval_frames] if args.depth_thres > 0 else None
                            ])

        N = mp.cpu_count() // 2
        with mp.Pool(processes=N) as p:
            results = p.map(eval, all_lst)
        iou_per_class = np.stack([result[0] for result in results])
        iou_per_class_all.append(iou_per_class)
        tp_per_class = np.stack([result[1] for result in results])
        tp_per_class_all.append(tp_per_class)
        fn_per_class = np.stack([result[2] for result in results])
        fn_per_class_all.append(fn_per_class)
        fp_per_class = np.stack([result[3] for result in results])
        fp_per_class_all.append(fp_per_class)
        # abs_rel = np.stack([result[4] for result in results]).mean(axis=0)
        epsilon = 1e-10
        iou_per_class = iou_per_class.sum(axis=0)[:14]
        tp_per_class = tp_per_class.sum(axis=0)[:14]
        fn_per_class = fn_per_class.sum(axis=0)[:14]
        fp_per_class = fp_per_class.sum(axis=0)[:14]
        sq = iou_per_class / (tp_per_class + epsilon)
        rq = tp_per_class / (tp_per_class + 0.5 *
                             fn_per_class + 0.5 * fp_per_class + epsilon)
        pq = sq * rq
        spq = pq[np.logical_not(things_index)]
        tpq = pq[things_index]
        print(
            r'{:.1f} {:.1f} {:.1f}'.format(
                pq.mean() * 100,
                tpq.mean() * 100,
                spq.mean() * 100))

    print("----------------final-----------------")
    iou_per_class_all = np.concatenate(iou_per_class_all, axis=0).sum(axis=0)[:14]
    tp_per_class_all = np.concatenate(tp_per_class_all, axis=0).sum(axis=0)[:14]
    fn_per_class_all = np.concatenate(fn_per_class_all, axis=0).sum(axis=0)[:14]
    fp_per_class_all = np.concatenate(fp_per_class_all, axis=0).sum(axis=0)[:14]

    sq = iou_per_class_all / (tp_per_class_all + epsilon)
    rq = tp_per_class_all / (tp_per_class_all + 0.5 *
                         fn_per_class_all + 0.5 * fp_per_class_all + epsilon)
    pq = sq * rq
    spq = pq[np.logical_not(things_index)]
    tpq = pq[things_index]
    print(
        r'{:.1f} {:.1f} {:.1f}'.format(
            pq.mean() * 100,
            tpq.mean() * 100,
            spq.mean() * 100))


if __name__ == '__main__':
    main()
