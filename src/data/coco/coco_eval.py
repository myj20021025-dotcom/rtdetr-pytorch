# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from src.misc import dist
from src.zoo.rtdetr.box_ops import box_iou


__all__ = ['CocoEvaluator', 'compute_ultralytics_prf1', 'process_ultralytics_batch']


def smooth(y, f=0.05):
    """Ultralytics-style box filter smoothing."""
    if len(y) == 0:
        return y

    nf = round(len(y) * f * 2) // 2 + 1
    if nf <= 1:
        return y

    p = np.ones(nf // 2, dtype=np.float64)
    yp = np.concatenate((p * y[0], y, p * y[-1]), axis=0)
    return np.convolve(yp, np.ones(nf, dtype=np.float64) / nf, mode='valid')


def compute_ap(recall, precision):
    """Ultralytics-style AP helper used to build the PR curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def process_ultralytics_batch(detections, labels, iouv):
    """Match detections to labels with the same greedy logic Ultralytics uses for val stats."""
    correct = torch.zeros((detections.shape[0], iouv.numel()), dtype=torch.bool, device=iouv.device)
    if labels.shape[0] == 0 or detections.shape[0] == 0:
        return correct

    iou, _ = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]

    for i, iou_thr in enumerate(iouv):
        matches = torch.where((iou >= iou_thr) & correct_class)
        if matches[0].numel() == 0:
            continue

        matched = torch.cat(
            (torch.stack(matches, dim=1), iou[matches[0], matches[1]][:, None]),
            dim=1,
        ).cpu().numpy()

        if matched.shape[0] > 1:
            matched = matched[matched[:, 2].argsort()[::-1]]
            matched = matched[np.unique(matched[:, 1], return_index=True)[1]]
            matched = matched[matched[:, 2].argsort()[::-1]]
            matched = matched[np.unique(matched[:, 0], return_index=True)[1]]

        correct[matched[:, 1].astype(int), i] = True

    return correct


def compute_ultralytics_prf1(stats, num_iou=10, eps=1e-16):
    """Compute P/R/F1 with the same confidence-swept max-F1 logic used by Ultralytics."""
    merged_stats = []
    for rank_stats in dist.all_gather(stats):
        merged_stats.extend(rank_stats)

    curve_x = np.linspace(0.0, 1.0, 1000)
    empty_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "best_index": 0,
        "confidence": 0.0,
        "curve_x": curve_x,
        "pr_curve": np.zeros_like(curve_x),
        "p_curve": np.zeros((0, curve_x.size), dtype=np.float64),
        "r_curve": np.zeros((0, curve_x.size), dtype=np.float64),
        "f1_curve": np.zeros((0, curve_x.size), dtype=np.float64),
        "class_precision": np.zeros((0,), dtype=np.float64),
        "class_recall": np.zeros((0,), dtype=np.float64),
        "class_f1": np.zeros((0,), dtype=np.float64),
        "unique_classes": np.zeros((0,), dtype=np.int64),
    }

    if not merged_stats:
        return empty_metrics

    correct = np.concatenate(
        [s[0] for s in merged_stats if s[0].size],
        axis=0,
    ) if any(s[0].size for s in merged_stats) else np.zeros((0, num_iou), dtype=bool)
    conf = np.concatenate(
        [s[1] for s in merged_stats if s[1].size],
        axis=0,
    ) if any(s[1].size for s in merged_stats) else np.zeros((0,), dtype=np.float64)
    pred_cls = np.concatenate(
        [s[2] for s in merged_stats if s[2].size],
        axis=0,
    ) if any(s[2].size for s in merged_stats) else np.zeros((0,), dtype=np.int64)
    target_cls = np.concatenate(
        [s[3] for s in merged_stats if s[3].size],
        axis=0,
    ) if any(s[3].size for s in merged_stats) else np.zeros((0,), dtype=np.int64)

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    if nc == 0:
        return empty_metrics

    order = np.argsort(-conf) if conf.size else np.zeros((0,), dtype=np.int64)
    correct, conf, pred_cls = correct[order], conf[order], pred_cls[order]

    p_curve = np.zeros((nc, curve_x.size), dtype=np.float64)
    r_curve = np.zeros((nc, curve_x.size), dtype=np.float64)
    prec_values = []

    for ci, c in enumerate(unique_classes):
        class_mask = pred_cls == c
        n_labels = nt[ci]
        n_preds = int(class_mask.sum())
        if n_preds == 0 or n_labels == 0:
            continue

        class_correct = correct[class_mask].astype(np.float64)
        tpc = class_correct.cumsum(0)
        fpc = (1.0 - class_correct).cumsum(0)

        recall = tpc / (n_labels + eps)
        precision = tpc / (tpc + fpc + eps)

        p_curve[ci] = np.interp(-curve_x, -conf[class_mask], precision[:, 0], left=1.0)
        r_curve[ci] = np.interp(-curve_x, -conf[class_mask], recall[:, 0], left=0.0)

        _, mpre, mrec = compute_ap(recall[:, 0], precision[:, 0])
        prec_values.append(np.interp(curve_x, mrec, mpre))

    pr_curve = np.mean(np.stack(prec_values, axis=0), axis=0) if prec_values else np.zeros_like(curve_x)
    f1_curve = 2.0 * p_curve * r_curve / (p_curve + r_curve + eps)

    best_index = int(smooth(f1_curve.mean(0), 0.1).argmax()) if f1_curve.size else 0
    class_precision = p_curve[:, best_index]
    class_recall = r_curve[:, best_index]
    class_f1 = f1_curve[:, best_index]

    return {
        "precision": float(class_precision.mean()) if class_precision.size else 0.0,
        "recall": float(class_recall.mean()) if class_recall.size else 0.0,
        "f1": float(class_f1.mean()) if class_f1.size else 0.0,
        "best_index": best_index,
        "confidence": float(curve_x[best_index]),
        "curve_x": curve_x,
        "pr_curve": pr_curve,
        "p_curve": p_curve,
        "r_curve": r_curve,
        "f1_curve": f1_curve,
        "class_precision": class_precision,
        "class_recall": class_recall,
        "class_f1": class_f1,
        "unique_classes": unique_classes.astype(np.int64),
    }


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        self.custom_prf1 = None
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()
            if iou_type == 'bbox' and self.custom_prf1 is not None:
                print(
                    " Ultralytics-style metrics @[ IoU=0.50 | maxF1 ]: "
                    "P = {:.3f}, R = {:.3f}, F1 = {:.3f}".format(
                        self.custom_prf1["precision"],
                        self.custom_prf1["recall"],
                        self.custom_prf1["f1"],
                    )
                )

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = dist.all_gather(img_ids)
    all_eval_imgs = dist.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


# import io
# from contextlib import redirect_stdout
# def evaluate(imgs):
#     with redirect_stdout(io.StringIO()):
#         imgs.evaluate()
#     return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
