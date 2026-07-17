"""
engine.py : training epoch, validation loss, and evaluation (COCO mAP + IoU/P/R/F1).

Evaluation reports:
  - mAP@0.5 and mAP@0.5:0.95 for both segm and bbox (pycocotools COCOeval),
  - mean mask IoU (best prediction vs GT),
  - precision / recall / F1 at a fixed score+IoU threshold.
"""
import contextlib
import io

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


def train_one_epoch(model, optimizer, loader, device, scaler=None):
    model.train()
    total, n = 0.0, 0
    for imgs, targets in loader:
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def compute_val_loss(model, loader, device):
    """Loss on val set (model in train() mode returns losses; no grad)."""
    model.train()
    total, n = 0.0, 0
    for imgs, targets in loader:
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        total += float(sum(loss_dict.values()).item())
        n += 1
    return total / max(n, 1)


def mask_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def _coco_map(gt_json, results, fg_cat_id, quiet=True):
    out = {"segm": [0.0, 0.0], "bbox": [0.0, 0.0]}
    if not results:
        return out
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:
        coco_gt = COCO(gt_json)
        coco_dt = coco_gt.loadRes(results)
        for it in ["bbox", "segm"]:
            E = COCOeval(coco_gt, coco_dt, it)
            E.params.catIds = [fg_cat_id]
            E.evaluate()
            E.accumulate()
            E.summarize()
            out[it] = [float(E.stats[0]), float(E.stats[1])]  # [AP@.5:.95, AP@.5]
    return out


@torch.no_grad()
def evaluate(model, loader, device, gt_json, fg_cat_id=1,
             score_thresh=0.5, iou_thresh=0.5, quiet=True):
    model.eval()
    results = []
    ious = []
    tp = fp = fn = 0

    for imgs, targets in loader:
        outputs = model([i.to(device) for i in imgs])
        for t, out in zip(targets, outputs):
            img_id = int(t["image_id"])
            gt_masks = t["masks"].numpy()

            preds = []
            for box, score, mask in zip(out["boxes"], out["scores"], out["masks"]):
                m = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)
                preds.append((float(score), m))
                rle = maskUtils.encode(np.asfortranarray(m))
                rle["counts"] = rle["counts"].decode("ascii")
                x1, y1, x2, y2 = [float(v) for v in box.cpu().numpy()]
                results.append({
                    "image_id": img_id, "category_id": fg_cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score), "segmentation": rle,
                })

            # precision/recall/F1 + IoU at threshold (greedy match by score)
            preds_f = sorted([p for p in preds if p[0] >= score_thresh], key=lambda p: -p[0])
            matched = set()
            for score, m in preds_f:
                best_iou, best_g = 0.0, -1
                for g in range(gt_masks.shape[0]):
                    if g in matched:
                        continue
                    iou = mask_iou(m, gt_masks[g])
                    if iou > best_iou:
                        best_iou, best_g = iou, g
                if best_iou >= iou_thresh and best_g >= 0:
                    tp += 1
                    matched.add(best_g)
                    ious.append(best_iou)
                else:
                    fp += 1
            fn += gt_masks.shape[0] - len(matched)

    mAP = _coco_map(gt_json, results, fg_cat_id, quiet)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "mAP@0.5:0.95_segm": mAP["segm"][0], "mAP@0.5_segm": mAP["segm"][1],
        "mAP@0.5:0.95_bbox": mAP["bbox"][0], "mAP@0.5_bbox": mAP["bbox"][1],
        "mean_IoU": float(np.mean(ious)) if ious else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }
    return metrics, results
