"""
train.py — Fine-tune Mask R-CNN (torchvision, ResNet50-FPN, COCO-pretrained) on
the medicine-box dataset.

Config-driven (models/configs/maskrcnn.yaml). Logs train/val loss and val mAP per
epoch, saves the best checkpoint (by segm mAP@0.5:0.95), plots loss curves,
evaluates the held-out test set, writes metrics.json, and renders test-set
prediction overlays.

Usage:
    python models/train.py --config models/configs/maskrcnn.yaml
"""
import argparse
import json
import os
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset import CocoBoxDataset, collate_fn
from engine import train_one_epoch, compute_val_loss, evaluate


def build_model(num_classes, min_size, max_size, pretrained=True):
    model = maskrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None, min_size=min_size, max_size=max_size)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    return model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def viz_predictions(model, dataset, device, out_dir, score_thresh=0.5, scale=0.33):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            img_t, target = dataset[i]
            out = model([img_t.to(device)])[0]
            img = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
            overlay = img.copy()
            dets = []
            for box, score, mask in zip(out["boxes"], out["scores"], out["masks"]):
                if score < score_thresh:
                    continue
                m = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)
                overlay[m > 0] = (0, 255, 0)                 # filled mask region
                dets.append(([int(v) for v in box.cpu().numpy()], float(score), m))
            # translucent mask fill, then draw crisp contour + box + score on top
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
            for (x1, y1, x2, y2), score, m in dets:
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (0, 180, 0), 3)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"box {score:.2f}", (x1, max(0, y1 - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
            small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, f"pred_{i+1}.jpg"), small)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models/configs/maskrcnn.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sd = cfg["data"]["splits_dir"]
    train_ds = CocoBoxDataset(os.path.join(sd, cfg["data"]["train"]), train=True,
                              hflip=cfg["augment"]["hflip"], photometric=cfg["augment"]["photometric"])
    val_ds = CocoBoxDataset(os.path.join(sd, cfg["data"]["val"]))
    test_ds = CocoBoxDataset(os.path.join(sd, cfg["data"]["test"]))
    fg = train_ds.fg_category_id

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["train"]["workers"], collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg["train"]["workers"], collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=cfg["train"]["workers"], collate_fn=collate_fn)

    val_gt = os.path.join(sd, cfg["data"]["val"], "_annotations.coco.json")
    test_gt = os.path.join(sd, cfg["data"]["test"], "_annotations.coco.json")

    model = build_model(cfg["model"]["num_classes"], cfg["model"]["min_size"],
                        cfg["model"]["max_size"], cfg["model"]["pretrained"]).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["train"]["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    else:
        optimizer = torch.optim.SGD(params, lr=cfg["train"]["lr"], momentum=cfg["train"]["momentum"],
                                    weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["train"]["lr_step_size"],
                                                gamma=cfg["train"]["lr_gamma"])
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    out_dir = cfg["output"]["dir"]
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)
    best_path = os.path.join(out_dir, "weights", cfg["output"]["weights_name"])

    history = {"train_loss": [], "val_loss": [], "val_mAP50_segm": [], "val_mAP_segm": []}
    best_val_loss = float("inf")
    best_epoch = 0
    ev = cfg["eval"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tl = train_one_epoch(model, optimizer, train_loader, device, scaler)
        vl = compute_val_loss(model, val_loader, device)
        scheduler.step()
        metrics, _ = evaluate(model, val_loader, device, val_gt, fg,
                              ev["score_thresh_prf"], ev["iou_thresh_prf"])
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_mAP50_segm"].append(metrics["mAP@0.5_segm"])
        history["val_mAP_segm"].append(metrics["mAP@0.5:0.95_segm"])
        print(f"[{epoch:02d}/{cfg['train']['epochs']}] train_loss={tl:.4f} val_loss={vl:.4f} "
              f"val_mAP@0.5(segm)={metrics['mAP@0.5_segm']:.3f} "
              f"val_mAP@[.5:.95]={metrics['mAP@0.5:0.95_segm']:.3f} IoU={metrics['mean_IoU']:.3f}")

        # Select best by lowest val loss (val mAP saturates on this easy task).
        if vl < best_val_loss:
            best_val_loss = vl
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"    saved best (val_loss={vl:.4f}, epoch {epoch}) -> {best_path}")

    # ---- Loss curves ----
    plt.figure(figsize=(8, 5))
    ep = range(1, len(history["train_loss"]) + 1)
    plt.plot(ep, history["train_loss"], label="train loss")
    plt.plot(ep, history["val_loss"], label="val loss")
    plt.plot(ep, history["val_mAP50_segm"], "--", label="val mAP@0.5 (segm)")
    plt.xlabel("epoch"); plt.ylabel("value"); plt.legend(); plt.grid(alpha=0.3)
    plt.title("Mask R-CNN training")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=120)
    print(f"Saved loss curves -> {out_dir}/loss_curves.png")

    # ---- Final test evaluation with best weights ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics, _ = evaluate(model, test_loader, device, test_gt, fg,
                               ev["score_thresh_prf"], ev["iou_thresh_prf"])
    print("\n=== TEST SET METRICS (best checkpoint) ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    json.dump({"config": cfg, "history": history, "test_metrics": test_metrics,
               "best_epoch": best_epoch, "best_val_loss": best_val_loss},
              open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    print(f"Saved metrics -> {out_dir}/metrics.json")

    viz_predictions(model, test_ds, device, os.path.join(out_dir, "test_predictions"),
                    score_thresh=ev["score_thresh_prf"])
    print(f"Saved test prediction overlays -> {out_dir}/test_predictions/")


if __name__ == "__main__":
    main()
