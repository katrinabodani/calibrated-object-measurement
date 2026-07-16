"""
infer.py — End-to-end segmentation inference.

Takes an image captured with the calibrated camera, undistorts it with the Step 1
intrinsics, runs the trained Mask R-CNN, and writes an annotated output showing
the detected mask (filled) + contour + bounding box + confidence score.

This is the Step 2 inference pipeline; Step 3's measurement script extends it with
reference-object scaling and pixel-to-mm conversion.

Usage:
    python inference/infer.py --image path/to/image.jpg
    python inference/infer.py --image path/to/folder --output inference/outputs
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import yaml
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(num_classes, min_size, max_size):
    # weights=None: we load our own fine-tuned checkpoint (no downloads).
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                  min_size=min_size, max_size=max_size)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    return model


def annotate(img_bgr, out, score_thresh):
    """Draw filled mask + contour + box + score for detections above threshold."""
    overlay = img_bgr.copy()
    dets = []
    for box, score, mask in zip(out["boxes"], out["scores"], out["masks"]):
        if float(score) < score_thresh:
            continue
        m = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)
        overlay[m > 0] = (0, 255, 0)
        dets.append(([int(v) for v in box.cpu().numpy()], float(score), m))
    img = cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0)
    for (x1, y1, x2, y2), score, m in dets:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 180, 0), 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"medicine_box {score:.2f}", (x1, max(0, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
    return img, dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="image file or folder")
    ap.add_argument("--weights", default="models/weights/maskrcnn_medicinebox_best.pth")
    ap.add_argument("--intrinsics", default="calibration/camera_intrinsics.npz")
    ap.add_argument("--config", default="models/configs/maskrcnn.yaml")
    ap.add_argument("--output", default="inference/outputs")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--no-undistort", action="store_true",
                    help="skip undistortion (NOT recommended — geometry will be off)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data = np.load(args.intrinsics)
    K, dist = data["K"], data["dist"]
    calib_size = tuple(int(x) for x in data["image_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["num_classes"], cfg["model"]["min_size"],
                        cfg["model"]["max_size"]).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    if os.path.isdir(args.image):
        files = sorted(glob.glob(os.path.join(args.image, "*.jpg")) +
                       glob.glob(os.path.join(args.image, "*.png")))
    else:
        files = [args.image]
    os.makedirs(args.output, exist_ok=True)

    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"[skip] {f} unreadable")
            continue
        h, w = img.shape[:2]
        if not args.no_undistort:
            if (w, h) != calib_size:
                print(f"[warn] {os.path.basename(f)} size {(w, h)} != calibration {calib_size}; "
                      f"undistortion may be invalid (use the same camera config as calibration)")
            img = cv2.undistort(img, K, dist, None, K)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
        with torch.no_grad():
            out = model([t])[0]

        annotated, dets = annotate(img, out, args.score_thresh)
        name = os.path.splitext(os.path.basename(f))[0]
        outp = os.path.join(args.output, f"{name}_pred.jpg")
        cv2.imwrite(outp, annotated)
        top = max((d[1] for d in dets), default=0.0)
        print(f"{os.path.basename(f)}: {len(dets)} detection(s), top score {top:.2f} -> {outp}")


if __name__ == "__main__":
    main()
