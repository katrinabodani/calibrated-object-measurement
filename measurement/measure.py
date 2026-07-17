"""
measure.py : Pixel-to-mm measurement of the box using a card reference.

Pipeline (per image):
  1. Undistort with the Step 1 intrinsics.
  2. Detect the reference card (rectangle) -> compute pixels_per_mm from its
     known real-world size.
  3. Run Mask R-CNN -> segment the box -> mask.
  4. Fit a min-area rectangle to the mask -> box side lengths in pixels.
  5. Convert to millimetres via pixels_per_mm.
  6. Annotate + (optionally) compare against ruler ground truth -> MAE / MPE.

Usage:
    # single image
    python measurement/measure.py --image measurement/images/1.jpg \
        --card-w 90 --card-h 60

    # folder + accuracy validation against ground truth
    python measurement/measure.py --image measurement/images \
        --card-w 90 --card-h 60 --gt-w 90 --gt-h 43
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
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                  min_size=min_size, max_size=max_size)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    return model

def order_corners(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)  # TL,TR,BR,BL

# card detection
def detect_card(img_bgr, card_ratio, exclude_mask=None,
                min_area_frac=0.006, max_area_frac=0.5, ratio_tol=0.45, fill_min=0.80):
    """Find the card as a solid rectangle of the right aspect ratio.
    Uses Otsu thresholding (both polarities, so it works for a bright card on a
    dark surface OR a dark card on a light surface), then keeps blobs that are
    (a) the right size, (b) rectangular (contour fills its min-area rect), and
    (c) the right aspect ratio. `exclude_mask` suppresses the box region.
    Returns ordered corners (4x2) or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    H, W = gray.shape
    img_area = H * W
    kernel = np.ones((7, 7), np.uint8)

    _, o1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, o2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    best, best_score = None, 0
    for bw in (o1, o2):
        if exclude_mask is not None:
            bw = bw.copy()
            bw[exclude_mask > 0] = 0
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area_frac * img_area or area > max_area_frac * img_area:
                continue
            rect = cv2.minAreaRect(c)
            (w, h) = rect[1]
            if w == 0 or h == 0:
                continue
            fill = area / (w * h)
            ar = max(w, h) / min(w, h)
            if fill < fill_min or abs(ar - card_ratio) > ratio_tol:
                continue
            score = fill * area
            if score > best_score:
                best_score, best = score, cv2.boxPoints(rect)
    return order_corners(best) if best is not None else None


def pixels_per_mm(corners, card_w, card_h):
    """Average pixels-per-mm from the 4 card edges (orientation-agnostic)."""
    tl, tr, br, bl = corners
    horiz = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    vert = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    long_px, short_px = max(horiz, vert), min(horiz, vert)
    long_mm, short_mm = max(card_w, card_h), min(card_w, card_h)
    return (long_px / long_mm + short_px / short_mm) / 2


# measurement
def measure_box(mask):
    """Min-area rectangle of the mask -> (long_px, short_px, box_points)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (w, h) = rect[1]
    box_pts = cv2.boxPoints(rect).astype(int)
    return max(w, h), min(w, h), box_pts


def run_one(img_bgr, model, device, K, dist, card_w, card_h, score_thresh, box_thickness=0.0):
    und = cv2.undistort(img_bgr, K, dist, None, K)
    rgb = cv2.cvtColor(und, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
    with torch.no_grad():
        out = model([t])[0]

    # best box detection
    if len(out["scores"]) == 0 or float(out["scores"][0]) < score_thresh:
        return und, None
    idx = int(torch.argmax(out["scores"]))
    score = float(out["scores"][idx])
    mask = (out["masks"][idx, 0].cpu().numpy() > 0.5).astype(np.uint8)

    long_px, short_px, box_pts = measure_box(mask)
    card = detect_card(und, card_ratio=max(card_w, card_h) / min(card_w, card_h),
                       exclude_mask=mask)
    if card is None:
        return und, {"score": score, "mask": mask, "box_pts": box_pts, "card": None}
    ppm = pixels_per_mm(card, card_w, card_h)
    long_mm, short_mm = long_px / ppm, short_px / ppm
    # Depth-offset correction: the box top face sits `box_thickness` mm closer to
    # the camera than the flat card, so its measured size is inflated by D/(D-t).
    # Estimate camera->card distance D = f / ppm, then rescale by (D - t)/D.
    if box_thickness > 0:
        f = (K[0, 0] + K[1, 1]) / 2.0
        D = f / ppm
        factor = (D - box_thickness) / D
        long_mm *= factor
        short_mm *= factor
    return und, {
        "score": score, "mask": mask, "box_pts": box_pts, "card": card, "ppm": ppm,
        "long_mm": long_mm, "short_mm": short_mm,
    }


def annotate(und, res):
    overlay = und.copy()
    overlay[res["mask"] > 0] = (0, 255, 0)
    img = cv2.addWeighted(overlay, 0.35, und, 0.65, 0)
    bp = res["box_pts"].astype(np.float32)
    cv2.drawContours(img, [bp.astype(int)], -1, (0, 165, 255), 4)      # orange: measured rect

    def label(org, text, scale=1.5):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 8)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3)

    if res.get("card") is not None:
        cv2.polylines(img, [res["card"].astype(int)], True, (255, 0, 0), 4)  # blue: card
        cxy = bp.mean(0)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        lens = [np.linalg.norm(bp[a] - bp[b]) for a, b in edges]
        li, si = int(np.argmax(lens)), int(np.argmin(lens))

        def place_on_edge(i, text):
            a, b = edges[i]
            m = (bp[a] + bp[b]) / 2
            d = m - cxy
            d = d / (np.linalg.norm(d) + 1e-6)
            p = m + d * 70                      # push the label outside the box edge
            label((int(p[0] - 130), int(p[1])), text)

        place_on_edge(li, f"W: {res['long_mm']:.1f} mm")   # width 
        place_on_edge(si, f"H: {res['short_mm']:.1f} mm")  # height
        c = bp.min(0).astype(int)
        label((c[0], c[1] - 20), f"conf {res['score']:.2f}", scale=1.2)
    else:
        label((int(bp[:, 0].mean()) - 220, int(bp[:, 1].mean())), "CARD NOT FOUND")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="image file or folder")
    ap.add_argument("--card-w", type=float, required=True, help="card width (mm)")
    ap.add_argument("--card-h", type=float, required=True, help="card height (mm)")
    ap.add_argument("--gt-w", type=float, help="box ground-truth width (mm)")
    ap.add_argument("--gt-h", type=float, help="box ground-truth height (mm)")
    ap.add_argument("--box-thickness", type=float, default=0.0,
                    help="box depth (mm) for the flat-card depth-offset correction; 0 = off")
    ap.add_argument("--weights", default="models/weights/maskrcnn_medicinebox_best.pth")
    ap.add_argument("--intrinsics", default="calibration/camera_intrinsics.npz")
    ap.add_argument("--config", default="models/configs/maskrcnn.yaml")
    ap.add_argument("--output", default="measurement/outputs")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data = np.load(args.intrinsics)
    K, dist = data["K"], data["dist"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["num_classes"], cfg["model"]["min_size"],
                        cfg["model"]["max_size"]).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    files = (sorted(glob.glob(os.path.join(args.image, "*.jpg")))
             if os.path.isdir(args.image) else [args.image])
    os.makedirs(args.output, exist_ok=True)

    gt_long = max(args.gt_w, args.gt_h) if args.gt_w and args.gt_h else None
    gt_short = min(args.gt_w, args.gt_h) if args.gt_w and args.gt_h else None
    rows, errs, perrs = [], [], []

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        # Rotate portrait captures back to the calibrated landscape orientation.
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        und, res = run_one(img, model, device, K, dist, args.card_w, args.card_h,
                           args.score_thresh, args.box_thickness)
        name = os.path.splitext(os.path.basename(f))[0]
        if res is None:
            print(f"{name}: no box detected"); continue
        img_out = annotate(und, res)
        cv2.imwrite(os.path.join(args.output, f"{name}_measured.jpg"), img_out)
        if res.get("card") is None:
            print(f"{name}: box conf {res['score']:.2f} but CARD NOT FOUND")
            continue
        L, S = res["long_mm"], res["short_mm"]
        line = f"{name}: {L:.1f} x {S:.1f} mm  (conf {res['score']:.2f})"
        if gt_long:
            eL, eS = abs(L - gt_long), abs(S - gt_short)
            errs += [eL, eS]
            perrs += [eL / gt_long * 100, eS / gt_short * 100]
            line += f"  | GT {gt_long:.1f}x{gt_short:.1f}  err {eL:.1f}/{eS:.1f} mm"
            rows.append((name, L, S, eL, eS))
        print(line)

    if gt_long and errs:
        mae = np.mean(errs)
        mpe = np.mean(perrs)
        print("\n=== ACCURACY (n={} images, {} measurements) ===".format(len(rows), len(errs)))
        print(f"  MAE = {mae:.2f} mm   MPE = {mpe:.2f} %")
        # write report
        with open(os.path.join("measurement", "accuracy_report.csv"), "w") as fp:
            fp.write("image,long_mm,short_mm,long_abs_err_mm,short_abs_err_mm\n")
            for n, L, S, eL, eS in rows:
                fp.write(f"{n},{L:.2f},{S:.2f},{eL:.2f},{eS:.2f}\n")
            fp.write(f"# GT long={gt_long} short={gt_short}; MAE={mae:.2f}mm MPE={mpe:.2f}%\n")
        print("  wrote measurement/accuracy_report.csv")


if __name__ == "__main__":
    main()
