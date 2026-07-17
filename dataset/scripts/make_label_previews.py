"""
make_label_previews.py - Render a COCO split's masks onto its images so the
labelling quality is viewable at a glance (no JSON tooling needed) -- just to visibly see the labels.

For each image in the chosen split, decodes the RLE mask and draws a translucent
fill + contour, then saves a resized preview. Intended for a small sample upload
(e.g. the train split) so reviewers can see the annotations.

Usage:
    python dataset/scripts/make_label_previews.py \
        --split dataset/splits/train --output dataset/label_previews --scale 0.5
"""
import argparse
import json
import os

import cv2
import numpy as np
from pycocotools import mask as maskUtils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dataset/splits/train", help="folder with images + _annotations.coco.json")
    ap.add_argument("--output", default="dataset/label_previews")
    ap.add_argument("--scale", type=float, default=0.5, help="resize factor for previews")
    args = ap.parse_args()

    d = json.load(open(os.path.join(args.split, "_annotations.coco.json")))
    id2name = {im["id"]: im["file_name"] for im in d["images"]}
    by_img = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    os.makedirs(args.output, exist_ok=True)
    n = 0
    for img_id, name in id2name.items():
        img = cv2.imread(os.path.join(args.split, name))
        if img is None:
            continue
        overlay = img.copy()
        for a in by_img.get(img_id, []):
            m = maskUtils.decode(a["segmentation"]).astype(np.uint8)
            overlay[m > 0] = (0, 200, 0)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), 4)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(args.output, f"preview_{name}"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        n += 1

    print(f"Wrote {n} label previews to {args.output}")


if __name__ == "__main__":
    main()
