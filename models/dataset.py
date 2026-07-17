"""
dataset.py : COCO instance-segmentation dataset for the medicine-box splits.

Reads a Roboflow COCO export split (images + _annotations.coco.json with RLE
masks) and yields (image_tensor, target) in the format torchvision detection
models expect. Single foreground class -> label 1 (0 = background).

Light augmentation (horizontal flip + photometric jitter) applied when train=True;
flips are applied consistently to image, masks, and boxes.
"""
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as maskUtils


def _photometric(img, strength):
    """Random brightness/contrast jitter on a uint8 HWC image."""
    img = img.astype(np.float32)
    b = 1.0 + random.uniform(-strength, strength)   # contrast
    c = random.uniform(-strength, strength) * 255.0  # brightness
    img = img * b + c
    return np.clip(img, 0, 255).astype(np.uint8)


class CocoBoxDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir, train=False, hflip=0.5, photometric=0.0):
        self.dir = split_dir
        self.train = train
        self.hflip = hflip
        self.photometric = photometric

        coco = json.load(open(os.path.join(split_dir, "_annotations.coco.json")))
        self.images = {im["id"]: im for im in coco["images"]}
        self.img_ids = sorted(self.images.keys())
        self.anns_by_img = {}
        for a in coco["annotations"]:
            self.anns_by_img.setdefault(a["image_id"], []).append(a)

        # Foreground category id (the class actually used by annotations).
        used = {a["category_id"] for a in coco["annotations"]}
        self.fg_category_id = sorted(used)[0] if used else 1

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.images[img_id]
        img = np.array(Image.open(os.path.join(self.dir, info["file_name"])).convert("RGB"))
        H, W = img.shape[:2]

        masks, boxes = [], []
        for a in self.anns_by_img.get(img_id, []):
            m = maskUtils.decode(a["segmentation"])
            if m.ndim == 3:
                m = (m.sum(2) > 0).astype(np.uint8)
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            masks.append(m.astype(np.uint8))
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])

        masks = np.stack(masks) if masks else np.zeros((0, H, W), np.uint8)
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)

        if self.train:
            if random.random() < self.hflip:
                img = img[:, ::-1, :].copy()
                masks = masks[:, :, ::-1].copy()
                x1 = W - boxes[:, 2]
                x2 = W - boxes[:, 0]
                boxes[:, 0], boxes[:, 2] = x1, x2
            if self.photometric > 0:
                img = _photometric(img, self.photometric)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        n = len(boxes)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((n,), dtype=torch.int64),  # single class -> 1
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor(img_id),
            "area": torch.as_tensor((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])),
            "iscrowd": torch.zeros((n,), dtype=torch.int64),
        }
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))
