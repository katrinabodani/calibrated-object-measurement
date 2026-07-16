# Model Training Report

## 1. Architecture

- **Model:** **Mask R-CNN** (instance segmentation) — `maskrcnn_resnet50_fpn`
  from **torchvision** 0.21.
- **Backbone:** ResNet-50 + Feature Pyramid Network (FPN).
- **Pretrained:** COCO-pretrained weights (transfer learning). The box and mask
  heads are replaced for **2 classes** (background + `medicine_box`).
- **Not** Ultralytics YOLO and **not** a Roboflow model (per assessment rules).

### Selection rationale
| Reason | Detail |
|--------|--------|
| Metric fit | The assessment requires **mAP@0.5 / mAP@0.5:0.95** — native instance-segmentation metrics that Mask R-CNN produces directly via pycocotools. |
| Step 3 needs a **confidence score** | Mask R-CNN outputs a per-instance confidence natively (required for the end-to-end demo output). |
| Small dataset (81 imgs) | COCO-pretraining + transfer learning fine-tunes well on few images. |
| Label compatibility | Trains directly on the COCO RLE masks — no format conversion. |
| Practicality | torchvision implementation installs cleanly on Windows (avoids Detectron2 build issues) and runs on the 6 GB RTX 4050. |

## 2. Training Configuration

Config file: `models/configs/maskrcnn.yaml` (fully reproducible).

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 25 |
| Batch size | 2 |
| Optimizer | SGD (momentum 0.9, weight decay 5e-4) |
| Learning rate | 0.005, StepLR ×0.1 at epoch 15 |
| Image resize | short side 640 / long side 1024 (fits 6 GB VRAM) |
| Mixed precision | Yes (AMP) |
| Augmentation | Random horizontal flip (p=0.5), brightness/contrast jitter |
| Seed | 42 |
| Hardware | NVIDIA RTX 4050 Laptop (6 GB), CUDA 12.4, torch 2.6 |

Best checkpoint selected by **lowest validation loss** (val mAP saturates at 1.0
on this single-object task, so it cannot discriminate between epochs).

## 3. Dataset

70/20/10 split — 57 train / 16 val / 8 test — single class `medicine_box`,
one instance per image. See `docs/DATASET_CARD.md`. All images undistorted with
the Step 1 intrinsics before training.

## 4. Metrics

### Validation (best epoch = 13, val loss 0.069)
| Metric | Value |
|--------|-------|
| mAP@0.5 (segm) | 1.000 |
| mAP@0.5:0.95 (segm) | 1.000 |
| mean IoU | 0.972 |

### Test set (held-out, 8 images, best checkpoint)
| Metric | Value |
|--------|-------|
| mAP@0.5 (segm) | **1.000** |
| mAP@0.5:0.95 (segm) | **1.000** |
| mAP@0.5 (bbox) | 1.000 |
| mAP@0.5:0.95 (bbox) | 0.938 |
| **mean IoU** | **0.972** |
| Precision / Recall / F1 | 1.00 / 1.00 / 1.00 |
| TP / FP / FN | 8 / 0 / 0 |

Full numbers in `models/metrics.json`.

## 5. Loss Curves

`models/loss_curves.png` — train and validation loss both drop sharply in the
first ~2 epochs (0.58/0.69 → ~0.11), then converge smoothly to ~0.06–0.07 and
plateau. Train and val track together (**no overfitting**); val mAP@0.5 reaches
1.0 by epoch 2.

## 6. Predictions on the Held-Out Test Set

`models/test_predictions/pred_1.jpg … pred_8.jpg` — each shows the predicted
**filled mask** (translucent green) + contour + bounding box + confidence score.
**Everything drawn here is the model's prediction** on held-out test images it
never saw during training — the green mask, the red bounding box, and the
confidence score are all model output; **no ground truth is drawn** in these
figures. Masks are tight to the box edges across varied poses and backgrounds;
confidence ≈ 1.0 on all test images.

## 7. Reproducibility

```bash
pip install -r requirements.txt   # torch/torchvision: use the CUDA index-url
python models/train.py --config models/configs/maskrcnn.yaml
```
Fixed seed (42). Config-driven. Best weights hosted on Drive (see `models/README.md`).

## 8. Assumptions & Limitations

- **Small test set (8 images).** The perfect mAP/precision/recall/F1 are
  encouraging but coarse at this size; **mean IoU (0.972)** is the more
  informative quality signal. The task is intrinsically easy — a single, large,
  visually distinct object per image.
- **Single instance / single class.** The pipeline is not stress-tested on
  multiple boxes per image or occlusion.
- **Controlled conditions.** All images are the same physical box captured with
  one camera; generalisation to new boxes/cameras is out of scope.
- **Val metric saturation.** Val mAP hits 1.0 early, so validation loss (not mAP)
  drives checkpoint selection.
