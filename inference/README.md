# Inference

End-to-end segmentation inference: **input image → undistort → Mask R-CNN → annotated output**.

## Contents
- `infer.py` — inference script (undistortion + segmentation + annotation)
- `outputs/` — annotated predictions (**NOT in git** — regenerated locally)

## Prerequisites
- Trained weights at `models/weights/maskrcnn_medicinebox_best.pth`
  (download from Drive — see `models/README.md`).
- Camera intrinsics at `calibration/camera_intrinsics.npz` (from Step 1).

## Usage
```bash
# single image (captured with the calibrated camera, 12 MP landscape)
python inference/infer.py --image path/to/image.jpg

# a folder of images
python inference/infer.py --image path/to/folder --output inference/outputs
```

### Options
| Flag | Default | Meaning |
|------|---------|---------|
| `--image` | (required) | Image file or folder |
| `--weights` | `models/weights/maskrcnn_medicinebox_best.pth` | Trained checkpoint |
| `--intrinsics` | `calibration/camera_intrinsics.npz` | Calibration parameters |
| `--config` | `models/configs/maskrcnn.yaml` | Model config (classes, resize) |
| `--output` | `inference/outputs` | Where annotated images are written |
| `--score-thresh` | `0.5` | Minimum detection confidence |
| `--no-undistort` | off | Skip undistortion (**not recommended**) |

## Output
For each input, writes `<name>_pred.jpg` — the **undistorted** image with the
detected **filled mask** + contour + bounding box + `medicine_box <confidence>`.
The console prints the number of detections and the top confidence score.

> **Important:** input images must come from the **same camera configuration** as
> calibration (iPhone 12 MP landscape, 4032×3024) — otherwise undistortion is
> invalid. The script warns on a size mismatch.

Step 3's `measurement/measure.py` extends this pipeline with reference-object
scaling to output width/height in millimetres.
