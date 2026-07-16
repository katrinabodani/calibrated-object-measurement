# XIS Technical Assessment — Calibrated Object Measurement Pipeline

An end-to-end computer-vision pipeline that segments a custom object, removes lens
distortion via intrinsic camera calibration, and computes the object's real-world
**width and height in millimetres** from calibrated pixel data.

> XIS AI / Computer Vision Department — Technical Hiring Assessment.

---

## What this system does

1. **Camera calibration** — intrinsic calibration from checkerboard images (OpenCV) to remove radial & tangential lens distortion.
2. **Dataset** — a self-collected, self-labelled image dataset of the target object.
3. **Segmentation model** — a trained deep-learning segmentation model (not YOLO / not Roboflow).
4. **Measurement** — pixel→mm conversion using a known-size reference object, validated against physical ruler/calliper measurements.

## Repository structure

```
project-root/
  calibration/    # Calibration images (Drive) and scripts
  dataset/        # Raw images and labels, train/val/test splits (Drive)
  models/         # Training configs and saved weights (weights on Drive)
  inference/      # Inference scripts and demo outputs
  measurement/    # Pixel-to-mm pipeline and accuracy report
  docs/           # All documentation files
  requirements.txt
  README.md
```

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/SETUP.md](docs/SETUP.md) | Installation, environment, run instructions |
| [docs/CALIBRATION_REPORT.md](docs/CALIBRATION_REPORT.md) | Method, parameters, reprojection error |
| [docs/DATASET_CARD.md](docs/DATASET_CARD.md) | Object, collection method, labelling tool, statistics |
| [docs/TRAINING_REPORT.md](docs/TRAINING_REPORT.md) | Architecture, hyperparameters, metrics, loss curves |
| [docs/MEASUREMENT_REPORT.md](docs/MEASUREMENT_REPORT.md) | Methodology, accuracy table, error analysis |

## Large files — Google Drive (Section 2.2)

Per the assessment rules, large files are **not** committed to GitHub. They are hosted
on Google Drive with "Anyone with the link can view" access:

| Contents | Link |
|----------|------|
| Calibration Images (26) | https://drive.google.com/drive/folders/1LQ4apP_msEcUxKVfd0TrxYbZwzsqiJ8q?usp=drive_link |
| Dataset — Raw Images (81) | https://drive.google.com/drive/folders/1xIZfy29QP5V6S062ttYWHtwFYX65tbnY?usp=sharing |
| Dataset — Undistorted Images (81) | https://drive.google.com/drive/folders/1A5up4ismFSmDij8h1h8rnFilsmp6yAFC?usp=sharing |
| Labelled Dataset (COCO, 70/20/10 splits) | https://drive.google.com/drive/folders/1Df8IbTGVSz5snYy91Ljvjl3dpJk55sYc?usp=sharing |
| Label Previews (train split) | https://drive.google.com/drive/folders/1uVhL6bZ58uZ3dP9xCHeG8szitGGb_I8r?usp=sharing |
| Trained Model Weights (Mask R-CNN, 169 MB) | https://drive.google.com/drive/folders/1Uq1IY6aswwlCG9xTPO8lu_6POrOXuaqT?usp=sharing |
| Test-Set Prediction Visualizations (8) | https://drive.google.com/drive/folders/18H43ikEMC6RMTZOZvXmquXXeLfF6vV0n?usp=sharing |
| Measurement Photos (box + card, 12) | https://drive.google.com/drive/folders/1riLU7_6wdC35noKtAQEi17CJ54YjlVxt?usp=sharing |
| Annotated Measurement Outputs | https://drive.google.com/drive/folders/1dDJJLvi_sjznP6Lqsm1T_Qea5iBP0txQ?usp=sharing |

## Quick start

See [docs/SETUP.md](docs/SETUP.md) for full instructions.

```bash
pip install -r requirements.txt
# 1. Calibrate:   python calibration/scripts/calibrate.py
# 2. Train:       python models/train.py --config models/configs/<config>
# 3. Inference:   python inference/infer.py --image <path>
# 4. Measure:     python measurement/measure.py --image <path>
```

## Status

- **Step 1 — Calibration & dataset:** camera calibrated (intrinsics + undistortion),
  81-image dataset collected, undistorted, labelled (COCO), split 70/20/10.
- **Step 2 — Model training:** Mask R-CNN trained (test mAP@0.5 = 1.0, IoU = 0.972);
  inference pipeline (`inference/infer.py`) undistorts → segments → annotates.
- **Step 3 — Pixel-to-mm measurement:** card-referenced measurement with depth
  correction; validated on 12 images → **MAE 1.53 mm, MPE 2.15%**.

All three steps complete. See commit history for incremental progress.
