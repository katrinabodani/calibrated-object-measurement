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

## ⚠️ Large files — Google Drive (Section 2.2)

Per the assessment rules, large files are **not** committed to GitHub. They are hosted
on Google Drive with "Anyone with the link can view" access:

| Contents | Link |
|----------|------|
| Calibration Images | _`<PASTE DRIVE URL>`_ |
| Full Dataset (raw images) | _`<PASTE DRIVE URL>`_ |
| Labelled Dataset Export | _`<PASTE DRIVE URL>`_ |
| Trained Model Weights | _`<PASTE DRIVE URL>`_ |

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

🚧 In progress — repository scaffolding set up. See commit history for incremental progress.
