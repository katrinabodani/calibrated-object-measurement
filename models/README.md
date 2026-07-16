# Models

Training configs and trained weights.

## Contents
- `train.py`, `dataset.py`, `engine.py` — Mask R-CNN training + evaluation code
- `configs/maskrcnn.yaml` — training configuration (tracked in git)
- `loss_curves.png`, `metrics.json` — training artifacts (tracked in git)
- `test_predictions/` — held-out test-set prediction overlays (**NOT in git** — hosted on Drive)
- `weights/maskrcnn_medicinebox_best.pth` — best checkpoint (**NOT in git** — hosted on Drive)

See `docs/TRAINING_REPORT.md` for architecture, hyperparameters, and metrics.

## Large files — Google Drive
> Per Section 2.2, trained model weights and prediction visualizations are hosted externally.

- **Trained Model Weights (Mask R-CNN best checkpoint, ~169 MB):** https://drive.google.com/drive/folders/1Uq1IY6aswwlCG9xTPO8lu_6POrOXuaqT?usp=sharing
- **Test-Set Prediction Visualizations (8 images):** https://drive.google.com/drive/folders/18H43ikEMC6RMTZOZvXmquXXeLfF6vV0n?usp=sharing

_Link access must be set to "Anyone with the link can view."_
