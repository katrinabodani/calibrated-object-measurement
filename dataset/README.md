# Dataset

Raw images, labels, and train/val/test splits for the chosen object.

## Contents
- `raw/` — original captured images, 12 MP landscape (**NOT in git** — hosted on Drive)
- `undistorted/` — raw images undistorted with Step 1 intrinsics; the actual
  training/inference inputs (**NOT in git** — hosted on Drive)
- `labels/` — annotation exports from the labelling tool (**NOT in git** — hosted on Drive)
- `splits/` — 70% train / 20% val / 10% test (**NOT in git** — hosted on Drive)
- `scripts/` — `undistort_images.py` (tracked in git)

Images are named `1.jpg … 81.jpg`; `raw/N.jpg` and `undistorted/N.jpg` are the
same source image. See `docs/DATASET_CARD.md` for object, collection strategy,
labelling tool, and class distribution.

## Large files — Google Drive
> Per Section 2.2, all dataset images and label exports are hosted externally.

- **Raw Dataset Images (81, originals):** https://drive.google.com/drive/folders/1xIZfy29QP5V6S062ttYWHtwFYX65tbnY?usp=sharing
- **Undistorted Dataset Images (81):** https://drive.google.com/drive/folders/1A5up4ismFSmDij8h1h8rnFilsmp6yAFC?usp=sharing
- **Labelled Export (images + masks, COCO):** _`<PASTE DRIVE URL>`_
- **Train/Val/Test Splits:** _`<PASTE DRIVE URL>`_

_All links must be set to "Anyone with the link can view."_
