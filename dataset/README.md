# Dataset

Raw images, labels, and train/val/test splits for the chosen object.

## Contents
- `raw/` : original captured images, 12 MP landscape (**NOT in git** — hosted on Drive)
- `undistorted/` : raw images undistorted with Step 1 intrinsics; the actual
  training/inference inputs (**NOT in git** — hosted on Drive)
- `splits/` : the labelled COCO dataset split 70/20/10: `train/`, `val/`, `test/`,
  each holding its images + `_annotations.coco.json` (masks). (**NOT in git** — hosted on Drive)
- `label_previews/` : sample images with masks drawn on (train split), for quick
  visual review of labelling quality. (**NOT in git** — hosted on Drive)
- `scripts/` : `undistort_images.py`, `make_label_previews.py` (tracked in git)

Images are named `1.jpg … 81.jpg`; `raw/N.jpg` and `undistorted/N.jpg` are the
same source image (Roboflow renames them again inside `splits/`). The masks are
stored as RLE inside each split's `_annotations.coco.json`. See
`docs/DATASET_CARD.md` for object, collection strategy, and class distribution.

## Large files - Google Drive
> Per Section 2.2, all dataset images and label exports are hosted externally.

- **Raw Dataset Images (81, originals):** https://drive.google.com/drive/folders/1xIZfy29QP5V6S062ttYWHtwFYX65tbnY?usp=sharing
- **Undistorted Dataset Images (81):** https://drive.google.com/drive/folders/1A5up4ismFSmDij8h1h8rnFilsmp6yAFC?usp=sharing
- **Labelled Dataset — COCO, 70/20/10 splits (images + masks + annotations):** https://drive.google.com/drive/folders/1Df8IbTGVSz5snYy91Ljvjl3dpJk55sYc?usp=sharing
- **Label Previews (sample masks drawn on images — train split, 57):** https://drive.google.com/drive/folders/1uVhL6bZ58uZ3dP9xCHeG8szitGGb_I8r?usp=sharing

> _Label previews show a sample of the annotated images (train split) with the
> segmentation masks overlaid, for quick visual review of labelling quality._