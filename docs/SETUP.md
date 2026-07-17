# Setup & Installation Guide

## 1. Environment
- **OS:** Windows 11 (Linux/macOS also work)
- **Python:** 3.12
- **GPU (recommended for training):** NVIDIA GPU with CUDA. Developed on an
  RTX 4050 Laptop (6 GB), CUDA 12.4. CPU works for inference but training is slow.

## 2. Installation
```bash
git clone <repo-url>
cd <repo>

python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

**PyTorch with CUDA** : install the build matching your GPU (the generic
`torch` on PyPI may be CPU-only). For CUDA 12.4:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Verify: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

## 3. Getting the large files (Google Drive)
Calibration images, the dataset, and the trained weights are hosted on Google
Drive (links in the root `README.md` and each folder's `README.md`). Download and
place them:

| File | Destination |
|------|-------------|
| Calibration images | `calibration/images/` |
| Dataset — raw | `dataset/raw/` |
| Labelled dataset (COCO splits) | `dataset/splits/` (`train/`, `val/`, `test/`) |
| Trained weights (`maskrcnn_medicinebox_best.pth`) | `models/weights/` |

## 4. Running the pipeline

### 4.1 Camera calibration (Step 1)
```bash
# validate checkerboard detection
python calibration/scripts/check_checkerboard.py --input calibration/images
# run intrinsic calibration -> calibration/camera_intrinsics.{npz,yaml}
python calibration/scripts/calibrate.py --input calibration/images \
    --out calibration --cols 8 --rows 11 --square-mm 20.0
# verify undistortion quality
python calibration/scripts/undistort_quality.py --input calibration/images --out calibration
```

### 4.2 Dataset preparation (Step 1)
```bash
# undistort dataset images with Step 1 intrinsics
python dataset/scripts/undistort_images.py --input dataset/raw --output dataset/undistorted
```
Labelling was done in Roboflow (SAM-assisted), exported as COCO into
`dataset/splits/{train,val,test}`.

### 4.3 Model training (Step 2)
```bash
python models/train.py --config models/configs/maskrcnn.yaml
```
Produces `models/weights/…best.pth`, `models/loss_curves.png`,
`models/metrics.json`, and `models/test_predictions/`.

### 4.4 Inference (Step 2)
```bash
python inference/infer.py --image path/to/image.jpg
```
Undistorts → segments → writes an annotated image to `inference/outputs/`.
See `inference/README.md`.

### 4.5 Measurement (Step 3)
```bash
python measurement/measure.py --image measurement/images \
    --card-w 89 --card-h 50 --gt-w 113 --gt-h 50 --box-thickness 22
```
Undistorts → detects the card (scale) → segments the box → outputs width/height in
mm + confidence per image, and (with `--gt-*`) reports MAE/MPE against ruler ground
truth. For a single new image, drop the `--gt-*` args. See `measurement/README.md`.
