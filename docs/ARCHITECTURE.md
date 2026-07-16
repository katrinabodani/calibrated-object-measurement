# System Architecture, Design Decisions & Module Documentation

## 1. Pipeline Architecture — End-to-End Data Flow

```
                    ┌──────────────────────────────────────────────┐
  Checkerboard  ─▶  │ CALIBRATION  calibrate.py                   |
  images (26)       │   detect corners → cv2.calibrateCamera       │
                    │   → K + distortion coeffs (camera_intrinsics)│
                    └───────────────┬──────────────────────────────┘
                                    │ intrinsics (K, dist)
                                    ▼
  Object photos ─▶ undistort_images ─▶ Roboflow label (COCO)
  (81, JPG)                         (dataset/undistorted) 
                                    ┬ 
                                    │
                                    ▼        
                    ┌─────────────────────────────────────────────┐
                    │ TRAINING  train.py                          │
                    │   COCO splits → Mask R-CNN (ResNet50-FPN)   │
                    │   → best weights (.pth) + metrics           │
                    └───────────────┬─────────────────────────────┘
                                    │ weights
                                    ▼
  New image  ─▶ ┌──────────────────────────────────────────────────────┐
  (box+card)     │ MEASUREMENT  measure.py                              │
                 │  undistort → detect card → pixels_per_mm             │
                 │  → Mask R-CNN segment box → min-area rect (px)       │
                 │  → mm → depth correction → W/H mm + confidence       │
                 └──────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    annotated image + width/height (mm) + accuracy (MAE/MPE)
```

Each stage consumes the previous stage's output; the intrinsics from Step 1 are
used both to undistort the dataset (Step 2) and every measurement image (Step 3).

## 2. Design Decisions & Trade-offs

| Decision | Choice | Rationale / trade-off |
|----------|--------|-----------------------|
| Object | Medicine box | Rigid, flat faces, sharp edges → clean segmentation + unambiguous W×H. |
| Calibration target | 8×11 checkerboard @ 20 mm | Exceeds the 7×9 minimum; printable, no special hardware. |
| Camera resolution | iPhone **12 MP native** (not 24 MP) | 24 MP is a soft computational upscale; native pixels are crisper for corner detection. |
| Labelling tool | Roboflow (SAM-assisted) | Fast polygon labelling + COCO export; used **only** for annotation (model is non-Roboflow). |
| Model | **Mask R-CNN** (torchvision) | Native instance mAP + per-instance confidence (both required); COCO-pretrained fits 81 imgs; avoids Detectron2's Windows build pain. Trade-off: heavier than U-Net. |
| Checkpoint selection | Lowest **val loss** | Val mAP saturates at 1.0 on this easy task, so loss is the discriminating signal. |
| Reference object | Plain card (89×50 mm) | No printer for ArUco; card is flat + known-size. Trade-off: rectangle detection needs a matte, contrasting surface. |
| Box measurement | Min-area rectangle of the mask | Robust to in-plane rotation vs. an axis-aligned box. |
| Depth handling | Distance-based correction | Corrects the flat-card depth offset (`(D−t)/D`) instead of a fiddly raised-card rig. |

## 3. API / Module Documentation

### `calibration/scripts/calibrate.py`
Calibrate from checkerboard images.
```
python calibration/scripts/calibrate.py --input calibration/images --out calibration \
    --cols 8 --rows 11 --square-mm 20.0
# Output: camera_intrinsics.{npz,yaml} (K, dist, image_size), undistort_sample.jpg
```

### `dataset/scripts/undistort_images.py`
`undistort_images(--input, --output, --intrinsics)` → writes undistorted JPGs
(same size, original K). Input: raw JPGs. Output: `dataset/undistorted/*.jpg`.

### `models/train.py`
`python models/train.py --config models/configs/maskrcnn.yaml`
→ `weights/*.pth`, `loss_curves.png`, `metrics.json`, `test_predictions/`.

### `inference/infer.py`
`infer(--image, --weights, --intrinsics)` → annotated mask image.
- **Input:** an image/folder (calibrated camera).
- **Output:** `inference/outputs/<name>_pred.jpg` (mask + box + confidence).
```
python inference/infer.py --image path/to/img.jpg
# stdout: "img.jpg: 1 detection(s), top score 1.00 -> ..._pred.jpg"
```

### `measurement/measure.py`
`measure(--image, --card-w, --card-h, --box-thickness, [--gt-w, --gt-h])`
- **Input:** image/folder (box + card), card size mm, box thickness mm.
- **Output:** `measurement/outputs/<name>_measured.jpg` (mask + W/H mm + conf),
  and with `--gt-*`: `accuracy_report.csv` + MAE/MPE.
```
python measurement/measure.py --image measurement/images \
    --card-w 89 --card-h 50 --gt-w 113 --gt-h 50 --box-thickness 22
# stdout: "1: 112.7 x 49.0 mm (conf 1.00) | GT 113.0x50.0 err 0.3/1.0 mm" ... "MAE = 1.53 mm  MPE = 2.15 %"
```
