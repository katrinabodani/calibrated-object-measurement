# Measurement Report

## 1. Methodology

End-to-end pixel-to-mm measurement of the box, per image:

1. **Undistort** with the Step 1 intrinsics (`cv2.undistort`). Portrait captures
   are rotated to the calibrated landscape orientation first.
2. **Detect the reference card** (a plain card, ruler-measured **89 × 50 mm**) as a
   solid rectangle (Otsu threshold, both polarities, + rectangularity + aspect
   filter), excluding the box region.
3. **Compute `pixels_per_mm`** by averaging the card's four edge lengths against
   its known size (orientation-agnostic).
4. **Segment the box** with the trained Mask R-CNN (best detection).
5. **Measure** the box via a **min-area rectangle** fit to the mask → side lengths
   in pixels (robust to in-plane rotation).
6. **Convert to mm** with `pixels_per_mm`, then apply the **depth correction**
   (Section 3).

Pipeline: `measurement/measure.py`.

## 2. Pixel-to-MM Conversion Derivation
```
pixels_per_mm = card_edge_pixels / card_edge_mm      (averaged over 4 edges)
box_side_mm   = box_side_pixels / pixels_per_mm
```

## 3. Calibration Dependency & Depth Correction

**Why undistortion is mandatory:** raw (distorted) images bend straight edges,
strongest near the frame periphery, so uncorrected pixel dimensions are
position-dependent and wrong. All measurement images are undistorted with the
Step 1 intrinsics before any measurement.

**Depth-offset correction:** the card lies flat on the surface, but the box's top
face (what the mask captures top-down) sits `t = 22 mm` (box thickness) closer to
the camera, inflating its measured size by `D / (D − t)`. The camera–card
distance is estimated from the card itself: `D = f / pixels_per_mm` (f = mean
focal length in px), and each measurement is rescaled by `(D − t) / D`. This
removed a consistent ~8% overestimate.

| | MAE | MPE |
|--|-----|-----|
| Before correction | ~7.3 mm | ~8.9 % |
| **After correction** | **1.53 mm** | **2.15 %** |

## 4. Reference Object
- Plain card (no personal data, for privacy). Ruler-measured **89 × 50 mm**.
- Placed flat on the same surface as the box, both fully visible.

## 5. Accuracy Validation

Ground truth (ruler): box front face **113 × 50 mm**. **12 images, 24 measurements**
(long + short side each), across varied surfaces/positions. System vs. ground truth:

- **Mean Absolute Error (MAE): 1.53 mm**
- **Mean Percentage Error (MPE): 2.15 %**
- Per-image long-side output ranged ~111–116 mm (GT 113); short-side ~48–52 mm (GT 50).

Full per-image data: `measurement/accuracy_report.csv`.

## 6. End-to-End Demo
`measurement/measure.py` takes an image (or folder) and outputs an annotated image
with the **mask overlay**, **width (mm)**, **height (mm)**, and **confidence**
drawn around the box (`measurement/outputs/<n>_measured.jpg`):
```bash
python measurement/measure.py --image measurement/images \
    --card-w 89 --card-h 50 --gt-w 113 --gt-h 50 --box-thickness 22
```

## 7. Assumptions & Limitations
- **Reference must be coplanar / depth known:** accuracy relies on the box
  thickness (22 mm) for the depth correction; an unknown-thickness object would
  reintroduce the offset.
- **Card detection needs contrast:** fails on glossy/reflective surfaces (glare
  mimics rectangles); a matte surface is required.
- **Fronto-parallel assumption:** the measured face must be roughly parallel to
  the sensor; strong tilt reintroduces perspective error.
- **Residual ~1.5 mm error** stems from mask-edge quantisation, card-detection
  precision, and the ~2.35 px calibration reprojection error (see CALIBRATION_REPORT).
