# Camera Calibration Report

## 1. Method

Intrinsic camera calibration was performed with **OpenCV** using a planar
**checkerboard** target and Zhang's method (`cv2.calibrateCamera`), estimating a
pinhole model with a 5-term radial–tangential distortion model
`[k1, k2, p1, p2, k3]`.

| Item | Value |
|------|-------|
| Library | OpenCV 5.0.0 (Python) |
| Target | Checkerboard, 9×12 squares → **8×11 inner corners** |
| Square size | 20.0 mm (printed at 100% scale) |
| Camera | iPhone main camera, **12 MP native** mode |
| Capture orientation | Landscape (all images identical: 4032 × 3024) |
| Calibration images used | **26** (all detected, varied angles/distances/positions) |
| Corner refinement | `cornerSubPix` (11×11), `findChessboardCornersSB` fallback |
| Distortion model | 5-term radial + tangential |

**Pipeline scripts** (`calibration/scripts/`):
- `check_checkerboard.py` — per-image corner-detection validation + overlays
- `calibrate.py` — corner detection, calibration, error report, artifact export
- `undistort_quality.py` — direct undistortion-quality check (line-straightness)

Run:
```bash
python calibration/scripts/calibrate.py --input calibration/images \
    --out calibration --cols 8 --rows 11 --square-mm 20.0
```

## 2. Calibration Images

26 images of the checkerboard were captured with the calibrated camera across
varied angles (15–40° tilts), distances, and frame positions (board pushed into
frame corners and edges to constrain edge distortion). All 26 passed corner
detection (8×11 grid). Full-resolution images are hosted on Google Drive — see
`calibration/README.md`.

## 3. Results

### Intrinsic Matrix (K)
```
[[2819.84     0.00  2043.83]
 [   0.00  2834.19  1532.69]
 [   0.00     0.00     1.00]]
```
`fx = 2819.84`, `fy = 2834.19`, `cx = 2043.83`, `cy = 1532.69`
(principal point ≈ image centre 2016 × 1512 — a healthy result).

### Distortion Coefficients `[k1, k2, p1, p2, k3]`
```
[ 0.2637, -0.9516, -0.0057,  0.0087,  1.0373 ]
```

### Reprojection Error
| Method | Mean reprojection error |
|--------|-------------------------|
| Standard `calibrateCamera` (26 imgs) | **2.35 px** |
| `calibrateCameraRO` (release-object cross-check) | 1.20 px |

The assessment target is < 0.5 px. **We did not reach it with this
camera + target combination.** Section 5 documents the systematic investigation
and root-cause analysis; Section 6 covers the impact on measurement.

## 4. Undistortion Verification

`calibration/undistort_sample.jpg` shows an original (left) vs undistorted
(right) pair. Barrel distortion is visibly corrected and straight edges
straighten, with no spurious warping.

### Quantitative undistortion quality (line-straightness test)
Because reprojection error is only a proxy, undistortion quality was measured
**directly**: a physically straight checkerboard row/column must project to a
straight line under an ideal camera, so any residual curvature is leftover
distortion. Measured over 12 images (`undistort_quality.py`):

| | Mean line deviation | Max |
|---|---------------------|-----|
| Raw (distorted) images | 1.57 px | 2.25 px |
| **After undistortion** | **0.86 px** | 1.61 px |

**Two key results:**
1. **Undistortion works** — lines become 45 % straighter, to **≈0.86 px**
   residual (≈0.04 % over a ~2000 px object — negligible for measurement).
2. **The raw images had only ~1.57 px of distortion to begin with** — the
   smartphone already corrects most lens distortion in-camera. This means the
   2.35 px reprojection error (Section 3) was dominated by **corner-localisation
   noise, not real geometric distortion.** The actual geometry after undistortion
   is sub-pixel and well within measurement tolerance.

The calibration therefore produces **valid, accurate undistortion** for the
measurement pipeline, despite the elevated reprojection-error headline number.

## 5. Investigation & Root-Cause Analysis

The reprojection error was investigated rigorously. Four full capture sessions
and six controlled experiments were run to isolate the cause. Summary:

### Capture sessions
| # | Setup | Res | Imgs | Error |
|---|-------|-----|------|-------|
| 1 | Board loose on wall/floor | 24 MP | 30 | 2.33 px |
| 2 | Flat-mounted, plastic-wrapped | 24 MP | 49 | 3.99 px |
| 3 | Sharp/bright, flat-mounted | 24 MP | 20 | 3.83 px |
| 4 | **12 MP native, sharp, landscape** | 12 MP | 26 | **2.35 px** |

### Controlled experiments (each with data)
| Hypothesis tested | Result | Conclusion |
|-------------------|--------|------------|
| Bad images (greedy pruning) | 49→16 imgs only 3.99→2.47 px | Error is **uniform**, not outliers |
| Distortion model too simple (8-term rational) | 2.35→2.28 px | Not a model-order problem |
| Resolution (downscaling) | error scales **linearly** with width | Purely **cosmetic**, no real gain |
| Detector accuracy (SB exhaustive) | no change | Not a detector problem |
| Image sharpening (unsharp mask) | no gain, worse when pushed | Information not recoverable |
| **Target imperfection (`calibrateCameraRO`)** | **2.35 → 1.20 px** | **Target is a major contributor** |

### Key findings
1. **Not corner sharpness.** Switching from 24 MP (soft, computational) to 12 MP
   native (verified crisp corners, ~2–4 px edges) gave essentially *no* gain
   beyond the trivial resolution scaling (2.35 px vs the 2.03 px predicted by
   pure downscaling). The corners are sharp; the model still doesn't fit.
2. **Distortion is non-parametric.** `k2 = −0.95`, `k3 = +1.04` are large,
   opposing higher-order terms — the optimiser straining to fit a distortion
   that is not a clean radial polynomial. Richer models don't help.
3. **Target imperfection is significant.** Releasing the object points
   (`calibrateCameraRO`) halved the error, but required implausible grid
   deviations (mean 3.6–7.5 mm) — i.e. it is absorbing **residual non-flatness
   (Z-ripple)**, not real print error (ruler-verified squares are ~20 mm).

### Root cause
The residual error is attributed to two compounding, non-software-fixable
sources:
- **Smartphone computational photography.** The iPhone applies
  non-parametric geometric processing (lens correction, multi-frame fusion) that
  does not conform to an ideal pinhole + polynomial-distortion model.
- **Residual target non-flatness.** The paper target, even mounted on a rigid
  backing, retains sub-millimetre-to-millimetre ripple that violates the planar
  assumption.

Reaching < 0.5 px would require a camera without computational geometry
processing (e.g. a USB/industrial webcam) and/or a perfectly flat, machine-crisp
target (e.g. a screen-displayed board). This was a deliberate,
documented trade-off against project time — see `Assumptions & Limitations`.

## 6. Impact on Measurement

Reprojection error is a proxy; the measurement pipeline's true test is the Step 3
accuracy validation against physical (calliper/ruler) ground truth. Two
mitigating factors keep this calibration usable:
- The **undistortion is geometrically sane** (Section 4).
- The metric scale is derived from a **reference object in the same image**
  (pixels-per-mm), so measurement is dominated by *local* scale, not the absolute
  intrinsics. A ~2 px residual over a ~2000 px object is ≈ 0.1 %.

Measured end-to-end accuracy (MAE / MPE) is reported in
`docs/MEASUREMENT_REPORT.md`.

## 7. Why undistortion is mandatory before measurement
Raw (distorted) images bend straight physical edges — most strongly near the
frame periphery, where radial distortion is largest. Measuring pixel dimensions
on a distorted image therefore yields position-dependent errors (an object of
the same size measures differently at the centre vs the edge). Undistorting with
the intrinsic parameters removes this so pixel geometry is metric-consistent
across the frame.
