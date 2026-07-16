# Camera Calibration Report

## Method
- **Library:** OpenCV
- **Calibration target:** checkerboard (≥ 7×9 inner corners) or ChArUco
- **Number of calibration images:** _TBD (≥ 20, varied angles/distances/positions)_
- **Procedure:** _corner detection → `cv2.calibrateCamera()` → undistortion verification_

## Calibration Images
_Hosted on Google Drive — see `calibration/README.md`. Include a contact sheet / sample here._

## Intrinsic Matrix (K)
```
_TBD_
```

## Distortion Coefficients
```
[k1, k2, p1, p2, k3] = _TBD_
```

## Reprojection Error
- **Mean reprojection error:** _TBD px_  (target: < 0.5 px acceptable, < 0.3 px excellent)

## Undistortion Verification
_Before/after example images demonstrating distortion removal._

## Why undistortion matters for measurement
_Raw (distorted) images bend straight edges, so pixel dimensions near the frame edge are
wrong. All measurement images MUST be undistorted first — see MEASUREMENT_REPORT.md._
