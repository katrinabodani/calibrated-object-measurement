# Measurement Report

## Methodology
End-to-end flow: capture → `cv2.undistort()` → detect reference object → compute
`pixels_per_mm` → segment target → extract pixel dimensions from mask contour →
convert to millimetres → annotate.

## Reference Object
- **Object:** _TBD (e.g. coin / ID card / ArUco marker)_
- **Known dimensions:** _TBD mm_
- **Detection method:** _TBD_

## Pixel-to-MM Conversion Derivation
```
pixels_per_mm = reference_pixel_size / reference_known_mm
width_mm  = object_pixel_width  / pixels_per_mm
height_mm = object_pixel_height / pixels_per_mm
```
_Full derivation — TBD._

## Calibration Dependency
_Why raw (distorted) images produce incorrect measurements — TBD.
All measurement images MUST be undistorted using the Step 1 intrinsics._

## Accuracy Validation
Measured 10+ instances with a physical ruler/calliper.

| # | Dim | Ground truth (mm) | System output (mm) | Abs error (mm) | % error |
|---|-----|-------------------|--------------------|----------------|---------|
| 1 | W | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| ... | | | | | |

- **Mean Absolute Error (MAE):** _TBD mm_
- **Mean Percentage Error (MPE):** _TBD %_

## Limitations
_TBD._

## End-to-End Usage
_Script/notebook: single image in → mask overlay + width(mm) + height(mm) + confidence out._
