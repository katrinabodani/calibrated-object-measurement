# Measurement

Pixel-to-mm measurement pipeline and accuracy report.

## Contents
- `measure.py` — end-to-end: undistort → detect reference object → `pixels_per_mm`
  → segment target → extract mask dimensions → output width (mm), height (mm), confidence.
- `outputs/` — annotated measurement results (**NOT in git** — a few small samples only)

## Usage
```bash
python measurement/measure.py --image <path-to-image>
```
_Methodology, accuracy table, and error analysis: see `docs/MEASUREMENT_REPORT.md`._
