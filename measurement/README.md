# Measurement

Pixel-to-mm measurement pipeline and accuracy validation.

## Contents
- `measure.py` — undistort → detect card (scale) → segment box → measure → depth-corrected mm
- `accuracy_report.csv` — per-image results + MAE/MPE (tracked in git)
- `images/` — measurement photos, box + card (**NOT in git** — hosted on Drive)
- `outputs/` — annotated results (**NOT in git** — hosted on Drive)

See `docs/MEASUREMENT_REPORT.md` for methodology, the depth correction, and analysis.

## Usage
```bash
python measurement/measure.py --image measurement/images \
    --card-w 89 --card-h 50 --gt-w 113 --gt-h 50 --box-thickness 22
```
Outputs annotated images (mask + width/height/confidence around the box) and, with
`--gt-*`, the MAE/MPE vs. ruler ground truth.

## Result
**12 images · MAE 1.53 mm · MPE 2.15%** (box front face, GT 113 × 50 mm).

## Large files — Google Drive
> Per Section 2.2, measurement photos and outputs are hosted externally.

- **Measurement Photos (box + card, 12):** https://drive.google.com/drive/folders/1riLU7_6wdC35noKtAQEi17CJ54YjlVxt?usp=sharing
- **Annotated Measurement Outputs:** https://drive.google.com/drive/folders/1dDJJLvi_sjznP6Lqsm1T_Qea5iBP0txQ?usp=sharing