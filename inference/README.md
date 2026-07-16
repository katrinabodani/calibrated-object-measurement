# Inference

Inference scripts and demo outputs.

## Contents
- `infer.py` — accepts an input image, runs **undistortion** (Step 1 intrinsics),
  runs model inference, and outputs annotated results with the detected mask.
- `outputs/` — sample annotated predictions (**NOT in git** — large; a few small samples only)

## Usage
```bash
python inference/infer.py --image <path-to-image>
```
_Full usage guide: see `docs/SETUP.md`. Model weights are hosted on Drive — see `models/README.md`._
