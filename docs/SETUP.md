# Setup & Installation Guide

## Environment
- **OS:** _TBD_
- **Python:** 3.10+
- **GPU:** _TBD (required for training)_

## Installation
```bash
git clone <repo-url>
cd <repo>
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the large files (Google Drive)
The calibration images, dataset, and model weights are hosted on Google Drive
(see links in the root `README.md`). Download and place them as:
- Calibration images → `calibration/images/`
- Dataset → `dataset/raw/`, `dataset/labels/`, `dataset/splits/`
- Model weights → `models/weights/`

## Running the pipeline
1. **Calibration** — _TBD_
2. **Training** — _TBD_
3. **Inference** — _TBD_
4. **Measurement** — _TBD_

## Docker (optional)
_TBD_
