"""
check_checkerboard.py — Validate checkerboard calibration images BEFORE calibrating.

For each image in --input, tries to detect the inner-corner grid. Reports PASS/FAIL
and writes a visual overlay (detected corners drawn) to --debug so you can eyeball
coverage and detection quality. Run this on your first batch before shooting more.

Board: 9x12 squares @ 20mm  ->  INNER corners = 8 x 11.

Usage:
    python calibration/scripts/check_checkerboard.py \
        --input calibration/images \
        --debug calibration/images/_debug \
        --cols 8 --rows 11
"""
import argparse
import glob
import os

import cv2
import numpy as np


def find_corners(gray, pattern_size):
    """Try fast detector, then the robust SB detector as fallback."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ok:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        return True, corners
    # Fallback: findChessboardCornersSB (more robust to blur/lighting)
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size)
        if ok:
            return True, corners
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="calibration/images", help="folder of images")
    ap.add_argument("--debug", default="calibration/images/_debug", help="overlay output folder")
    ap.add_argument("--cols", type=int, default=8, help="inner corners per row (width)")
    ap.add_argument("--rows", type=int, default=11, help="inner corners per column (height)")
    args = ap.parse_args()

    pattern_size = (args.cols, args.rows)
    os.makedirs(args.debug, exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = sorted({f for e in exts for f in glob.glob(os.path.join(args.input, e))})

    if not files:
        print(f"No images found in {args.input}")
        return

    passed, failed = [], []
    print(f"Checking {len(files)} images for a {args.cols}x{args.rows} inner-corner grid...\n")

    for f in files:
        img = cv2.imread(f)
        name = os.path.basename(f)
        if img is None:
            print(f"  [SKIP] {name:35s}  could not read (HEIC? convert to JPG)")
            failed.append(name)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = find_corners(gray, pattern_size)

        if ok:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, ok)
            cv2.imwrite(os.path.join(args.debug, f"ok_{name}"), vis)
            print(f"  [PASS] {name:35s}  {img.shape[1]}x{img.shape[0]}")
            passed.append(name)
        else:
            print(f"  [FAIL] {name:35s}  corners NOT found")
            failed.append(name)

    print("\n" + "=" * 60)
    print(f"PASS: {len(passed)}/{len(files)}   FAIL: {len(failed)}/{len(files)}")
    if failed:
        print("\nFailed images (reshoot these):")
        for n in failed:
            print(f"  - {n}")
    print(f"\nOverlays saved to: {args.debug}")
    print("Open them and confirm the coloured grid lands on every inner corner.")


if __name__ == "__main__":
    main()
