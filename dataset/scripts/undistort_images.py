"""
undistort_images.py - Undistort dataset images with the Step 1 intrinsics.

The measurement/inference pipeline undistorts every input, so the model must be
trained on undistorted images too. This script undistorts each raw image with the
stored camera matrix K and distortion coefficients, keeping the original K as the
new camera matrix (same image size, no re-scaling of the intrinsics).

Usage:
    python dataset/scripts/undistort_images.py \
        --input dataset/raw --output dataset/undistorted \
        --intrinsics calibration/camera_intrinsics.npz
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset/raw")
    ap.add_argument("--output", default="dataset/undistorted")
    ap.add_argument("--intrinsics", default="calibration/camera_intrinsics.npz")
    args = ap.parse_args()

    data = np.load(args.intrinsics)
    K, dist = data["K"], data["dist"]
    calib_size = tuple(int(x) for x in data["image_size"])  # (w, h) used in calibration

    os.makedirs(args.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input, "*.jpg")))
    if not files:
        print(f"No .jpg images in {args.input}")
        return

    print(f"Undistorting {len(files)} images (calib size {calib_size})...")
    done, skipped = 0, 0
    for f in files:
        img = cv2.imread(f)
        h, w = img.shape[:2]
        if (w, h) != calib_size:
            print(f"  [skip] {os.path.basename(f)} size {(w, h)} != calib {calib_size}")
            skipped += 1
            continue
        # Undistort keeping original K -> same size, intrinsics unchanged downstream.
        und = cv2.undistort(img, K, dist, None, K)
        out = os.path.join(args.output, os.path.basename(f))
        cv2.imwrite(out, und, [cv2.IMWRITE_JPEG_QUALITY, 95])
        done += 1

    print(f"Done. {done} undistorted -> {args.output}" + (f"  ({skipped} skipped)" if skipped else ""))


if __name__ == "__main__":
    main()
