"""
undistort_quality.py — Directly measure how well the calibration undistorts, by
checking that checkerboard rows/columns become STRAIGHT lines after undistortion.

A physically straight line (a checkerboard row/column) projects to a straight
line under an ideal pinhole camera. Any residual curvature after undistortion is
leftover lens distortion. This measures undistortion quality directly, in pixels,
independent of the reprojection-error metric.

Reports mean straight-line deviation of corners BEFORE vs AFTER undistortion.

Usage:
    python calibration/scripts/undistort_quality.py --input calibration/images --out calibration
"""
import argparse
import glob
import os

import cv2
import numpy as np


def line_residual(points):
    """RMS perpendicular distance of points from their best-fit line (px)."""
    p = np.asarray(points, np.float64)
    c = p.mean(0)
    u, s, vt = np.linalg.svd(p - c)
    normal = vt[1]                       # direction perpendicular to best-fit line
    d = (p - c) @ normal
    return float(np.sqrt(np.mean(d ** 2)))


def grid_straightness(corners, cols, rows):
    """Mean line residual over all rows and all columns of the corner grid."""
    g = corners.reshape(rows, cols, 2)
    res = []
    for r in range(rows):
        res.append(line_residual(g[r, :, :]))     # each row of corners
    for c in range(cols):
        res.append(line_residual(g[:, c, :]))     # each column of corners
    return np.mean(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="calibration/images")
    ap.add_argument("--out", default="calibration")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=11)
    args = ap.parse_args()

    data = np.load(os.path.join(args.out, "camera_intrinsics.npz"))
    K, dist = data["K"], data["dist"]
    ps = (args.cols, args.rows)

    files = sorted(glob.glob(os.path.join(args.input, "*.jpg")),
                   key=lambda p: (len(os.path.basename(p)), p))[:12]  # sample 12

    before, after = [], []
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        und = cv2.undistort(img, K, dist, None, K)
        for tag, im, store in [("dist", img, before), ("undist", und, after)]:
            ok, c = cv2.findChessboardCorners(
                im, ps, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                c = cv2.cornerSubPix(im, c, (11, 11), (-1, -1), term)
                store.append(grid_straightness(c.reshape(-1, 2), *ps))

    print(f"Straight-line deviation of checkerboard rows/columns (px), n={len(after)} images:\n")
    print(f"  BEFORE undistortion (raw):   mean {np.mean(before):.3f} px   max {np.max(before):.3f} px")
    print(f"  AFTER  undistortion:         mean {np.mean(after):.3f} px   max {np.max(after):.3f} px")
    improvement = (1 - np.mean(after) / np.mean(before)) * 100
    print(f"\n  Distortion removed: {improvement:.1f}% straighter after undistortion.")
    print(f"  (Lower 'after' = better. This is the real measure of undistortion quality.)")


if __name__ == "__main__":
    main()
