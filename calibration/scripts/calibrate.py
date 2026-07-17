"""
calibrate.py - Intrinsic camera calibration from checkerboard images (OpenCV).

Detects the inner-corner grid in every calibration image, runs
cv2.calibrateCamera to estimate the intrinsic matrix K and distortion
coefficients, then reports the reprojection error (overall + per image).
Results are saved to calibration/camera_intrinsics.{npz,yaml} and a
before/after undistortion sample is written for the calibration report.

Board: 9x12 squares @ 20 mm  ->  INNER corners = 8 x 11, square size = 20 mm.

Usage:
    python calibration/scripts/calibrate.py \
        --input calibration/images \
        --out calibration \
        --cols 8 --rows 11 --square-mm 20.0
"""
import argparse
import glob
import os

import cv2
import numpy as np


def find_corners(gray, pattern_size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ok:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        return True, corners
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size)
        if ok:
            return True, corners
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="calibration/images")
    ap.add_argument("--out", default="calibration")
    ap.add_argument("--cols", type=int, default=8, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=11, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, default=20.0, help="square size in mm")
    args = ap.parse_args()

    pattern_size = (args.cols, args.rows)

    # 3D object points for one board: (0,0,0), (1,0,0)... scaled to mm.
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_mm

    exts = ("*.jpg", "*.jpeg", "*.png")
    files = sorted({f for e in exts for f in glob.glob(os.path.join(args.input, e))},
                   key=lambda p: (len(os.path.basename(p)), p))

    objpoints, imgpoints, used = [], [], []
    image_size = None

    # Corner detection at 24MP is slow (~15 s/image), so cache results.
    cache_path = os.path.join(args.out, "_corners_cache.npz")
    cache = None
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        if list(data["files"]) == [os.path.basename(f) for f in files] and \
           tuple(data["pattern"]) == pattern_size:
            cache = data
            print("Loaded corners from cache (delete _corners_cache.npz to re-detect).")

    if cache is not None:
        imgpoints = [c.astype(np.float32) for c in cache["imgpoints"]]
        used = list(cache["used"])
        image_size = tuple(int(x) for x in cache["image_size"])
        objpoints = [objp for _ in imgpoints]
    else:
        print(f"Detecting corners in {len(files)} images...")
        for f in files:
            img = cv2.imread(f)
            if img is None:
                print(f"  [skip] {os.path.basename(f)} (unreadable)")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if image_size is None:
                image_size = gray.shape[::-1]  # (w, h)
            elif gray.shape[::-1] != image_size:
                print(f"  [skip] {os.path.basename(f)} size {gray.shape[::-1]} != {image_size}")
                continue
            ok, corners = find_corners(gray, pattern_size)
            if ok:
                # Normalise shape to (N,1,2) float32 for consistency.
                corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
                objpoints.append(objp)
                imgpoints.append(corners)
                used.append(os.path.basename(f))
                print(f"  [ok]   {os.path.basename(f)}")
            else:
                print(f"  [fail] {os.path.basename(f)} — corners not found")
        np.savez(cache_path,
                 files=np.array([os.path.basename(f) for f in files]),
                 imgpoints=np.array(imgpoints, dtype=object),
                 used=np.array(used), image_size=np.array(image_size),
                 pattern=np.array(pattern_size))

    n = len(objpoints)
    print(f"\nUsing {n} images for calibration (size {image_size}).")
    if n < 10:
        raise SystemExit("Not enough valid images (need >= 10). Capture more.")

    # Calibrate: 5-term distortion model (k1,k2,p1,p2,k3) — radial + tangential.
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Per-image reprojection error.
    per_image = []
    total_err, total_pts = 0.0, 0
    for i in range(n):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        proj = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        pts = np.asarray(imgpoints[i], dtype=np.float64).reshape(-1, 2)
        d2 = np.sum((pts - proj) ** 2, axis=1)  # squared pixel distance per corner
        per_image.append((used[i], float(np.sqrt(np.mean(d2)))))
        total_err += float(np.sum(d2))
        total_pts += len(d2)
    mean_reproj = np.sqrt(total_err / total_pts)

    # ---- Report ----
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Image size (w x h): {image_size[0]} x {image_size[1]}")
    print(f"RMS reprojection error (cv2):     {rms:.4f} px")
    print(f"Mean reprojection error (per-pt): {mean_reproj:.4f} px")
    verdict = "EXCELLENT (<0.3)" if mean_reproj < 0.3 else \
              "ACCEPTABLE (<0.5)" if mean_reproj < 0.5 else "TOO HIGH (>=0.5)"
    print(f"Verdict: {verdict}")
    print("\nIntrinsic matrix K:")
    print(K)
    print(f"\nfx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print("\nDistortion coefficients [k1, k2, p1, p2, k3]:")
    print(dist.ravel())

    worst = sorted(per_image, key=lambda x: -x[1])[:5]
    print("\nWorst 5 images by reprojection error:")
    for name, e in worst:
        print(f"  {name:12s} {e:.4f} px")

    # ---- Save ----
    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, "camera_intrinsics.npz")
    np.savez(npz_path, K=K, dist=dist, image_size=np.array(image_size),
             rms=rms, mean_reproj=mean_reproj,
             square_mm=args.square_mm, pattern=np.array(pattern_size))

    yaml_path = os.path.join(args.out, "camera_intrinsics.yaml")
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", dist)
    fs.write("rms_reprojection_error", rms)
    fs.write("mean_reprojection_error", mean_reproj)
    fs.write("num_images", n)
    fs.write("square_size_mm", args.square_mm)
    fs.release()

    print(f"\nSaved: {npz_path}")
    print(f"Saved: {yaml_path}")

    # ---- Undistortion sample (before/after) ----
    sample = cv2.imread(files[0])
    h, w = sample.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0, newImgSize=(w, h))
    undist = cv2.undistort(sample, K, dist, None, newK)
    combo = np.hstack([sample, undist])
    combo = cv2.resize(combo, (combo.shape[1] // 4, combo.shape[0] // 4))
    sample_path = os.path.join(args.out, "undistort_sample.jpg")
    cv2.imwrite(sample_path, combo)
    print(f"Saved: {sample_path}  (left=original, right=undistorted)")


if __name__ == "__main__":
    main()
