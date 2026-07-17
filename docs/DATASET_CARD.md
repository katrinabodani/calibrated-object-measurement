# Dataset Card

## Object Chosen
- **Object:** : Pharmaceutical carton, a small medicine box (rigid
  rectangular cardboard box with printed graphics).
- **Real-world dimensions (front face, ruler-measured):** **113 × 50 mm** 
  (width × height). This face is the Step 3 measurement target / ground truth.
  (Depth ≈ 20 mm, not measured for the 2D width×height task.)
- **Justification:**
  - **Geometry** : rigid rectangular box with flat faces and sharp, well-defined
    edges → segments cleanly and gives unambiguous width/height to measure.
  - **Labelling ease** : high-contrast printed graphics (yellow/red/white) make
    the silhouette easy to distinguish from most backgrounds.
  - **Measurability** : small enough to fit in-frame alongside a reference object
    and to measure precisely with a ruler, yet large enough to segment reliably.
  - **Availability** : a common, consistent everyday object.
- **Measured face:** the large front face (width × height) is the target face for
  the Step 3 metric measurement.

## Collection Strategy
- **Camera:** iPhone main camera : **12 MP native, landscape (4032 × 3024)**,
  identical configuration to the Step 1 calibration (required for the calibration
  to apply). Transferred via USB.
- **Number of images:** 81 captured -→ **81 labelled**.
- **Variation:** viewpoint (front/top/3-4/tilted), box orientation (standing/flat/
  on-side), distance (close/far), background (multiple surfaces), and lighting.
- **All images undistorted:** **Yes** - undistorted with the Step 1 intrinsics
  before labelling, so the training distribution matches the inference pipeline
  (which undistorts every input).

## Labelling
- **Tool:** **Roboflow** : SAM-assisted "Smart Polygon" annotation. Used as an
  **annotation tool only**; the model architecture (Step 2) is non-Roboflow /
  non-YOLO per the rules.
- **Annotation type:** instance segmentation. Exported as **COCO Segmentation**
  with **RLE masks** (SAM output).
- **Classes:** 1 : `medicine_box` (COCO also carries an index-0 dummy supercategory,
  standard Roboflow behaviour).

## Statistics
| Split | Images | % |
|-------|--------|---|
| Train | 57 | 70% |
| Val   | 16 | 20% |
| Test  | 8  | 10% |
| **Total** | **81** | 100% |

- **Class distribution:** single class (`medicine_box`), exactly one instance per
  image (81 masks total). Split performed in Roboflow (random).

## Access
_Full dataset (raw images, labels, splits) hosted on Google Drive, see
`dataset/README.md`._