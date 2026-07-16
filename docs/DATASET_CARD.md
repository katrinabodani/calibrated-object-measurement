# Dataset Card

## Object Chosen
- **Object:** Pharmaceutical carton — a small "Crystolife" medicine box (rigid
  rectangular cardboard box with printed graphics).
- **Approximate real-world dimensions:** _width × height × depth in mm — TBD
  (to be measured with a ruler/calliper; used as ground truth in Step 3)._
- **Justification:**
  - **Geometry** — rigid rectangular box with flat faces and sharp, well-defined
    edges → segments cleanly and gives unambiguous width/height to measure.
  - **Labelling ease** — high-contrast printed graphics (yellow/red/white) make
    the silhouette easy to distinguish from most backgrounds.
  - **Measurability** — small enough to fit in-frame alongside a reference object
    and to measure precisely with a ruler, yet large enough to segment reliably.
  - **Availability** — a common, consistent everyday object.
- **Measured face:** the large front face (width × height) is the target face for
  the Step 3 metric measurement.

## Collection Strategy
- **Camera:** iPhone main camera — **12 MP native, landscape (4032 × 3024)**,
  identical configuration to the Step 1 calibration (required for the calibration
  to apply). Transferred via USB.
- **Number of images:** target ~80 captured → **≥ 70 labelled** _(final count TBD)._
- **Variation:** viewpoint (front/top/3-4/tilted), box orientation (standing/flat/
  on-side), distance (close/far), background (multiple surfaces), and lighting.
- **All images undistorted:** **Yes** — undistorted with the Step 1 intrinsics
  before labelling, so the training distribution matches the inference pipeline
  (which undistorts every input).

## Labelling
- **Tool:** _TBD (Roboflow or CVAT — annotation tool only; note the model-architecture
  restriction bans Roboflow/YOLO models, not the labelling tool)._
- **Annotation type:** instance segmentation (polygon masks).
- **Classes:** 1 — `box`.

## Statistics
| Split | Images | % |
|-------|--------|---|
| Train | _TBD_ | 70% |
| Val   | _TBD_ | 20% |
| Test  | _TBD_ | 10% |

- **Class distribution:** single class (`box`), one instance per image _(TBD)._

## Access
_Full dataset (raw images, labels, splits) hosted on Google Drive — see
`dataset/README.md`._
