# Homework 4 — Stereo Disparity Map (Uncalibrated)

This folder contains the solution for Homework 4: estimating a relative depth
map from an uncalibrated stereo image pair via fundamental-matrix
estimation, uncalibrated rectification, and Semi-Global Block Matching.

## Contents

- `data/dobosi_peter_laszlo/`: assigned stereo pair (`im0.png` left, `im1.png` right).
- `data/description.pdf`: official assignment description.
- `src/mw79on_submission_hw4/main.py`: complete pipeline implementation.
- `output/`: generated figures and results.
- `PLAN.md`: detailed plan and design decisions for the assignment.

## Requirements

- Python `>=3.10`
- Packages:
  - `opencv-python`
  - `numpy`
  - `matplotlib`

## Environment Setup

Create a venv from the repository root and install all packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e homework_1
pip install -e homework_2
pip install -e homework_3
pip install -e homework_4
```

## Run

From `homework_4/`, run:

```bash
python src/mw79on_submission_hw4/main.py
```

The script executes the full pipeline:

1. **Image Loading**: reads the left/right stereo pair as grayscale.
2. **Keypoint Detection & Matching**: detects features, matches descriptors,
   sorts by distance, and returns paired `(x, y)` arrays as `float32`.
3. **Fundamental Matrix**: estimates `F` with `cv2.findFundamentalMat` and
   `FM_RANSAC`, keeping only inlier point pairs.
4. **Stereo Rectification**: computes rectifying homographies via
   `cv2.stereoRectifyUncalibrated` and warps both images with
   `cv2.warpPerspective`.
5. **Disparity Computation**: runs Semi-Global Block Matching (SGBM) on the
   rectified pair and normalises the result to `uint8` in `[0, 255]`.
6. **Visualisation**: saves a 1×3 figure (rectified left, rectified right,
   colorised disparity map) to `output/`.

Build the PDF report (from `homework_4/`):

```bash
pdflatex -interaction=nonstopmode -halt-on-error REPORT.tex
```
