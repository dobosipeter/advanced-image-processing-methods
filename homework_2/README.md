# Homework 2 - ROI Localization with Local Features

This folder contains a scaffold for Homework 2 (noise-aware preprocessing, keypoint descriptors, feature matching, and ROI localization on the full image).

## Contents

- `data/noisy_motherboard.jpg`: Noisy full-size input image.
- `data/roi/dobosi_peter/*.jpg`: Dynamic ROI query images for this student.
- `data/description.pdf`: Official assignment description and formal requirements.
- `src/mw79on_submission_hw2/main.py`: Pipeline skeleton with TODO checkpoints.
- `output/final_localization.png`: Target final image with found ROI rectangles (generated after the TODOs are completed).
- `output/matches/`: Target folder for per-ROI matching visualizations.
- `RESULTS.md`: Work-in-progress tracker and notes.
- `REPORT.tex`: Submission report template.

## Requirements

- Python `>=3.10`
- Packages:
  - `opencv-python`
  - `numpy`
  - `matplotlib`

## Environment Setup

Create a venv from the repository root and install both packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e homework_1
pip install -e homework_2
```

## Run the Scaffold

From `homework_2/`, run:

```bash
python src/mw79on_submission_hw2/main.py
```

The script currently provides:

1. **Subtask 1 (Preprocessing)**: Loads the scene and all ROI images, converts each to grayscale, and applies Gaussian blur (5×5) denoising. Optional unsharp-mask sharpening is available via the `sharpen` flag.
2. **Subtask 2 (Keypoints & Descriptors)**: Creates an ORB detector (`nfeatures=3000`) and computes keypoints + descriptors for the denoised scene and every ROI image.
3. **Subtask 3–4 (TODO)**: Function stubs for BFMatcher matching, ROI centre estimation, bounding-box drawing, and pair-level debug visualization.

Expected outputs after implementing TODOs:

- `output/final_localization.png`
- `output/matches/roi_X_match.png`

## Notes on Documentation Consistency

- Some classroom examples mix APIs and matcher recommendations across OpenCV variants. This implementation uses Python OpenCV APIs only (`cv2.ORB_create`, `cv2.BFMatcher`, `cv2.drawMatches`, `cv2.rectangle`) and keeps detector/descriptor compatibility explicit.
- The assignment allows either FLANN or BF matcher. ORB descriptors are binary, so BF with Hamming distance is used here as the robust default.

Build the PDF report template (from `homework_2/`):

```bash
pdflatex -interaction=nonstopmode -halt-on-error REPORT.tex
```
