# Advanced Image Processing Methods

This [repository](https://github.com/dobosipeter/advanced-image-processing-methods) contains my homework submissions for the Advanced Image Processing Methods course.

## Repository Navigation

### Homework 1 — Histogram-Based Image Enhancement

- `homework_1/src/mw79on_submission/main.py`: Main solution script.
- `homework_1/data/description.pdf`: Official assignment description.
- `homework_1/README.md`: Run instructions and reproducibility notes.
- `homework_1/RESULTS.md`: Method summary and outputs.
- `homework_1/output/task4_results.png`: Generated final 2×3 comparison plot.

### Homework 2 — Feature-Based ROI Localization

- `homework_2/src/mw79on_submission_hw2/main.py`: Main solution script.
- `homework_2/data/description.pdf`: Official assignment description.
- `homework_2/README.md`: Run instructions and notes.
- `homework_2/REPORT.tex`: LaTeX source for the submission report.
- `homework_2/REPORT.pdf`: Compiled submission report.
- `homework_2/output/final_localization.png`: Final localization image with ROI bounding rectangles.
- `homework_2/output/matches/`: Per-ROI matching debug figures.

### Homework 3 — Noise Reduction, Image Compression & Quality Measurement

- `homework_3/src/mw79on_submission_hw3/main.py`: Main solution script.
- `homework_3/data/description.pdf`: Official assignment description.
- `homework_3/README.md`: Run instructions and notes.
- `homework_3/data/clean/`: Clean reference images (Kodak dataset).
- `homework_3/data/dobosi_peter_laszlo_norm_dist/`: Noisy image variants (Gaussian noise).
- `homework_3/output/`: Generated figures and results.

### Homework 4 — Stereo Disparity Map (Uncalibrated)

- `homework_4/src/mw79on_submission_hw4/main.py`: Main solution script.
- `homework_4/data/description.pdf`: Official assignment description.
- `homework_4/data/dobosi_peter_laszlo/`: Assigned stereo pair (`im0.png`, `im1.png`).
- `homework_4/README.md`: Run instructions and notes.
- `homework_4/REPORT.tex`: LaTeX source for the submission report.
- `homework_4/REPORT.pdf`: Compiled submission report.
- `homework_4/output/disparity_results.png`: Final 1×3 figure (rectified pair + disparity map).

## Quick Start

For setup and execution:

- Homework 1: `homework_1/README.md`
- Homework 2: `homework_2/README.md`
- Homework 3: `homework_3/README.md`
- Homework 4: `homework_4/README.md`

## Python Environment

Use one shared virtual environment at repository root for all homeworks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e homework_1
pip install -e homework_2
pip install -e homework_3
pip install -e homework_4
```

This keeps dependencies for all assignments in a single reproducible environment.

## Note on Method Documentation Consistency

Some classroom examples and references mix API variants or detector/matcher combinations across OpenCV ecosystems.
In this repository, implementations are kept explicit and Python OpenCV compatible:

- Homework 1: histogram-based enhancement with Python OpenCV color conversions.
- Homework 2: ORB binary descriptors with BFMatcher (Hamming), Lowe-ratio filtering, and MAD-based outlier rejection for robust localization.
- Homework 3: Gaussian denoising, WebP compression, and quality metrics (MSE, PSNR, SSIM).
- Homework 4: SIFT + FLANN matching, RANSAC fundamental-matrix estimation, uncalibrated rectification, and SGBM disparity computation.
