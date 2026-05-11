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

### Homework 5 — Transfer-Learning Image Classification (PyTorch)

- `homework_5/src/mw79on_submission_hw5/main.py`: Main solution script (MaxViT-T + DTD).
- `homework_5/data/description.pdf`: Official assignment description.
- `homework_5/README.md`: Run instructions, model/dataset choice rationale, and notes.
- `homework_5/REPORT.tex`: LaTeX source for the submission report.
- `homework_5/REPORT.pdf`: Compiled submission report.
- `homework_5/output/training_curves.png`: Training and validation loss/accuracy curves.
- `homework_5/output/confusion_matrix.png`: Test-set confusion matrix.
- `homework_5/output/training_output`: Kaggle training log with final metrics.

### Homework 6 — YOLO26 Object Detection Comparison

- `homework_6/src/mw79on_submission_hw6/main.py`: Main solution script (predict + val pipeline across YOLO26 sizes).
- `homework_6/data/description.pdf`: Official assignment description.
- `homework_6/README.md`: Run instructions, environment notes, and dataset/weights extraction steps.
- `homework_6/REPORT.tex`: LaTeX source for the submission report.
- `homework_6/REPORT.pdf`: Compiled submission report (2 pages).
- `homework_6/output/montages/yolo26{n,s,m,l,x}.png`: 4×4 prediction montages, one per model size, on a fixed seeded test sample.
- `homework_6/output/predictions/yolo26{n,s,m,l,x}/`: per-image annotated PNGs backing the montages.
- `homework_6/output/val_runs/yolo26{n,s,m,l,x}/`: Ultralytics' validation artefacts (P/R/F1/PR curves and confusion matrix per model).
- `homework_6/output/metrics.csv` / `output/metrics.md`: comparison table (mAP@0.5, mAP@0.75, mAP@0.5:0.95, runtime).
- `homework_6/output/run_log.txt`: captured stdout from the full pipeline run.

## Quick Start

For setup and execution:

- Homework 1: `homework_1/README.md`
- Homework 2: `homework_2/README.md`
- Homework 3: `homework_3/README.md`
- Homework 4: `homework_4/README.md`
- Homework 5: `homework_5/README.md`
- Homework 6: `homework_6/README.md`

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
pip install -e homework_5
pip install -e homework_6
```

This keeps dependencies for all assignments in a single reproducible environment.

## Note on Method Documentation Consistency

Homeworks 1–4 are Python OpenCV based; some classroom examples and references mix API variants or detector/matcher combinations across OpenCV ecosystems, so implementations here are kept explicit and Python OpenCV compatible. Homework 5 switches stacks to PyTorch + torchvision for deep-learning transfer learning. Homework 6 uses the Ultralytics YOLO API on top of PyTorch.

- Homework 1: histogram-based enhancement with Python OpenCV color conversions.
- Homework 2: ORB binary descriptors with BFMatcher (Hamming), Lowe-ratio filtering, and MAD-based outlier rejection for robust localization.
- Homework 3: Gaussian denoising, WebP compression, and quality metrics (MSE, PSNR, SSIM).
- Homework 4: SIFT + FLANN matching, RANSAC fundamental-matrix estimation, uncalibrated rectification, and SGBM disparity computation.
- Homework 5: PyTorch transfer learning — MaxViT-T (Google, ECCV 2022) fine-tuned on the Describable Textures Dataset (DTD, 47 classes) with ImageNet-pretrained weights.
- Homework 6: Ultralytics YOLO26 inference + validation across the five pretrained sizes (n/s/m/l/x) on the African Wildlife dataset; pretrained weights only, no training.
