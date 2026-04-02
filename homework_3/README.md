# Homework 3 — Noise Reduction, Image Compression & Quality Measurement

This folder contains the solution for Homework 3 (Gaussian denoising, JPEG/WebP compression, and image quality assessment using MSE/PSNR/SSIM).

## Contents

- `data/clean/`: 10 clean reference images (Kodak dataset, `kodim01`–`kodim10`).
- `data/dobosi_peter_laszlo_norm_dist/`: 10 corresponding noisy images (Gaussian noise, `kodim00_noisy`–`kodim09_noisy`).
- `data/description.pdf`: Official assignment description and formal requirements.
- `src/mw79on_submission_hw3/main.py`: Complete pipeline implementation.
- `output/`: Generated figures and results.

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
```

## Run

From `homework_3/`, run:

```bash
python src/mw79on_submission_hw3/main.py
```

The script executes the full pipeline:

1. **Noise Reduction**: Loads clean/noisy image pairs, applies Gaussian filtering to reduce normally-distributed noise.
2. **Image Compression**: Compresses images using JPEG or WebP at varying quality levels.
3. **Quality Measurement**: Computes MSE, PSNR and SSIM between reference and processed images.
4. **Visualization**: Generates comparison figures saved to `output/`.

Build the PDF report (from `homework_3/`):

```bash
pdflatex -interaction=nonstopmode -halt-on-error REPORT.tex
```
