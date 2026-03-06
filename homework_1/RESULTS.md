# Homework 1 Results and Solution Notes

## Assignment Tasks Implemented

## 1. Preprocessing

- Loaded input image with OpenCV from `data/dobosi_peter_laszlo.jpg`.
- Rotated image by `-9.41` degrees to align text horizontally.
- Applied scale during rotation to reduce black edge regions.

Implementation reference: `src/mw79on_submission/main.py` (`preprocessing`).

## 2. Global Histogram Equalization

- Converted preprocessed image from BGR to HSV.
- Performed global histogram equalization on the V channel only.
- Blended original and equalized V channels with `0.5 / 0.5` weights to reduce over-contrast.
- Converted back to BGR for visualization.

Implementation reference: `src/mw79on_submission/main.py` (`global_histogram_equalization`).

## 3. CLAHE

- Converted image to HSV and applied CLAHE to V channel.
- Parameter choice used in this solution:
  - `clipLimit=1.205`
  - `tileGridSize=(9, 9)`
- Converted result back to BGR.

Implementation reference: `src/mw79on_submission/main.py` (`apply_clahe`).

## 4. Visualization (Task 4)

- Built a single `2x3` subplot figure:
  - Top row: rotated/cropped, global equalized, CLAHE images.
  - Bottom row: corresponding RGB histograms.
- Histogram plotting includes:
  - Separate `R`, `G`, and `B` channels.
  - Intensity axis explicitly scaled to `0-255`.
  - Titles, labels, legend, and grid for readability.
- Saved final plot to `output/task4_results.png`.

Implementation reference: `src/mw79on_submission/main.py` (`plot_results`).

## Output Artifact

- Final figure: `output/task4_results.png`

## Notes on Reproducibility

- Run instructions are documented in `README.md` in this folder.
- Dependencies are declared in `pyproject.toml`.
