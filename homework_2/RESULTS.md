# Homework 2 Work Log

Use this file as a progress tracker while implementing the pipeline step by step.

## Task Checklist

- [x] Project scaffolding created (`pyproject.toml`, `README.md`, `REPORT.tex`, package structure).
- [x] Data loading helper implemented (dynamic ROI discovery via `Path.glob`).
- [x] Preprocessing helper implemented (grayscale + Gaussian denoise + optional unsharp-mask sharpening).
- [x] Scene **and** ROI images preprocessed in the pipeline (subtask 1 complete).
- [x] ORB detector/descriptor setup implemented (`nfeatures=3000`).
- [x] Keypoint + descriptor extraction implemented for scene and all ROIs (subtask 2 complete).
- [ ] Descriptor matching + Lowe ratio filtering implemented.
- [ ] ROI center estimation implemented.
- [ ] Final rectangle rendering implemented.
- [ ] Per-ROI debug visualization implemented.
- [ ] Final artifacts regenerated and reviewed.

## Planned Output Artifacts

- `output/final_localization.png`
- `output/matches/roi_*.png`

## Design Choices

- **ORB over SIFT**: binary descriptors + Hamming distance are faster; ROIs are same-scale axis-aligned crops so SIFT's scale-space advantage is unnecessary; ORB is the approach shown in course slides.
- **Gaussian blur (5×5)**: balances noise reduction with detail preservation for ORB keypoint detection.
- **nfeatures=3000**: provides dense scene coverage on a 2962×1935 image; ROI keypoint counts range from ~230 to ~800.

## Notes

- Keep implementation notes short and focused on what changed and why.
- Record parameter values you test (e.g., detector settings, ratio threshold, min matches).
