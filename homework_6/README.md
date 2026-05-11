## Homework 6 — YOLO26 Object Detection Comparison

This folder contains the solution for Homework 6: comparing five
pretrained **YOLO26** sizes (`n`/`s`/`m`/`l`/`x`) on the **African
Wildlife** dataset using inference + validation only — no training.
Each size is run through `predict` (qualitative output) and `val`
(mAP@0.5, mAP@0.75, mAP@0.5:0.95, runtime), and the trade-offs are
discussed in `REPORT.pdf`.

## Contents

- `data/description.pdf`: official assignment description.
- `data/african-wildlife.yaml`: dataset config (kept with the original
  relative `path:` stub; see "Dataset & weights" below).
- `data/african-wildlife/`: extracted dataset (gitignored).
- `data/weights/`: extracted YOLO26 weights `yolo26{n,s,m,l,x}.pt`
  (gitignored).
- `src/mw79on_submission_hw6/main.py`: complete pipeline implementation.
- `output/montages/yolo26{n,s,m,l,x}.png`: 4×4 prediction montages, one
  per model size, on the same seeded set of 16 test images.
- `output/predictions/yolo26{n,s,m,l,x}/`: per-image annotated PNGs
  (16 frames per model) backing the montages.
- `output/val_runs/yolo26{n,s,m,l,x}/`: Ultralytics' per-model
  validation artefacts — F1/P/R/PR curves, confusion matrices, and
  `val_batchN_{labels,pred}.jpg` previews.
- `output/metrics.csv` / `output/metrics.md`: comparison table
  (mAP@0.5, mAP@0.75, mAP@0.5:0.95, total runtime, per-image speed).
- `output/run_log.txt`: captured stdout from the full pipeline run.
- `output/report_figures/yolo26x_montage.jpg`: downscaled JPEG of the
  YOLO26-x montage embedded in the PDF report.
- `REPORT.tex` / `REPORT.pdf`: 2-page submission report.

## Dataset & weights

Both the dataset zip and the weights zip ship in this folder
(`african-wildlife.zip`, `wildlife_yolo26_weights.zip`); both files are
gitignored. Extract before running:

```bash
unzip african-wildlife.zip       -d data/
unzip wildlife_yolo26_weights.zip -d data/weights/
```

This produces `data/african-wildlife/{images,labels}/{train,val,test}/`
plus the `african-wildlife.yaml` config, and `data/weights/yolo26{n,s,m,l,x}.pt`.

The shipped `african-wildlife.yaml` keeps the original Ultralytics
`path: african-wildlife` (a relative stub). Ultralytics resolves that
against its global `DATASETS_DIR` setting rather than the yaml's
location, so calling `model.val(data=...)` directly on it would fail.
`main.py` works around this by writing a temporary copy of the yaml at
runtime with `path:` rewritten to the absolute extracted dataset root,
and deletes the temp file in a `finally` block. This keeps the on-disk
yaml portable across machines.

## Requirements

- Python `>= 3.10` (verified on **3.14.4**).
- Packages (installed via the package below):
  - `ultralytics` (verified with `8.4.48`)
  - `torch`, `torchvision` (CPU build verified with `2.11.0+cpu` /
    `0.26.0+cpu`)
  - `matplotlib`, `numpy`, `opencv-python`, `pyyaml`, `tqdm`

## Environment Setup

Use the shared repo virtualenv from the repository root (same one used
by the earlier homeworks):

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

CPU-only torch install (recommended on this machine — no local GPU):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Two notes worth flagging:

1. The torch CPU wheel is ~170 MB. On the dev machine used here the
   PyPI mirror was throttled to ~240 KB/s, so the install took ~15 min.
   Plan accordingly if you rebuild the venv from scratch.
2. `ultralytics 8.4.48` does not list Python 3.14 in its PyPI
   classifiers (only 3.8–3.12), but it is pure Python and installs and
   runs cleanly on 3.14.4.

## Run

From `homework_6/` (after extracting the data and weights):

```bash
python src/mw79on_submission_hw6/main.py
```

The script executes the full pipeline:

1. **Dataset yaml**: writes a temporary copy with an absolute `path:` so
   `model.val` resolves the splits portably.
2. **Montage sample**: picks 16 test images using a fixed seed
   (`MONTAGE_SEED = 0`) so the same scenes are shown across all model
   sizes, making the visual comparison direct.
3. **Per model size** (`n`/`s`/`m`/`l`/`x`):
   - load the weights via `ultralytics.YOLO`,
   - run `model.predict` on the 16 sampled images, render bounding
     boxes + class labels + confidence with `Results.plot()`, save each
     annotated PNG, and stitch them into a 4×4 montage,
   - run `model.val(split="test")` on the full 227-image test split,
     timed with `time.perf_counter()`, recording mAP@0.5, mAP@0.75,
     mAP@0.5:0.95 plus Ultralytics' per-image preprocess/inference/
     postprocess speeds.
4. **Comparison table**: writes `output/metrics.csv` and
   `output/metrics.md`.

End-to-end runtime on the 8-core CPU used here is ~6 minutes
(`n` ≈ 14 s, `s` ≈ 22 s, `m` ≈ 57 s, `l` ≈ 73 s, `x` ≈ 154 s for `val`,
plus a few seconds of `predict` per model).

## Build the report

From `homework_6/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error REPORT.tex
pdflatex -interaction=nonstopmode -halt-on-error REPORT.tex
```

Two passes are needed for the cross-references to `Table 1` and
`Figure 1` to resolve. The compiled `REPORT.pdf` is 2 pages.
