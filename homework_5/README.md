## Homework 5 — Transfer-Learning Image Classification (PyTorch)

This folder contains the solution for Homework 5: building a deep-learning
image classification pipeline in PyTorch by fine-tuning a pretrained
backbone on a `torchvision` dataset.

## Contents

- `data/description.pdf`: official assignment description.
- `src/mw79on_submission_hw5/main.py`: complete pipeline implementation.
- `output/`: generated figures, training curves, and confusion matrix.

## Requirements

- Python `>=3.10`
- Packages:
  - `torch`
  - `torchvision`
  - `matplotlib`
  - `seaborn`
  - `tqdm`
  - `scikit-learn`
  - `numpy`

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
pip install -e homework_5
```

## Run

From `homework_5/`, run:

```bash
python src/mw79on_submission_hw5/main.py
```

The script executes the full pipeline:

1. **Data Preparation & Augmentation**: loads a `torchvision` dataset, splits
   it into train/val/test, applies at least three online augmentations on the
   training set (e.g. `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`),
   resizes inputs to the backbone's expected resolution (e.g. 224×224), and
   applies the standard ImageNet normalisation. Validation and test loaders
   use only resize + normalisation.
2. **Transfer Learning & Model Setup**: loads a pretrained backbone from
   `torchvision.models` (e.g. ResNet-18 or EfficientNet-B0), replaces the
   classifier head to match the target class count, and enables gradients on
   every layer so the entire network is fine-tuned.
3. **Training Loop**: runs the training/validation loops with
   `CrossEntropyLoss` and Adam (or SGD), evaluates each epoch under
   `model.eval()` + `torch.no_grad()`, and applies early stopping that keeps
   only the best checkpoint.
4. **Evaluation & Visualisation**: scores the held-out test set, plots the
   training/validation loss and accuracy curves, and saves a confusion matrix
   figure of the test predictions.

Build the PDF report (from `homework_5/`):

```bash
pdflatex -interaction=nonstopmode -halt-on-error REPORT.tex
```
