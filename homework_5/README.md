## Homework 5 — Transfer-Learning Image Classification (PyTorch)

This folder contains the solution for Homework 5: building a deep-learning
image classification pipeline in PyTorch by fine-tuning a pretrained
backbone on a `torchvision` dataset. The chosen combination is
**MaxViT-T** (Google, ECCV 2022) fine-tuned on the **Describable Textures
Dataset** (DTD, Oxford VGG, 47 classes).

## Contents

- `data/description.pdf`: official assignment description.
- `src/mw79on_submission_hw5/main.py`: complete pipeline implementation.
- `output/`: generated figures (training curves, confusion matrix) and the
  best model checkpoint.

## Model & Dataset Choice

Out of the available pretrained backbones in `torchvision.models`, **MaxViT-T**
([Tu et al., ECCV 2022](https://arxiv.org/abs/2204.01697); Google Research +
UT Austin) seemed like the most interesting pick:

- **Genuinely novel attention pattern.** Each block stacks `MBConv` →
  block (local windowed) attention → grid (dilated global) attention. The
  grid-attention component gives the network a global receptive field at
  every stage with **linear complexity** in token count, which standard
  ViTs and Swin variants don't offer.
- **Hybrid CNN + ViT design.** The MBConv stem and per-block convolutions
  restore the local inductive bias that pure ViTs lack, which usually
  helps fine-tuning on smaller datasets.
- **Underused in coursework.** ConvNeXt and ResNet are the typical picks;
  MaxViT-T is a less-trodden but still standard-recipe-friendly choice.
- **Reasonable footprint.** 30.9M params, 5.56 GFLOPs, 224×224 input —
  fine-tunes on a single GPU in standard time budgets.

For the dataset, **DTD** complements the model well:

- **Texture classification, not object recognition.** This forces the
  backbone to lean on its mid-layer texture features rather than its
  top-level ImageNet semantics, which makes the "all layers trainable /
  fine-tune the whole network" requirement actually pay off.
- **47 classes** (`banded`, `bubbly`, `chequered`, `crosshatched`, `dotted`,
  `fibrous`, `swirly`, `zigzagged`, ...) give a meaningfully shaped
  confusion matrix, with semantically related textures expected to cluster.
- **Small footprint** (5 640 images, 120 per class) enables fast training
  iterations and multiple hyperparameter passes.
- **Official 10-fold splits** ship with the dataset; this pipeline uses
  `partition=1` for reproducibility.

A small caveat for the report: torchvision's `maxvit_t` weights were
trained with a known
[`BatchNorm2d` momentum bug](https://github.com/pytorch/vision/commits/main/torchvision/models)
(0.99 instead of 0.01), which is why torchvision reports 83.7% top-1 on
ImageNet versus the paper's 86.5%. The model still fine-tunes well; the
caveat is worth mentioning for transparency.

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

1. **Data Preparation & Augmentation**: downloads/loads `torchvision.datasets.DTD`
   (`partition=1`) into train/val/test splits. The training transform stacks
   `RandomHorizontalFlip`, `RandomRotation` and `ColorJitter` on top of resize
   to 224×224 and ImageNet normalisation
   (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Validation and
   test transforms apply only resize + normalisation.
2. **Transfer Learning & Model Setup**: loads `torchvision.models.maxvit_t`
   with `IMAGENET1K_V1` weights, replaces the final linear layer in the
   classifier head with a 47-way output, and leaves all parameters
   trainable for full fine-tuning.
3. **Training Loop**: runs the training/validation loops with
   `CrossEntropyLoss` and Adam (or SGD), evaluates each epoch under
   `model.eval()` + `torch.no_grad()`, and applies early stopping that keeps
   only the best checkpoint.
4. **Evaluation & Visualisation**: scores the held-out test set, plots the
   training/validation loss and accuracy curves, and saves a 47×47
   confusion matrix figure of the test predictions.

Build the PDF report (from `homework_5/`):

```bash
pdflatex -interaction=nonstopmode -halt-on-error REPORT.tex
```
