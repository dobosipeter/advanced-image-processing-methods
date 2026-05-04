"""Transfer-learning image classification on DTD with MaxViT-T (PyTorch).

Backbone: ``torchvision.models.maxvit_t`` initialised with the
``IMAGENET1K_V1`` weights (Google Research, ECCV 2022 — multi-axis attention).
Dataset: ``torchvision.datasets.DTD`` (Describable Textures Dataset, 47
texture categories, 5 640 images, official ``partition=1`` split).

Pipeline:

1. Data preparation & augmentation: download/load the DTD train/val/test
   partition, apply at least three online augmentations to the training
   split, resize all splits to 224×224 and ImageNet-normalise, and wrap
   them in ``DataLoader``s.
2. Transfer learning & model setup: load pretrained MaxViT-T, replace its
   classifier head to output 47 logits, and enable gradients on every
   layer for full fine-tuning.
3. Training loop: pick a loss and optimiser, run ``forward``/``backward``/
   ``step`` per batch, evaluate on the validation split each epoch with
   ``model.eval()`` and ``torch.no_grad()``, and apply early stopping that
   keeps only the best checkpoint.
4. Evaluation & visualisation: compute test-set accuracy, plot the
   training/validation loss and accuracy curves, and save a confusion
   matrix figure of the test predictions.
"""

import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Return the ``homework_5`` project root directory."""
    return Path(__file__).resolve().parents[2]


# 1. Data preparation & augmentation
def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/val/test ``DataLoader``s for DTD.

    DTD ships with 10 official train/val/test partitions; this pipeline uses
    ``partition=1``. The training transform stacks ``RandomHorizontalFlip``,
    ``RandomRotation`` and ``ColorJitter`` on top of the standard resize +
    ImageNet normalisation; validation and test transforms perform only
    resize + normalisation (no augmentation).

    Returns:
        ``(train_loader, val_loader, test_loader, class_names)`` where
        ``class_names`` is a 47-element list aligned with DTD's class indices.
    """
    raise NotImplementedError


# 2. Transfer learning & model setup
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Build a MaxViT-T backbone with a fresh ``num_classes``-way classifier.

    Loads ``torchvision.models.maxvit_t`` with the ``IMAGENET1K_V1`` weights,
    replaces the final ``nn.Linear`` in the classifier head to output
    ``num_classes`` logits (47 for DTD), and moves the model to *device*.
    All parameters are left trainable so the entire network is fine-tuned.
    """
    raise NotImplementedError


# 3. Training loop
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return ``(mean_loss, accuracy)``."""
    raise NotImplementedError


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run evaluation in ``eval`` mode and return ``(mean_loss, accuracy)``."""
    raise NotImplementedError


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    checkpoint_path: Path,
) -> dict[str, list[float]]:
    """Train with early stopping; persist the best checkpoint to *checkpoint_path*.

    Returns:
        Per-epoch history dict with keys
        ``"train_loss"``, ``"train_acc"``, ``"val_loss"``, ``"val_acc"``.
    """
    raise NotImplementedError


# 4. Evaluation & visualisation
@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Return ``(y_true, y_pred)`` for every sample in *loader*."""
    raise NotImplementedError


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save a 1×2 figure: train/val loss and train/val accuracy vs. epoch."""
    raise NotImplementedError


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save a labelled 47×47 confusion matrix figure for the test predictions."""
    raise NotImplementedError


# Pipeline
def main() -> None:
    """Run the full homework 5 pipeline end-to-end."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
