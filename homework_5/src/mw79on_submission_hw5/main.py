"""Transfer-learning image classification pipeline (PyTorch).

Pipeline:

1. Data preparation & augmentation: load a torchvision dataset, build
   train/val/test splits, apply at least three online augmentations to the
   training split, resize and ImageNet-normalise all splits, and wrap them
   in DataLoaders.
2. Transfer learning & model setup: load a pretrained backbone from
   ``torchvision.models``, replace its classifier head to match the target
   number of classes, and enable gradients on every layer for fine-tuning.
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
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    image_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/val/test ``DataLoader``s and return the class label list.

    Augmentations are applied to the training split only; the validation
    and test splits use resize + ImageNet normalisation only.

    Returns:
        ``(train_loader, val_loader, test_loader, class_names)``.
    """
    raise NotImplementedError


# 2. Transfer learning & model setup
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Load a pretrained backbone, swap its classifier, and move to *device*.

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
    """Save a labelled confusion matrix figure for the test predictions."""
    raise NotImplementedError


# Pipeline
def main() -> None:
    """Run the full homework 5 pipeline end-to-end."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
