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
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import DTD
from torchvision.models import MaxVit_T_Weights, maxvit_t
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_project_root() -> Path:
    """Return the ``homework_5`` project root directory."""
    return Path(__file__).resolve().parents[2]


# 1. Data preparation & augmentation
def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    pin_memory: bool,
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
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    data_root.mkdir(parents=True, exist_ok=True)
    train_set = DTD(
        root=str(data_root), split="train", partition=1,
        download=True, transform=train_transform,
    )
    val_set = DTD(
        root=str(data_root), split="val", partition=1,
        download=True, transform=eval_transform,
    )
    test_set = DTD(
        root=str(data_root), split="test", partition=1,
        download=True, transform=eval_transform,
    )

    class_names = list(train_set.classes)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    logger.info(
        "DTD partition=1 — train: %d  val: %d  test: %d  classes: %d",
        len(train_set), len(val_set), len(test_set), len(class_names),
    )
    return train_loader, val_loader, test_loader, class_names


# 2. Transfer learning & model setup
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Build a MaxViT-T backbone with a fresh ``num_classes``-way classifier.

    Loads ``torchvision.models.maxvit_t`` with the ``IMAGENET1K_V1`` weights,
    replaces the final ``nn.Linear`` in the classifier head to output
    ``num_classes`` logits (47 for DTD), and moves the model to *device*.
    All parameters are left trainable so the entire network is fine-tuned.
    """
    weights = MaxVit_T_Weights.IMAGENET1K_V1
    model = maxvit_t(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = True

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "MaxViT-T loaded — total params: %.2fM, trainable: %.2fM (head -> %d classes)",
        total_params / 1e6, trainable_params / 1e6, num_classes,
    )
    return model


# 3. Training loop
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return ``(mean_loss, accuracy)``."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run evaluation in ``eval`` mode and return ``(mean_loss, accuracy)``."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


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
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
    }
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            best_marker = " (new best, saved)"
        else:
            epochs_without_improvement += 1
            best_marker = ""

        logger.info(
            "epoch %d/%d — train_loss=%.4f train_acc=%.4f  "
            "val_loss=%.4f val_acc=%.4f  best_val_loss=%.4f  patience=%d/%d%s",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
            best_val_loss, epochs_without_improvement, patience, best_marker,
        )

        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping at epoch %d (no val_loss improvement for %d epochs).",
                epoch, patience,
            )
            break

    # Reload best weights so the downstream test eval uses the best checkpoint.
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    logger.info("Reloaded best checkpoint from %s", checkpoint_path)
    return history


# 4. Evaluation & visualisation
@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Return ``(y_true, y_pred)`` for every sample in *loader*."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in tqdm(loader, desc="predict", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save a 1×2 figure: train/val loss and train/val accuracy vs. epoch."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved to %s", output_path)


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save a labelled 47×47 confusion matrix figure for the test predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        cmap="viridis",
        cbar=True,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix (test set, {len(class_names)} classes)")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)


# Pipeline
def main() -> None:
    """Run the full homework 5 pipeline end-to-end."""
    root = get_project_root()
    data_root = root / "data"
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    cpu_count = os.cpu_count() or 1
    num_workers = min(4, max(1, cpu_count // 2))
    batch_size = 32 if device.type == "cuda" else 8
    image_size = 224
    epochs = 30
    patience = 5
    learning_rate = 1e-4
    weight_decay = 1e-4

    logger.info("Project root : %s", root)
    logger.info("Data root    : %s", data_root)
    logger.info("Output dir   : %s", output_dir)
    logger.info(
        "Runtime config — device=%s batch_size=%d num_workers=%d "
        "pin_memory=%s image_size=%d epochs=%d patience=%d lr=%.1e wd=%.1e",
        device,
        batch_size,
        num_workers,
        pin_memory,
        image_size,
        epochs,
        patience,
        learning_rate,
        weight_decay,
    )

    # 1. Data preparation & augmentation
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        pin_memory=pin_memory,
    )
    logger.info("Subtask 1 complete.")

    # 2. Transfer learning & model setup
    model = build_model(num_classes=len(class_names), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    logger.info("Subtask 2 complete.")

    # 3. Training loop
    checkpoint_path = output_dir / "best_model.pt"
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
    )
    logger.info("Subtask 3 complete — best checkpoint saved to %s.", checkpoint_path)

    # 4. Evaluation & visualisation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info("Test metrics — loss=%.4f accuracy=%.4f", test_loss, test_acc)

    y_true, y_pred = collect_predictions(model, test_loader, device)
    plot_training_curves(history, output_dir / "training_curves.png")
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=output_dir / "confusion_matrix.png",
    )
    logger.info("Subtask 4 complete.")

    logger.info(
        "Homework 5 pipeline complete — test_loss=%.4f test_accuracy=%.4f",
        test_loss,
        test_acc,
    )


if __name__ == "__main__":
    main()
