"""YOLO26 object-detection comparison pipeline (Ultralytics).

Compare five pretrained YOLO26 model sizes (``n``/``s``/``m``/``l``/``x``)
on the African Wildlife dataset by running inference and validation only
(no training). Both the per-image annotated outputs and the aggregate
mAP / runtime metrics are saved to ``output/``.

Pipeline:

1. Discover the dataset and the five pretrained weights under ``data/``.
2. Pick a fixed, seeded set of 16 test images so the visual montage is
   identical across models.
3. For each model size:
   - load the weights via :class:`ultralytics.YOLO`,
   - run ``predict`` on the 16 sampled images and save annotated frames
     plus a 4x4 montage,
   - run ``val`` on the official ``test`` split and record
     ``mAP@0.5``, ``mAP@0.75``, ``mAP@0.5:0.95`` and wall-clock runtime.
4. Persist a ``metrics.csv`` (and Markdown table) comparing the models.
"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_SIZES: tuple[str, ...] = ("n", "s", "m", "l", "x")
NUM_MONTAGE_IMAGES = 16
MONTAGE_GRID: tuple[int, int] = (4, 4)
MONTAGE_SEED = 0


def get_project_root() -> Path:
    """Return the ``homework_6`` project root directory."""
    return Path(__file__).resolve().parents[2]


# 1. Discovery
def resolve_weights(root: Path, size: str) -> Path:
    """Return the path to the ``yolo26<size>.pt`` weights file under *root*."""
    raise NotImplementedError


def resolve_data_yaml(root: Path) -> Path:
    """Return the path to the dataset ``african-wildlife.yaml`` under *root*."""
    raise NotImplementedError


# 2. Montage sampling
def pick_montage_images(
    test_dir: Path,
    n: int = NUM_MONTAGE_IMAGES,
    seed: int = MONTAGE_SEED,
) -> list[Path]:
    """Return *n* test-image paths sampled with a fixed seed for reproducibility."""
    raise NotImplementedError


# 3. Per-model inference + validation
def run_predict(
    model,
    images: list[Path],
    out_dir: Path,
) -> list[Path]:
    """Run ``model.predict`` and save annotated frames (bbox + label + conf).

    Returns:
        Paths of the saved annotated images in the same order as *images*.
    """
    raise NotImplementedError


def build_montage(
    image_paths: list[Path],
    out_path: Path,
    grid: tuple[int, int] = MONTAGE_GRID,
) -> None:
    """Stitch annotated frames into a single ``grid[0]xgrid[1]`` montage."""
    raise NotImplementedError


def run_val(model, data_yaml: Path) -> dict[str, float]:
    """Run ``model.val(split="test")`` and return mAP + wall-clock runtime.

    Returns:
        ``{"map50": ..., "map75": ..., "map50_95": ..., "runtime_s": ...}``.
    """
    raise NotImplementedError


# 4. Reporting
def write_comparison_table(
    rows: list[dict[str, float | str]],
    csv_path: Path,
    md_path: Path,
) -> None:
    """Persist the per-model comparison as both CSV and Markdown."""
    raise NotImplementedError


# Pipeline
def main() -> None:
    """Run the full homework 6 pipeline end-to-end."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
