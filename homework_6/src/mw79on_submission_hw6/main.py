"""YOLO26 object-detection comparison pipeline (Ultralytics).

Compare five pretrained YOLO26 model sizes (``n``/``s``/``m``/``l``/``x``)
on the African Wildlife dataset by running inference and validation only
(no training). Both the per-image annotated outputs and the aggregate
mAP / runtime metrics are saved to ``output/``.

Pipeline:

1. Discover the dataset and the five pretrained weights under ``data/``.
2. Materialise a runtime copy of the dataset yaml with an absolute
   ``path:`` injected so ``model.val`` resolves the split directories on
   any workstation without touching Ultralytics' global ``DATASETS_DIR``.
3. Pick a fixed, seeded set of 16 test images so the visual montage is
   identical across models.
4. For each model size:

   - load the weights via :class:`ultralytics.YOLO`,
   - run ``predict`` on the 16 sampled images and save annotated frames
     plus a 4x4 montage,
   - run ``val`` on the official ``test`` split and record
     ``mAP@0.5``, ``mAP@0.75``, ``mAP@0.5:0.95`` and wall-clock runtime.

5. Persist a ``metrics.csv`` (and Markdown table) comparing the models.
"""

import csv
import logging
import os
import random
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import yaml
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_SIZES: tuple[str, ...] = ("n", "s", "m", "l", "x")
NUM_MONTAGE_IMAGES = 16
MONTAGE_GRID: tuple[int, int] = (4, 4)
MONTAGE_SEED = 0
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


def get_project_root() -> Path:
    """Return the ``homework_6`` project root directory."""
    return Path(__file__).resolve().parents[2]


# 1. Discovery
def resolve_weights(root: Path, size: str) -> Path:
    """Return the path to the ``yolo26<size>.pt`` weights file under *root*."""
    path = root / "data" / "weights" / f"yolo26{size}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing YOLO26 weights: {path}")
    return path


def resolve_data_yaml(root: Path) -> Path:
    """Return the path to the on-disk dataset ``african-wildlife.yaml``."""
    path = root / "data" / "african-wildlife" / "african-wildlife.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset yaml: {path}")
    return path


# 2. Runtime yaml materialisation
def make_runtime_data_yaml(root: Path) -> Path:
    """Write a tempfile copy of ``african-wildlife.yaml`` with ``path:`` injected.

    Ultralytics resolves the yaml's ``path:`` against its global
    ``DATASETS_DIR`` setting, not the yaml's directory. The dataset ships
    with ``path: african-wildlife`` (a relative stub), so calling
    ``model.val(data=...)`` directly on it would fail unless the global
    setting were pre-configured. This helper rewrites ``path:`` to the
    absolute extracted dataset root and returns the tempfile path; the
    caller is responsible for deleting it once done.
    """
    src = resolve_data_yaml(root)
    cfg = yaml.safe_load(src.read_text())
    cfg["path"] = str((root / "data" / "african-wildlife").resolve())

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="african-wildlife-runtime-",
        suffix=".yaml",
        delete=False,
    )
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()
    logger.info("Runtime dataset yaml -> %s (path=%s)", tmp.name, cfg["path"])
    return Path(tmp.name)


# 3. Montage sampling
def pick_montage_images(
    test_dir: Path,
    n: int = NUM_MONTAGE_IMAGES,
    seed: int = MONTAGE_SEED,
) -> list[Path]:
    """Return *n* test-image paths sampled with a fixed seed for reproducibility.

    The returned list is sorted alphabetically so the same image lands in
    the same grid cell across every model run.
    """
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing test split dir: {test_dir}")
    images = sorted(
        p for p in test_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) < n:
        raise ValueError(
            f"Need at least {n} test images, found {len(images)} in {test_dir}"
        )
    sample = random.Random(seed).sample(images, n)
    return sorted(sample)


# 4. Per-model inference + validation
def run_predict(
    model: YOLO,
    images: list[Path],
    out_dir: Path,
) -> list[Path]:
    """Run ``model.predict`` on *images* and save annotated frames as PNGs.

    Returns:
        Paths of the saved annotated images in the same order as *images*.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = model.predict(
        source=[str(p) for p in images],
        verbose=False,
        save=False,
    )
    saved: list[Path] = []
    for source_path, result in zip(images, results, strict=True):
        plotted = result.plot()  # BGR uint8 ndarray with bbox+label+conf
        out_path = out_dir / f"{source_path.stem}.png"
        if not cv2.imwrite(str(out_path), plotted):
            raise IOError(f"cv2.imwrite failed for {out_path}")
        saved.append(out_path)
    return saved


def build_montage(
    image_paths: list[Path],
    out_path: Path,
    grid: tuple[int, int] = MONTAGE_GRID,
) -> None:
    """Stitch annotated frames into a single ``rows x cols`` montage figure."""
    rows, cols = grid
    expected = rows * cols
    if len(image_paths) != expected:
        raise ValueError(
            f"Got {len(image_paths)} images but grid {grid} needs {expected}"
        )

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    for ax, img_path in zip(axes.flat, image_paths, strict=True):
        img = cv2.imread(str(img_path))
        if img is None:
            raise IOError(f"Could not read annotated frame: {img_path}")
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(img_path.stem, fontsize=8)
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_val(
    model: YOLO,
    data_yaml: Path,
    project_dir: Path,
    run_name: str,
) -> dict[str, float]:
    """Run ``model.val(split="test")`` and return mAP + wall-clock runtime.

    The Ultralytics per-image ``speed`` dict (preprocess/inference/
    postprocess, in ms) is preserved as a footnote-level metric.

    Returns:
        Dict with ``map50``, ``map75``, ``map50_95``, ``runtime_s``,
        ``speed_preprocess_ms``, ``speed_inference_ms``,
        ``speed_postprocess_ms``.
    """
    t0 = time.perf_counter()
    results = model.val(
        data=str(data_yaml),
        split="test",
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    runtime_s = time.perf_counter() - t0

    box = results.box
    speed = getattr(results, "speed", {}) or {}
    return {
        "map50": float(box.map50),
        "map75": float(box.map75),
        "map50_95": float(box.map),
        "runtime_s": runtime_s,
        "speed_preprocess_ms": float(speed.get("preprocess", 0.0)),
        "speed_inference_ms": float(speed.get("inference", 0.0)),
        "speed_postprocess_ms": float(speed.get("postprocess", 0.0)),
    }


# 5. Reporting
def write_comparison_table(
    rows: list[dict[str, float | str]],
    csv_path: Path,
    md_path: Path,
) -> None:
    """Persist the per-model comparison as both CSV and Markdown."""
    if not rows:
        raise ValueError("write_comparison_table requires at least one row")
    fieldnames = list(rows[0].keys())

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def fmt(value: float | str) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    md_lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        md_lines.append(
            "| " + " | ".join(fmt(row[k]) for k in fieldnames) + " |"
        )
    md_path.write_text("\n".join(md_lines) + "\n")


# Pipeline
def main() -> None:
    """Run the full homework 6 pipeline end-to-end."""
    root = get_project_root()
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    montages_dir = output_dir / "montages"
    predictions_dir = output_dir / "predictions"
    val_runs_dir = output_dir / "val_runs"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_count = os.cpu_count() or 1

    logger.info("Project root : %s", root)
    logger.info("Output dir   : %s", output_dir)
    logger.info("Device       : %s (cpu_count=%d)", device, cpu_count)
    logger.info("Model sizes  : %s", ", ".join(MODEL_SIZES))

    runtime_yaml = make_runtime_data_yaml(root)
    try:
        test_dir = root / "data" / "african-wildlife" / "images" / "test"
        montage_images = pick_montage_images(test_dir)
        logger.info(
            "Sampled %d test images (seed=%d) for the montage:",
            len(montage_images),
            MONTAGE_SEED,
        )
        for p in montage_images:
            logger.info("  %s", p.name)

        rows: list[dict[str, float | str]] = []
        for size in MODEL_SIZES:
            run_name = f"yolo26{size}"
            weights = resolve_weights(root, size)
            logger.info("=== %s ===", run_name)
            logger.info("Loading weights: %s", weights)
            model = YOLO(str(weights))
            model.to(device)

            per_image_dir = predictions_dir / run_name
            logger.info(
                "Predicting %d montage images -> %s",
                len(montage_images),
                per_image_dir,
            )
            saved = run_predict(model, montage_images, per_image_dir)

            montage_path = montages_dir / f"{run_name}.png"
            logger.info("Building %dx%d montage -> %s", *MONTAGE_GRID, montage_path)
            build_montage(saved, montage_path)

            logger.info("Running model.val(split='test') ...")
            metrics = run_val(model, runtime_yaml, val_runs_dir, run_name)
            rows.append({"model": run_name, **metrics})
            logger.info(
                "%s -> mAP@.5=%.4f mAP@.75=%.4f mAP@.5:.95=%.4f "
                "runtime=%.2fs  speed(ms/img): pre=%.2f inf=%.2f post=%.2f",
                run_name,
                metrics["map50"],
                metrics["map75"],
                metrics["map50_95"],
                metrics["runtime_s"],
                metrics["speed_preprocess_ms"],
                metrics["speed_inference_ms"],
                metrics["speed_postprocess_ms"],
            )

        csv_path = output_dir / "metrics.csv"
        md_path = output_dir / "metrics.md"
        write_comparison_table(rows, csv_path, md_path)
        logger.info("Wrote %s and %s", csv_path, md_path)
    finally:
        try:
            runtime_yaml.unlink()
            logger.info("Cleaned up runtime yaml: %s", runtime_yaml)
        except OSError as exc:
            logger.warning("Could not delete runtime yaml %s: %s", runtime_yaml, exc)

    logger.info("Homework 6 pipeline complete.")


if __name__ == "__main__":
    main()
