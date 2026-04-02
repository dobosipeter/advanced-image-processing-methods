import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Return the ``homework_3`` project root directory.

    Returns:
        Path to the ``homework_3`` folder, resolved from the location of this file.
    """
    return Path(__file__).resolve().parents[2]


def load_image_pairs(
    clean_dir: Path,
    noisy_dir: Path,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load matching pairs of clean and noisy images.

    Images are discovered dynamically via ``Path.glob``.  The clean set is
    indexed from 1 (``kodim01.png`` … ``kodim10.png``) while the noisy set is
    indexed from 0 (``kodim00_noisy.png`` … ``kodim09_noisy.png``), so pairing
    is done by sorted order rather than by filename index.

    Args:
        clean_dir: Path to directory containing clean reference images.
        noisy_dir: Path to directory containing noisy images.

    Returns:
        Sorted list of ``(name, clean_image, noisy_image)`` tuples in BGR
        format, where *name* is the clean filename stem (e.g. ``"kodim01"``).
    """
    clean_paths = sorted(clean_dir.glob("*.png"))
    noisy_paths = sorted(noisy_dir.glob("*.png"))

    assert len(clean_paths) > 0, f"No clean images found in {clean_dir}"
    assert len(noisy_paths) > 0, f"No noisy images found in {noisy_dir}"
    assert len(clean_paths) == len(noisy_paths), (
        f"Image count mismatch: {len(clean_paths)} clean vs {len(noisy_paths)} noisy"
    )

    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for clean_path, noisy_path in zip(clean_paths, noisy_paths):
        clean_img = cv2.imread(str(clean_path))
        noisy_img = cv2.imread(str(noisy_path))

        assert clean_img is not None, f"Failed to load clean image: {clean_path}"
        assert noisy_img is not None, f"Failed to load noisy image: {noisy_path}"
        assert clean_img.shape == noisy_img.shape, (
            f"Shape mismatch for {clean_path.name} / {noisy_path.name}: "
            f"{clean_img.shape} vs {noisy_img.shape}"
        )

        name = clean_path.stem
        pairs.append((name, clean_img, noisy_img))
        logger.info(
            "Loaded pair: %s ↔ %s  (%s)",
            clean_path.name,
            noisy_path.name,
            "×".join(str(d) for d in clean_img.shape),
        )

    logger.info("Loaded %d image pairs.", len(pairs))
    return pairs


# Noise Reduction

def apply_gaussian_filter(image: np.ndarray) -> np.ndarray:
    """Reduce noise using a Gaussian blur filter.

    Args:
        image: Noisy input image (BGR).

    Returns:
        Denoised image (BGR).
    """
    raise NotImplementedError


# Image Compression

def compress_image(image: np.ndarray, quality: int) -> np.ndarray:
    """Compress an image using JPEG encoding at the given quality level.

    Args:
        image: Input image (BGR).
        quality: JPEG quality parameter (0–100).

    Returns:
        Reconstructed image after compression (BGR).
    """
    raise NotImplementedError


# Quality Measurement

def compute_psnr(reference: np.ndarray, distorted: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        reference: Clean reference image.
        distorted: Distorted (noisy/compressed/denoised) image.

    Returns:
        PSNR value in dB.
    """
    raise NotImplementedError


def compute_ssim(reference: np.ndarray, distorted: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images.

    Args:
        reference: Clean reference image.
        distorted: Distorted (noisy/compressed/denoised) image.

    Returns:
        SSIM value in [0, 1].
    """
    raise NotImplementedError


# Visualization

def plot_results() -> None:
    """Generate comparison figures and save to the output directory."""
    raise NotImplementedError


# Pipeline

def main() -> None:
    """Run the full homework 3 pipeline end-to-end."""
    root = get_project_root()
    data_dir = root / "data"
    clean_dir = data_dir / "clean"
    noisy_dir = data_dir / "dobosi_peter_laszlo_norm_dist"
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    assert clean_dir.is_dir(), f"Clean image directory not found: {clean_dir}"
    assert noisy_dir.is_dir(), f"Noisy image directory not found: {noisy_dir}"

    logger.info("Project root: %s", root)
    logger.info("Clean images: %s", clean_dir)
    logger.info("Noisy images: %s", noisy_dir)
    logger.info("Output dir:   %s", output_dir)

    # 1. Load image pairs
    pairs = load_image_pairs(clean_dir, noisy_dir)

    # TODO: Implement and wire remaining pipeline stages
    #   2. Apply noise reduction (per kernel size)
    #   3. Apply compression at varying quality levels
    #   4. Compute quality metrics (MSE, PSNR, SSIM)
    #   5. Generate comparison plots

    logger.info("Remaining pipeline stages not yet implemented.")


if __name__ == "__main__":
    main()
