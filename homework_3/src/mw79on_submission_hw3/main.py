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

    Args:
        clean_dir: Path to directory containing clean reference images.
        noisy_dir: Path to directory containing noisy images.

    Returns:
        List of (name, clean_image, noisy_image) tuples in BGR format.
    """
    raise NotImplementedError


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

    # TODO: Implement and wire pipeline stages together
    #   1. Load image pairs
    #   2. Apply noise reduction
    #   3. Apply compression at varying quality levels
    #   4. Compute quality metrics (PSNR, SSIM)
    #   5. Generate comparison plots

    logger.info("Not yet implemented.")


if __name__ == "__main__":
    main()
