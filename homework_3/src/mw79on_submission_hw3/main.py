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

KERNEL_SIZES: list[int] = [3, 5, 7]
"""Gaussian kernel sizes to evaluate."""

QUALITY_LEVELS: list[int] = list(range(10, 101, 10))
"""WebP quality parameters to evaluate (10, 20, …, 100)."""

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
def apply_gaussian_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Reduce Gaussian noise using a Gaussian blur filter.

    The noisy images in this dataset are corrupted with normally-distributed
    (Gaussian) noise, so use a Gaussian low-pass filter.

    Sigma is set to 0 so that OpenCV derives it automatically from the kernel
    size.

    Args:
        image: Noisy input image (BGR, uint8).
        kernel_size: Side length of the square Gaussian kernel.

    Returns:
        Denoised image (BGR, uint8).
    """
    assert kernel_size % 2 == 1, f"Kernel size must be odd, got {kernel_size}"
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)


# Image Compression
def compress_image(
    image: np.ndarray,
    quality: int,
) -> tuple[np.ndarray, int]:
    """Compress an image using WebP encoding at the given quality level.

    The image is encoded into an in-memory WebP buffer and immediately decoded
    back, simulating a lossy compression round-trip.

    Args:
        image: Input image (BGR, uint8).
        quality: WebP quality parameter (1–100).

    Returns:
        Tuple of (reconstructed image in BGR uint8, compressed size in bytes).
    """
    ok, buf = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, quality])
    assert ok, f"WebP encoding failed at quality={quality}"
    compressed_size = len(buf)
    reconstructed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert reconstructed is not None, "WebP decoding failed"
    return reconstructed, compressed_size


# Quality Measurement
def compute_mse(reference: np.ndarray, distorted: np.ndarray) -> float:
    """Compute Mean Squared Error between two images.

    Args:
        reference: Clean reference image (uint8).
        distorted: Distorted image (uint8).

    Returns:
        MSE value.
    """
    return float(np.mean((reference.astype(np.float64) - distorted.astype(np.float64)) ** 2))


def compute_psnr(reference: np.ndarray, distorted: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        reference: Clean reference image (uint8).
        distorted: Distorted image (uint8).

    Returns:
        PSNR value in dB.
    """
    return float(cv2.PSNR(reference, distorted))


def compute_ssim(reference: np.ndarray, distorted: np.ndarray) -> float:
    """Compute the mean Structural Similarity Index between two images.

    The SSIM is computed per-channel on a per-pixel basis using the formulation
    from Wang et al. (2004) with an 11×11 Gaussian window (sigma=1.5) and the
    default stabilisation constants (C1 and C2 derived from the dynamic range).
    The per-pixel maps are averaged across all channels and all pixels to yield
    a single scalar.

    Args:
        reference: Clean reference image (uint8, BGR).
        distorted: Distorted image (uint8, BGR).

    Returns:
        Mean SSIM in [0, 1].
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ksize = (11, 11)
    sigma = 1.5

    ref = reference.astype(np.float64)
    dist = distorted.astype(np.float64)

    mu_ref = cv2.GaussianBlur(ref, ksize, sigma)
    mu_dist = cv2.GaussianBlur(dist, ksize, sigma)

    mu_ref_sq = mu_ref ** 2
    mu_dist_sq = mu_dist ** 2
    mu_ref_dist = mu_ref * mu_dist

    sigma_ref_sq = cv2.GaussianBlur(ref ** 2, ksize, sigma) - mu_ref_sq
    sigma_dist_sq = cv2.GaussianBlur(dist ** 2, ksize, sigma) - mu_dist_sq
    sigma_ref_dist = cv2.GaussianBlur(ref * dist, ksize, sigma) - mu_ref_dist

    numerator = (2 * mu_ref_dist + C1) * (2 * sigma_ref_dist + C2)
    denominator = (mu_ref_sq + mu_dist_sq + C1) * (sigma_ref_sq + sigma_dist_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def compute_compression_ratio(raw_size: int, compressed_size: int) -> float:
    """Compute the compression ratio.

    Args:
        raw_size: Uncompressed image size in bytes (H × W × C).
        compressed_size: Compressed buffer size in bytes.

    Returns:
        Compression ratio (raw / compressed); higher means more compression.
    """
    return raw_size / compressed_size


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

    # 2. Noise reduction: apply Gaussian filter at each kernel size
    # denoised[kernel_size] = [(name, clean, denoised), ...]
    denoised: dict[int, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    for ksize in KERNEL_SIZES:
        denoised[ksize] = []
        for name, clean_img, noisy_img in pairs:
            filtered = apply_gaussian_filter(noisy_img, ksize)
            denoised[ksize].append((name, clean_img, filtered))
        logger.info(
            "Denoised %d images with %d×%d Gaussian kernel.",
            len(pairs),
            ksize,
            ksize,
        )

    # 3. Compression: encode each denoised image at every quality level
    # compressed[kernel_size][quality] = [(name, clean, reconstructed, size_bytes), ...]
    compressed: dict[int, dict[int, list[tuple[str, np.ndarray, np.ndarray, int]]]] = {}
    for ksize in KERNEL_SIZES:
        compressed[ksize] = {}
        for q in QUALITY_LEVELS:
            compressed[ksize][q] = []
            for name, clean_img, denoised_img in denoised[ksize]:
                reconstructed, size_bytes = compress_image(denoised_img, q)
                compressed[ksize][q].append((name, clean_img, reconstructed, size_bytes))
            logger.info(
                "Compressed %d images: kernel %d×%d, WebP quality %d.",
                len(pairs),
                ksize,
                ksize,
                q,
            )

    # 4. Quality metrics: compare reconstructed images against clean originals
    # metrics[kernel_size][quality] = {"mse": [...], "psnr": [...], "ssim": [...], "cr": [...]}
    metrics: dict[int, dict[int, dict[str, list[float]]]] = {}
    for ksize in KERNEL_SIZES:
        metrics[ksize] = {}
        for q in QUALITY_LEVELS:
            mse_vals, psnr_vals, ssim_vals, cr_vals = [], [], [], []
            for name, clean_img, recon_img, size_bytes in compressed[ksize][q]:
                raw_size = clean_img.nbytes
                mse_vals.append(compute_mse(clean_img, recon_img))
                psnr_vals.append(compute_psnr(clean_img, recon_img))
                ssim_vals.append(compute_ssim(clean_img, recon_img))
                cr_vals.append(compute_compression_ratio(raw_size, size_bytes))
            metrics[ksize][q] = {
                "mse": mse_vals,
                "psnr": psnr_vals,
                "ssim": ssim_vals,
                "cr": cr_vals,
            }
            logger.info(
                "Metrics computed: kernel %d×%d, quality %d  →  "
                "MSE=%.1f  PSNR=%.2f dB  SSIM=%.4f  CR=%.1f:1",
                ksize,
                ksize,
                q,
                np.mean(mse_vals),
                np.mean(psnr_vals),
                np.mean(ssim_vals),
                np.mean(cr_vals),
            )

    # TODO: Implement and wire remaining pipeline stages
    #   5. Generate comparison plots

    logger.info("Remaining pipeline stages not yet implemented.")


if __name__ == "__main__":
    main()
