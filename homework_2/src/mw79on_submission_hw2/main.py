import logging
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Return the ``homework_2`` project root directory.

    Returns:
        Path to the ``homework_2`` folder, resolved from the location of this file.
    """
    return Path(__file__).resolve().parents[2]


def load_inputs(
    main_image_path: Path,
    roi_dir: Path,
) -> tuple[MatLike, list[tuple[str, MatLike]]]:
    """Load the full scene image and all ROI images from *roi_dir*.

    Args:
        main_image_path: Path to the full noisy motherboard image.
        roi_dir: Directory containing ROI ``.jpg`` files.

    Returns:
        A tuple of ``(full_image, rois)`` where *full_image* is the BGR scene
        image and *rois* is a list of ``(stem_name, bgr_image)`` pairs.

    Raises:
        AssertionError: If the scene image cannot be read, no ``.jpg`` files
            exist in *roi_dir*, or none of the found files are readable.
    """
    logger.info("Loading scene image from %s", main_image_path)
    full_image = cv2.imread(str(main_image_path), cv2.IMREAD_COLOR)
    assert full_image is not None, f"Could not load full image: {main_image_path}"
    logger.info(
        "Scene image loaded — dimensions: %dx%d, channels: %d",
        full_image.shape[1],
        full_image.shape[0],
        full_image.shape[2],
    )

    roi_files = sorted(roi_dir.glob("*.jpg"))
    assert roi_files, f"No ROI images found in: {roi_dir}"
    logger.info("Found %d ROI file(s) in %s", len(roi_files), roi_dir)

    rois: list[tuple[str, MatLike]] = []
    for roi_file in roi_files:
        roi_image = cv2.imread(str(roi_file), cv2.IMREAD_COLOR)
        if roi_image is None:
            logger.warning("Skipping unreadable ROI image: %s", roi_file)
            continue
        logger.info(
            "  ROI '%s' loaded — dimensions: %dx%d",
            roi_file.stem,
            roi_image.shape[1],
            roi_image.shape[0],
        )
        rois.append((roi_file.stem, roi_image))

    assert rois, f"No readable ROI images found in: {roi_dir}"
    return full_image, rois


def preprocess_image_for_features(
    image_bgr: MatLike,
    blur_kernel_size: tuple[int, int] = (5, 5),
    sharpen: bool = False,
    sharpen_strength: float = 0.3,
) -> MatLike:
    """Convert a BGR image to grayscale, denoise it, and optionally sharpen.

    The function applies Gaussian blur for denoising.  When *sharpen* is
    ``True`` it additionally performs soft sharpening via an unsharp-mask
    convolution: ``sharpened = (1 + strength) * gray - strength * blurred``,
    clipped to ``[0, 255]``.

    Args:
        image_bgr: Input image in BGR colour order.
        blur_kernel_size: Kernel size ``(w, h)`` for Gaussian blur.
        sharpen: Whether to apply soft sharpening after denoising.
        sharpen_strength: Blending weight for the sharpening pass (only used
            when *sharpen* is ``True``).

    Returns:
        Preprocessed single-channel ``uint8`` image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, blur_kernel_size, sigmaX=0)
    h, w = denoised.shape[:2]
    logger.info(
        "Preprocessed image — grayscale %dx%d, blur kernel %s",
        w,
        h,
        blur_kernel_size,
    )

    if sharpen:
        gray_f = np.asarray(gray, dtype=np.float64)
        blur_f = np.asarray(denoised, dtype=np.float64)
        blended = (1.0 + sharpen_strength) * gray_f - sharpen_strength * blur_f
        denoised = np.clip(blended, 0, 255).astype(np.uint8)
        logger.info("Applied soft sharpening (strength=%.2f)", sharpen_strength)

    return denoised


def build_orb_detector(n_features: int = 3000) -> cv2.ORB:
    """Create and return an ORB detector/descriptor instance.

    Args:
        n_features: Maximum number of features to retain.

    Returns:
        A configured ``cv2.ORB`` instance.
    """
    detector = cv2.ORB.create(nfeatures=n_features)
    logger.info("Created ORB detector (nfeatures=%d)", n_features)
    return detector


def compute_keypoints_and_descriptors(
    detector: cv2.ORB,
    image_gray: MatLike,
) -> tuple[list[cv2.KeyPoint], MatLike]:
    """Compute keypoints and descriptors for a grayscale image.

    Args:
        detector: An ORB detector instance (from :func:`build_orb_detector`).
        image_gray: Single-channel preprocessed image.

    Returns:
        A tuple ``(keypoints, descriptors)``.
    """
    img = np.asarray(image_gray)
    mask: MatLike = np.full(img.shape[:2], 255, dtype=np.uint8)
    kp_seq, descriptors = detector.detectAndCompute(img, mask)
    keypoints: list[cv2.KeyPoint] = list(kp_seq)
    h, w = img.shape[:2]
    logger.info("Detected %d keypoints (image %dx%d)", len(keypoints), w, h)
    return keypoints, descriptors


def match_descriptors(
    roi_descriptors: MatLike,
    scene_descriptors: MatLike,
    ratio_threshold: float = 0.75,
) -> list[cv2.DMatch]:
    """TODO: Match ROI descriptors against scene descriptors with filtering.

    Args:
        roi_descriptors: Descriptor array for the ROI image.
        scene_descriptors: Descriptor array for the full scene image.
        ratio_threshold: Lowe ratio test threshold.

    Returns:
        A filtered list of good ``cv2.DMatch`` objects.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError("TODO: implement descriptor matching")


def estimate_roi_center(
    good_matches: list[cv2.DMatch],
    scene_keypoints: list[cv2.KeyPoint],
) -> tuple[int, int]:
    """TODO: Estimate the ROI centre position in the scene from matches.

    Args:
        good_matches: Filtered descriptor matches.
        scene_keypoints: Keypoints detected in the full scene image.

    Returns:
        ``(x, y)`` pixel coordinates of the estimated ROI centre.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError("TODO: implement ROI center estimation")


def center_to_bounding_box(
    center_xy: tuple[int, int],
    roi_shape: tuple[int, ...],
    scene_shape: tuple[int, ...],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert a centre point to a clamped bounding-box rectangle.

    The rectangle has the same width/height as the ROI and is clamped so it
    does not exceed the scene boundaries.

    Args:
        center_xy: ``(x, y)`` centre coordinates.
        roi_shape: Shape of the ROI image (``height, width[, channels]``).
        scene_shape: Shape of the full scene image.

    Returns:
        ``((x1, y1), (x2, y2))`` — top-left and bottom-right corners.
    """
    cx, cy = center_xy
    roi_h, roi_w = roi_shape[:2]
    scene_h, scene_w = scene_shape[:2]

    x1 = max(0, cx - roi_w // 2)
    y1 = max(0, cy - roi_h // 2)
    x2 = min(scene_w - 1, x1 + roi_w)
    y2 = min(scene_h - 1, y1 + roi_h)
    logger.debug(
        "Bounding box for centre (%d, %d): (%d, %d)–(%d, %d)",
        cx, cy, x1, y1, x2, y2,
    )
    return (x1, y1), (x2, y2)


def draw_pair_debug_figure(
    roi_name: str,
    roi_bgr: MatLike,
    full_bgr: MatLike,
    roi_keypoints: list[cv2.KeyPoint],
    full_keypoints: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    output_path: Path,
) -> None:
    """TODO: Save a per-ROI visualisation figure for analysis/debugging.

    Args:
        roi_name: Human-readable name for the ROI (used in titles/filenames).
        roi_bgr: The ROI image in BGR colour order.
        full_bgr: The full scene image in BGR colour order.
        roi_keypoints: Keypoints detected in the ROI.
        full_keypoints: Keypoints detected in the scene.
        good_matches: Filtered matches linking ROI ↔ scene keypoints.
        output_path: File path where the figure should be saved.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError("TODO: implement pair debug figure generation")


def run_pipeline() -> None:
    """Execute the Homework 2 pipeline scaffold with explicit TODO checkpoints."""
    root = get_project_root()
    main_image_path = root / "data" / "noisy_motherboard.jpg"
    roi_dir = root / "data" / "roi" / "dobosi_peter"

    output_dir = root / "output"
    matches_dir = output_dir / "matches"
    output_dir.mkdir(parents=True, exist_ok=True)
    matches_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Project root : %s", root)
    logger.info("Scene image  : %s", main_image_path)
    logger.info("ROI directory: %s", roi_dir)
    logger.info("Output dir   : %s", output_dir)

    full_bgr, roi_images = load_inputs(
        main_image_path=main_image_path,
        roi_dir=roi_dir,
    )

    # --- Subtask 1: Preprocessing ------------------------------------------------
    logger.info("Preprocessing scene image…")
    full_gray = preprocess_image_for_features(full_bgr)

    roi_preprocessed: list[tuple[str, MatLike, MatLike]] = []
    for name, roi_bgr in roi_images:
        logger.info("Preprocessing ROI '%s'…", name)
        roi_gray = preprocess_image_for_features(roi_bgr)
        roi_preprocessed.append((name, roi_bgr, roi_gray))

    # --- Subtask 2: Keypoints & descriptors --------------------------------------
    detector = build_orb_detector()

    logger.info("Computing keypoints & descriptors for scene image…")
    scene_kp, _scene_desc = compute_keypoints_and_descriptors(detector, full_gray)

    roi_features: list[tuple[str, MatLike, list[cv2.KeyPoint], MatLike]] = []
    for name, roi_bgr, roi_gray in roi_preprocessed:
        logger.info("Computing keypoints & descriptors for ROI '%s'…", name)
        kp, desc = compute_keypoints_and_descriptors(detector, roi_gray)
        roi_features.append((name, roi_bgr, kp, desc))

    logger.info(
        "Subtasks 1–2 complete: %d ROIs processed, %d scene keypoints.",
        len(roi_features),
        len(scene_kp),
    )
    logger.info("TODO 3: implement matching + filtering")
    logger.info("TODO 4: implement localisation + drawing")
    logger.info("TODO 5: implement pair-level debug visualisations")
    logger.info("TODO 6: save final localisation image")


if __name__ == "__main__":
    run_pipeline()
