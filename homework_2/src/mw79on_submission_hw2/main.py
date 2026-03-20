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
    """Match ROI descriptors against scene descriptors with Lowe ratio filtering.

    Uses ``cv2.BFMatcher`` with ``NORM_HAMMING`` and ``knnMatch(k=2)``.
    Only matches where the best distance is below *ratio_threshold* times the
    second-best distance are retained.

    Args:
        roi_descriptors: Descriptor array for the ROI image.
        scene_descriptors: Descriptor array for the full scene image.
        ratio_threshold: Lowe ratio test threshold.

    Returns:
        A filtered list of good ``cv2.DMatch`` objects.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(roi_descriptors, scene_descriptors, k=2)
    good: list[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good.append(m)
    logger.info(
        "Matching: %d raw pairs → %d good (ratio=%.2f)",
        len(knn_matches),
        len(good),
        ratio_threshold,
    )
    return good


def estimate_roi_center(
    good_matches: list[cv2.DMatch],
    scene_keypoints: list[cv2.KeyPoint],
) -> tuple[int, int]:
    """Estimate the ROI centre position in the scene from matches.

    Uses a robust outlier-rejection scheme: the median of the matched scene
    keypoint coordinates is computed first as an outlier-resistant initial
    estimate.  Points farther than $2 \times \text{MAD}$ (Median Absolute
    Deviation) from the median on either axis are discarded, and the final
    centre is the mean of the remaining inliers.

    Args:
        good_matches: Filtered descriptor matches.
        scene_keypoints: Keypoints detected in the full scene image.

    Returns:
        ``(x, y)`` pixel coordinates of the estimated ROI centre.

    Raises:
        ValueError: If *good_matches* is empty.
    """
    if not good_matches:
        raise ValueError("Cannot estimate centre from zero matches")
    pts = np.array(
        [scene_keypoints[m.trainIdx].pt for m in good_matches], dtype=np.float64,
    )

    # Robust outlier rejection using Median Absolute Deviation (MAD).
    # The median is a robust initial estimate unaffected by outlier false
    # positives.  Points farther than 2 × MAD from the median on either
    # axis are discarded before computing the final mean.
    median = np.median(pts, axis=0)
    abs_dev = np.abs(pts - median)
    mad = np.median(abs_dev, axis=0)
    # Avoid zero MAD (happens when most points coincide) by using a
    # minimum of 1 pixel.
    mad = np.maximum(mad, 1.0)
    inlier_mask = np.all(abs_dev <= 2.0 * mad, axis=1)
    inliers = pts[inlier_mask]

    if len(inliers) == 0:
        # Fallback to median if all points are rejected (edge case).
        logger.warning("All matches rejected as outliers — falling back to median")
        cx, cy = median
    else:
        cx, cy = inliers.mean(axis=0)

    n_rejected = len(pts) - len(inliers) if len(inliers) > 0 else 0
    center = (int(round(cx)), int(round(cy)))
    logger.info(
        "Estimated ROI centre at (%d, %d) from %d matches (%d inliers, %d outliers rejected)",
        center[0], center[1], len(good_matches), len(inliers), n_rejected,
    )
    return center


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
    """Save a per-ROI visualisation figure showing matched keypoints.

    Uses ``cv2.drawMatches`` to produce a side-by-side image of the ROI and
    the scene with lines connecting each good match.

    Args:
        roi_name: Human-readable name for the ROI (used in logging).
        roi_bgr: The ROI image in BGR colour order.
        full_bgr: The full scene image in BGR colour order.
        roi_keypoints: Keypoints detected in the ROI.
        full_keypoints: Keypoints detected in the scene.
        good_matches: Filtered matches linking ROI ↔ scene keypoints.
        output_path: File path where the figure should be saved.
    """
    vis: MatLike = cv2.drawMatches(
        img1=roi_bgr, keypoints1=roi_keypoints,
        img2=full_bgr, keypoints2=full_keypoints,
        matches1to2=good_matches,
        outImg=np.array([]),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(output_path), vis)
    logger.info("Saved debug figure for '%s' → %s", roi_name, output_path)


def run_pipeline() -> None:
    """Execute the full Homework 2 feature-matching pipeline."""
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
    full_gray = preprocess_image_for_features(full_bgr, sharpen=True)

    roi_preprocessed: list[tuple[str, MatLike, MatLike]] = []
    for name, roi_bgr in roi_images:
        logger.info("Preprocessing ROI '%s'…", name)
        roi_gray = preprocess_image_for_features(roi_bgr, sharpen=True)
        roi_preprocessed.append((name, roi_bgr, roi_gray))

    # --- Subtask 2: Keypoints & descriptors --------------------------------------
    detector = build_orb_detector(n_features=10000)

    logger.info("Computing keypoints & descriptors for scene image…")
    scene_kp, scene_desc = compute_keypoints_and_descriptors(detector, full_gray)

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

    # --- Subtask 3: Feature matching ---------------------------------------------
    roi_matches: list[tuple[str, MatLike, list[cv2.KeyPoint], list[cv2.DMatch]]] = []
    for name, roi_bgr, roi_kp, roi_desc in roi_features:
        logger.info("Matching ROI '%s' against scene…", name)
        good = match_descriptors(roi_desc, scene_desc)
        roi_matches.append((name, roi_bgr, roi_kp, good))

    logger.info("Subtask 3 complete: matching done for %d ROIs.", len(roi_matches))

    # --- Subtask 4a: Localisation ------------------------------------------------
    colors = [
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
        (0, 0, 255),    # red
        (0, 255, 255),  # yellow
    ]
    scene_annotated = np.copy(np.asarray(full_bgr))

    for idx, (name, roi_bgr, roi_kp, good) in enumerate(roi_matches):
        if not good:
            logger.warning("ROI '%s' has 0 good matches — skipping localisation.", name)
            continue

        center = estimate_roi_center(good, scene_kp)
        roi_shape = np.asarray(roi_bgr).shape
        scene_shape = np.asarray(full_bgr).shape
        pt1, pt2 = center_to_bounding_box(center, roi_shape, scene_shape)

        color = colors[idx % len(colors)]
        cv2.rectangle(scene_annotated, pt1, pt2, color, thickness=3)
        logger.info(
            "ROI '%s': box (%d,%d)–(%d,%d), colour=%s",
            name, pt1[0], pt1[1], pt2[0], pt2[1], color,
        )

    final_path = output_dir / "final_localization.png"
    cv2.imwrite(str(final_path), scene_annotated)
    logger.info("Saved final localisation image → %s", final_path)

    # --- Subtask 4b: Per-ROI debug figures ---------------------------------------
    for name, roi_bgr, roi_kp, good in roi_matches:
        debug_path = matches_dir / f"{name}_matches.png"
        draw_pair_debug_figure(name, roi_bgr, full_bgr, roi_kp, scene_kp, good, debug_path)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
