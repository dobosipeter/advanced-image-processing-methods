"""Stereo disparity map estimation from an uncalibrated stereo image pair.

Pipeline:

1. Load the left/right images as grayscale.
2. Detect keypoints and compute descriptors; match them across the two views.
3. Estimate the fundamental matrix with RANSAC, keeping only inliers.
4. Compute uncalibrated rectifying homographies and warp both images.
5. Compute a dense disparity map with Semi-Global Block Matching (SGBM)
   and normalise it to ``uint8`` in ``[0, 255]``.
6. Save a 1×3 visualisation: rectified left, rectified right, disparity (colormap).
"""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Return the ``homework_4`` project root directory."""
    return Path(__file__).resolve().parents[2]


# 1. Image loading
def load_stereo_pair(
    left_path: Path, right_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load the left/right stereo images as single-channel grayscale arrays.

    Both files are read with ``cv2.IMREAD_GRAYSCALE`` so the rest of the
    pipeline operates on ``uint8`` 2-D arrays.  The two images are required
    to have identical shape because the downstream rectification step
    assumes a single common ``(width, height)``.

    Args:
        left_path: Path to the left view PNG.
        right_path: Path to the right view PNG.

    Returns:
        Tuple ``(left_gray, right_gray)`` of single-channel ``uint8`` arrays.
    """
    left: cv2.typing.MatLike | None = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right: cv2.typing.MatLike | None = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

    assert left is not None, f"Failed to load left image: {left_path}"
    assert right is not None, f"Failed to load right image: {right_path}"
    assert left.shape == right.shape, (
        f"Stereo pair shape mismatch: left {left.shape} vs right {right.shape}"
    )

    h, w = left.shape[:2]
    logger.info("Loaded left  image: %s  (%dx%d, %s)", left_path.name, w, h, left.dtype)
    logger.info("Loaded right image: %s  (%dx%d, %s)", right_path.name, w, h, right.dtype)
    return left, right


# 2. Keypoints, descriptors, matching
def detect_and_describe(image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Detect keypoints and compute descriptors on a grayscale image."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    logger.info("Detected %d keypoints", len(keypoints))
    return keypoints, descriptors


def match_and_extract_points(
    keypoints_left: list[cv2.KeyPoint],
    descriptors_left: np.ndarray,
    keypoints_right: list[cv2.KeyPoint],
    descriptors_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match descriptors, sort by distance and return paired ``(x, y)`` arrays.

    Returns:
        Tuple ``(points_left, points_right)`` as ``float32`` arrays of shape
        ``(N, 2)`` containing the matched pixel coordinates from each view.
    """
    index_params = {"algorithm": 1, "trees": 5}
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)

    good: list[cv2.DMatch] = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    good.sort(key=lambda m: m.distance)
    logger.info("Good matches after Lowe's ratio test: %d", len(good))

    pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in good])
    pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in good])
    return pts_left, pts_right


# 3. Fundamental matrix estimation
def estimate_fundamental_matrix(
    points_left: np.ndarray,
    points_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the fundamental matrix with RANSAC and return inlier point sets.

    Returns:
        ``(F, inlier_points_left, inlier_points_right)`` where ``F`` is the
        ``3x3`` fundamental matrix and the inlier arrays contain only the
        point pairs whose mask value is ``1``.
    """
    F, mask = cv2.findFundamentalMat(
        points_left, points_right,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.99,
    )
    inlier_mask = np.ravel(mask) == 1
    inlier_left = points_left[inlier_mask]
    inlier_right = points_right[inlier_mask]

    assert F.shape == (3, 3), f"Unexpected F shape: {F.shape}"
    assert inlier_left.shape[0] >= 8, (
        f"Only {inlier_left.shape[0]} inliers — need at least 8"
    )
    logger.info(
        "Fundamental matrix estimated — %d / %d inliers",
        inlier_left.shape[0], points_left.shape[0],
    )
    return F, inlier_left, inlier_right


# 4. Uncalibrated stereo rectification
def rectify_uncalibrated(
    points_left: np.ndarray,
    points_right: np.ndarray,
    fundamental_matrix: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the rectifying homographies for the left and right views."""
    retval, H_left, H_right = cv2.stereoRectifyUncalibrated(
        points_left, points_right, fundamental_matrix, image_size,
    )
    assert retval, "stereoRectifyUncalibrated failed"
    logger.info("Rectifying homographies computed")
    return H_left, H_right


def warp_with_homography(
    image: np.ndarray,
    homography: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Apply a perspective warp to *image* using *homography*."""
    return cv2.warpPerspective(image, homography, image_size)


# 5. Disparity computation
def compute_disparity_sgbm(
    rectified_left: np.ndarray,
    rectified_right: np.ndarray,
) -> np.ndarray:
    """Compute the disparity map with SGBM and return it as ``uint8`` in [0, 255]."""
    filtered_left = cv2.bilateralFilter(rectified_left, 9, 75, 75)
    filtered_right = cv2.bilateralFilter(rectified_right, 9, 75, 75)

    block_size = 11
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,
        blockSize=block_size,
        P1=8 * 1 * block_size ** 2,
        P2=32 * 1 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=200,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )

    disparity_raw = sgbm.compute(filtered_left, filtered_right)
    disparity_valid = disparity_raw.copy()
    disparity_valid[disparity_raw < 0] = 0
    disparity_norm = cv2.normalize(
        disparity_valid, None,
        alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX,
    )
    disparity_uint8 = np.uint8(disparity_norm)
    logger.info("Disparity map computed — range [%d, %d], %.1f%% valid pixels",
                disparity_raw.min(), disparity_raw.max(),
                100 * (disparity_raw >= 0).sum() / disparity_raw.size)
    return disparity_uint8


# 6. Visualisation
def save_results_figure(
    rectified_left: np.ndarray,
    rectified_right: np.ndarray,
    disparity: np.ndarray,
    output_path: Path,
) -> None:
    """Save a 1×3 figure: rectified left, rectified right, disparity (colormap)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rectified_left, cmap="gray")
    axes[0].set_title("Rectified Left")
    axes[0].axis("off")

    axes[1].imshow(rectified_right, cmap="gray")
    axes[1].set_title("Rectified Right")
    axes[1].axis("off")

    im = axes[2].imshow(disparity, cmap="magma")
    axes[2].set_title("Disparity (brighter ≈ closer)")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Results figure saved to %s", output_path)


# Pipeline
def main() -> None:
    """Run the full homework 4 pipeline end-to-end."""
    root = get_project_root()
    data_dir = root / "data" / "dobosi_peter_laszlo"
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    left_path = data_dir / "im0.png"
    right_path = data_dir / "im1.png"

    assert data_dir.is_dir(), f"Stereo image directory not found: {data_dir}"
    logger.info("Project root : %s", root)
    logger.info("Stereo dir   : %s", data_dir)
    logger.info("Output dir   : %s", output_dir)

    # 1. Load stereo pair
    left_gray, right_gray = load_stereo_pair(left_path, right_path)
    logger.info("Subtask 1 complete.")

    # 2. Keypoint detection & matching
    kp_left, desc_left = detect_and_describe(left_gray)
    kp_right, desc_right = detect_and_describe(right_gray)
    pts_left, pts_right = match_and_extract_points(
        kp_left, desc_left, kp_right, desc_right,
    )
    logger.info("Subtask 2 complete — %d matched point pairs.", len(pts_left))

    # 3. Fundamental matrix estimation
    F, inlier_left, inlier_right = estimate_fundamental_matrix(pts_left, pts_right)
    logger.info("Subtask 3 complete.")

    # 4. Uncalibrated stereo rectification
    h, w = left_gray.shape[:2]
    image_size = (w, h)
    H_left, H_right = rectify_uncalibrated(
        inlier_left, inlier_right, F, image_size,
    )
    rect_left = warp_with_homography(left_gray, H_left, image_size)
    rect_right = warp_with_homography(right_gray, H_right, image_size)
    cv2.imwrite(str(output_dir / "rectified_left.png"), rect_left)
    cv2.imwrite(str(output_dir / "rectified_right.png"), rect_right)
    logger.info("Subtask 4 complete — rectified images saved to output/.")

    # 5. Disparity computation
    disparity = compute_disparity_sgbm(rect_left, rect_right)
    cv2.imwrite(str(output_dir / "disparity.png"), disparity)
    logger.info("Subtask 5 complete — disparity map saved to output/.")

    # 6. Visualisation
    save_results_figure(
        rect_left, rect_right, disparity,
        output_dir / "disparity_results.png",
    )
    logger.info("Subtask 6 complete — all done.")


if __name__ == "__main__":
    main()
