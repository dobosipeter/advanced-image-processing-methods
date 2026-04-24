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
    """Load the left/right stereo images as single-channel grayscale arrays."""
    raise NotImplementedError


# 2. Keypoints, descriptors, matching
def detect_and_describe(image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Detect keypoints and compute descriptors on a grayscale image."""
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


# 4. Uncalibrated stereo rectification
def rectify_uncalibrated(
    points_left: np.ndarray,
    points_right: np.ndarray,
    fundamental_matrix: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the rectifying homographies for the left and right views."""
    raise NotImplementedError


def warp_with_homography(
    image: np.ndarray,
    homography: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Apply a perspective warp to *image* using *homography*."""
    raise NotImplementedError


# 5. Disparity computation
def compute_disparity_sgbm(
    rectified_left: np.ndarray,
    rectified_right: np.ndarray,
) -> np.ndarray:
    """Compute the disparity map with SGBM and return it as ``uint8`` in [0, 255]."""
    raise NotImplementedError


# 6. Visualisation
def save_results_figure(
    rectified_left: np.ndarray,
    rectified_right: np.ndarray,
    disparity: np.ndarray,
    output_path: Path,
) -> None:
    """Save a 1×3 figure: rectified left, rectified right, disparity (colormap)."""
    raise NotImplementedError


# Pipeline
def main() -> None:
    """Run the full homework 4 pipeline end-to-end."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
