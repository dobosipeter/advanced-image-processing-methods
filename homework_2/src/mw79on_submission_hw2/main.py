# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false

import logging
from pathlib import Path
from typing import Any

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Return the `homework_2` project root directory."""
    return Path(__file__).resolve().parents[2]


def load_inputs(main_image_path: Path, roi_dir: Path) -> tuple[Any, list[tuple[str, Any]]]:
    """Load full image and all ROI images dynamically from the ROI directory."""
    full_image = cv2.imread(str(main_image_path), cv2.IMREAD_COLOR)
    assert full_image is not None, f"Could not load full image: {main_image_path}"

    roi_files = sorted(roi_dir.glob("*.jpg"))
    assert roi_files, f"No ROI images found in: {roi_dir}"

    rois: list[tuple[str, Any]] = []
    for roi_file in roi_files:
        roi_image = cv2.imread(str(roi_file), cv2.IMREAD_COLOR)
        if roi_image is None:
            logger.warning("Skipping unreadable ROI image: %s", roi_file)
            continue
        rois.append((roi_file.stem, roi_image))

    assert rois, f"No readable ROI images found in: {roi_dir}"
    return full_image, rois


def preprocess_image_for_features(image_bgr: Any, blur_kernel_size: tuple[int, int] = (5, 5)) -> Any:
    """Convert to grayscale and apply Gaussian denoising."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, blur_kernel_size, sigmaX=0)
    return denoised


def build_orb_detector(n_features: int = 3000) -> Any:
    """TODO: Create and return the local feature detector/descriptor instance."""
    raise NotImplementedError("TODO: implement ORB detector construction")


def compute_keypoints_and_descriptors(
    detector: Any,
    image_gray: Any,
) -> tuple[list[cv2.KeyPoint], Any]:
    """TODO: Compute keypoints and descriptors for a grayscale image."""
    raise NotImplementedError("TODO: implement keypoint + descriptor extraction")


def match_descriptors(
    roi_descriptors: Any,
    scene_descriptors: Any,
    ratio_threshold: float = 0.75,
) -> list[cv2.DMatch]:
    """TODO: Match descriptors and apply your filtering strategy (e.g., Lowe ratio)."""
    raise NotImplementedError("TODO: implement descriptor matching")


def estimate_roi_center(
    good_matches: list[cv2.DMatch],
    scene_keypoints: list[cv2.KeyPoint],
) -> tuple[int, int]:
    """TODO: Estimate ROI center from filtered matches."""
    raise NotImplementedError("TODO: implement ROI center estimation")


def center_to_bounding_box(
    center_xy: tuple[int, int],
    roi_shape: tuple[int, int, int],
    scene_shape: tuple[int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert center coordinates to a clamped rectangle using ROI dimensions."""
    cx, cy = center_xy
    roi_h, roi_w = roi_shape[:2]
    scene_h, scene_w = scene_shape[:2]

    x1 = max(0, cx - roi_w // 2)
    y1 = max(0, cy - roi_h // 2)
    x2 = min(scene_w - 1, x1 + roi_w)
    y2 = min(scene_h - 1, y1 + roi_h)
    return (x1, y1), (x2, y2)


def draw_pair_debug_figure(
    roi_name: str,
    roi_bgr: Any,
    full_bgr: Any,
    roi_keypoints: list[cv2.KeyPoint],
    full_keypoints: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    output_path: Path,
) -> None:
    """TODO: Save your per-ROI visualization figure for analysis/debugging."""
    raise NotImplementedError("TODO: implement pair debug figure generation")


def run_pipeline() -> None:
    """Execute Homework 2 pipeline scaffold with explicit TODO checkpoints."""
    root = get_project_root()
    main_image_path = root / "data" / "noisy_motherboard.jpg"
    roi_dir = root / "data" / "roi" / "dobosi_peter"

    output_dir = root / "output"
    matches_dir = output_dir / "matches"
    output_dir.mkdir(parents=True, exist_ok=True)
    matches_dir.mkdir(parents=True, exist_ok=True)

    full_bgr, roi_images = load_inputs(main_image_path=main_image_path, roi_dir=roi_dir)
    _ = preprocess_image_for_features(full_bgr)

    logger.info("Loaded %d ROI images and initialized scaffold.", len(roi_images))
    logger.info("TODO 1: implement detector + descriptor extraction functions")
    logger.info("TODO 2: implement matching + filtering")
    logger.info("TODO 3: implement localization + drawing")
    logger.info("TODO 4: implement pair-level debug visualizations")
    logger.info("TODO 5: save final localization image")


if __name__ == "__main__":
    run_pipeline()
