import logging
from pathlib import Path
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run the full homework pipeline end-to-end."""
    test_imports()

    preprocessed_image: np.ndarray = preprocessing()
    hist_equalized_image: np.ndarray = global_histogram_equalization(preprocessed_image)
    clahe_equalized_image: np.ndarray = apply_clahe(preprocessed_image)

    images: list[np.ndarray] = [preprocessed_image, hist_equalized_image, clahe_equalized_image]
    histograms: list[list[np.ndarray]] = [calculate_histograms(image) for image in images]
    titles: list[str] = ["Rotated/Cropped", "Global Histogram EQ", "CLAHE"]

    output_dir: Path = Path(__file__).resolve().parents[2] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_plot_path: Path = output_dir / "task4_results.png"

    plot_results(images=images, histograms=histograms, titles=titles, output_path=output_plot_path)
    logger.info("Saved final plot to %s", output_plot_path)


def test_imports():
    """Log dependency versions."""
    try:
        logger.debug("OpenCV version: %s", cv2.__version__)
        logger.debug("Numpy version: %s", np.__version__)
        logger.debug("Matplotlib version: %s", matplotlib.__version__)
    except Exception as e:
        logger.exception("Unable to get version info of all dependencies, are you sure they are all installed? Check the docs. %s", e)


def preprocessing() -> np.ndarray:
    """
    Subtask 1: load, rotate, and crop/scale the input image.

    Returns:
        np.ndarray: The preprocessed image.
    """
    logger.info("Starting preprocessing...")

    # Determine image path
    project_root: Path = Path(__file__).resolve().parents[2]
    image_path: Path = project_root / "data" / "dobosi_peter_laszlo.jpg"

    # Load the image
    dpl_image: np.ndarray | None = cv2.imread(str(image_path), flags=cv2.IMREAD_COLOR)
    assert dpl_image is not None, (
        f"Failed to load image from '{image_path}'. cwd='{Path.cwd()}'."
    )
    logger.debug("Loaded image shape: %s", dpl_image.shape)

    show_image(dpl_image, "dpl_image")

    def rotate_crop_to_fit(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate the image by the given angle while scaling to reduce black borders.

        Args:
            image (np.ndarray): The input image to rotate.
            angle (float): The angle in degrees to rotate the image. Positive values mean counter-clockwise rotation.

        Returns:
            np.ndarray: The rotated and scaled image.
        """
        # Calculate the scale factor to ensure the rotated image fits within the original dimensions
        height, width = image.shape[:2]
        center = (width / 2, height / 2)

        angle_rad = np.radians(angle)
        sin_a = abs(np.sin(angle_rad))
        cos_a = abs(np.cos(angle_rad))

        ratio = max(width / height, height / width)
        scale = cos_a + ratio * sin_a
        logger.debug("Calculated scale factor: %f", scale)

        rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)

        return cv2.warpAffine(image, rotation_matrix, (width, height))

    dpl_image_rotated: np.ndarray = rotate_crop_to_fit(dpl_image, -9.41)
    show_image(dpl_image_rotated, "dpl_image_rotated")
    logger.info("Preprocessing completed.")

    return dpl_image_rotated

def global_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Complete subtask 2:
        * convert the image to HSV color space
        * perform global histogram equalization on the V (Value) channel of the preprocessed image
        * blend the V channel with the original V channel linearly to avoid too high contrast
        * return the final color image

    Args:
        image (np.ndarray): The preprocessed image to perform histogram equalization on.
    
    Returns:
        np.ndarray: The resulting image after histogram equalization and blending.
    """
    # Convert the image to HSV color space
    hsv_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Equalize the histogram of the V channel
    hue, saturation, value = cv2.split(hsv_image)
    equalized_value: np.ndarray = cv2.equalizeHist(value)

    # Non blended result
    non_blended_hsv_image: np.ndarray = cv2.merge((hue, saturation, equalized_value))
    non_blended_bgr_image: np.ndarray = cv2.cvtColor(non_blended_hsv_image, cv2.COLOR_HSV2BGR)
    show_image(non_blended_bgr_image, "non_blended_bgr")

    # Blended result
    blended_value: np.ndarray = cv2.addWeighted(value, 0.5, equalized_value, 0.5, 0)
    blended_hsv_image: np.ndarray = cv2.merge((hue, saturation, blended_value))
    blended_bgr_image: np.ndarray = cv2.cvtColor(blended_hsv_image, cv2.COLOR_HSV2BGR)
    show_image(blended_bgr_image, "blended_bgr")

    return blended_bgr_image

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Complete subtask 3: apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the V channel of the preprocessed image and plot the result.

    Args:
        image (np.ndarray): The preprocessed image to apply CLAHE on.

    Returns:
        np.ndarray: The resulting image after applying CLAHE.
    """
    # Convert the image to HSV color space
    hsv_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply CLAHE to the V channel
    hue, saturation, value = cv2.split(hsv_image)
    clahe = cv2.createCLAHE(clipLimit=1.205, tileGridSize=(9, 9))
    clahe_value: np.ndarray = clahe.apply(value)

    # Merge the channels back
    clahe_hsv_image: np.ndarray = cv2.merge((hue, saturation, clahe_value))
    clahe_bgr_image: np.ndarray = cv2.cvtColor(clahe_hsv_image, cv2.COLOR_HSV2BGR)
    show_image(clahe_bgr_image, "clahe_bgr")

    return clahe_bgr_image

def calculate_histograms(image: np.ndarray) -> list[np.ndarray]:
    """
    Calculate R, G, B histograms of a BGR OpenCV image.

    Args:
        image (np.ndarray): The input image to calculate the histograms for.

    Returns:
        list[np.ndarray]: A list containing the histograms for the R, G, and B channels.
    """
    rgb_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    channels = cv2.split(rgb_image)
    histograms = [cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten() for channel in channels]
    return histograms

def plot_results(
    images: list[np.ndarray],
    histograms: list[list[np.ndarray]],
    titles: list[str],
    output_path: Path,
) -> None:
    """
    Subtask 4: create the required 2x3 summary figure.

    Top row: original rotated, global EQ, CLAHE images.
    Bottom row: corresponding RGB histograms.

    Args:
        images (list[np.ndarray]): Images in BGR order.
        histograms (list[list[np.ndarray]]): RGB histograms for each image.
        titles (list[str]): Column titles.
        output_path (Path): Path of the saved summary figure.
    """
    assert len(images) == 3, "Exactly 3 images are required."
    assert len(histograms) == 3, "Exactly 3 histogram sets are required."
    assert len(titles) == 3, "Exactly 3 titles are required."

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    x_values = np.arange(256)
    channel_colors = ["r", "g", "b"]
    channel_labels = ["R", "G", "B"]

    for col in range(3):
        # Plot image in RGB for Matplotlib.
        axes[0, col].imshow(cv2.cvtColor(images[col], cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(f"{titles[col]} Image")
        axes[0, col].axis("off")

        # Plot RGB histograms with required 0-255 x range.
        for channel_idx in range(3):
            axes[1, col].plot(
                x_values,
                histograms[col][channel_idx],
                color=channel_colors[channel_idx],
                label=channel_labels[channel_idx],
            )

        axes[1, col].set_title(f"{titles[col]} RGB Histogram")
        axes[1, col].set_xlim(0, 255)
        axes[1, col].set_xlabel("Intensity (0-255)")
        axes[1, col].set_ylabel("Pixel Count")
        axes[1, col].grid(True, alpha=0.2)
        axes[1, col].legend()

    fig.suptitle("Homework 1 - Image Enhancement Comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def show_image(image: np.ndarray, title: str = "Image"):
    """Display an image with OpenCV."""
    logger.info("Showing image : %s, press any key to continue...", title)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
