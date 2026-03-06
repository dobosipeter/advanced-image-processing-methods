import logging
from pathlib import Path
import cv2
import numpy as np
import matplotlib

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """ The main method of the module, started when the module is started. """
    test_imports()
    preprocessed_image: np.ndarray = preprocessing()
    global_histogram_equalization(preprocessed_image)

def test_imports():
    """ Test that all libraries are available. """
    try:
        logger.debug("OpenCV version: %s", cv2.__version__)
        logger.debug("Numpy version: %s", np.__version__)
        logger.debug("Matplotlib version: %s", matplotlib.__version__)
    except Exception as e:
        logger.exception("Unable to get version info of all dependencies, are you sure they are all installed? Check the docs. %s", e)

def preprocessing() -> np.ndarray:
    """
    Complete subtask 1: load the image, rotate it to make the text horizontal, scale/crop it to remove the black borders.

    Returns:
        np.ndarray: The preprocessed image.
    """
    logger.info("Starting preprocessing...")

    # Determine image path
    project_root: Path = Path(__file__).resolve().parents[2]
    image_path: Path = project_root / "data" / "dobosi_peter_laszlo.jpg"

    # Load the image
    dpl_image: np.ndarray | None = cv2.imread(str(image_path), flags=cv2.IMREAD_COLOR) # slides say flag, typo
    assert dpl_image is not None, (
        f"Failed to load image from '{image_path}'. cwd='{Path.cwd()}'."
    )
    logger.debug("Loaded image shape: %s", dpl_image.shape)

    show_image(dpl_image, "dpl_image")

    def rotate_crop_to_fit(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate the image by the given angle while scaling it to avoid black borders.

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

        rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale) # slides say that default values are available, but that is not the case...

        return cv2.warpAffine(image, rotation_matrix, (width, height))

    dpl_image_rotated: np.ndarray = rotate_crop_to_fit(dpl_image, -9.41) # eye-balled the angle, probably could be calculated more precisely
    show_image(dpl_image_rotated, "dpl_image_rotated")
    logger.info("Preprocessing completed.")

    return dpl_image_rotated

def global_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Complete subtask 2:
        * convert the image to HSV color space
        * perform global histogram equalization on the V (Value) channel of the preprocessed image
        * blend the V channel with the original V channel linearly to avoid too high contrast
        * plot the result on a color picture

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
    non_blended_rgb_image: np.ndarray = cv2.cvtColor(non_blended_hsv_image, cv2.COLOR_HSV2BGR)
    show_image(non_blended_rgb_image, "non_blended_rgb")

    # Blended result
    blended_value: np.ndarray = cv2.addWeighted(value, 0.5, equalized_value, 0.5, 0)
    blended_hsv_image: np.ndarray = cv2.merge((hue, saturation, blended_value))
    blended_rgb_image: np.ndarray = cv2.cvtColor(blended_hsv_image, cv2.COLOR_HSV2BGR)
    show_image(blended_rgb_image, "blended_rgb")

    return blended_rgb_image

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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_value: np.ndarray = clahe.apply(value)

    # Merge the channels back
    clahe_hsv_image: np.ndarray = cv2.merge((hue, saturation, clahe_value))
    clahe_rgb_image: np.ndarray = cv2.cvtColor(clahe_hsv_image, cv2.COLOR_HSV2BGR)
    show_image(clahe_rgb_image, "clahe_rgb")

    return clahe_rgb_image

def calculate_histograms(image: np.ndarray) -> list[np.ndarray]:
    """
    Calculate the histograms for the R, G, and B channels of the given image.

    Args:
        image (np.ndarray): The input image to calculate the histograms for.

    Returns:
        list[np.ndarray]: A list containing the histograms for the R, G, and B channels.
    """
    channels = cv2.split(image)
    histograms = [cv2.calcHist([channel], [0], None, [256], [0, 256]) for channel in channels]
    return histograms

def plot_results(images: list[np.ndarray], histograms: list[np.ndarray], titles: list[str]):
    """
    """

def show_image(image: np.ndarray, title: str = "Image"):
    """ Utility function to show an image using OpenCV. """
    logger.info("Showing an image, press any key to continue...")
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
