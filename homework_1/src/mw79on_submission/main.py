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


def show_image(image: np.ndarray, title: str = "Image"):
    """ Utility function to show an image using OpenCV. """
    logger.info("Showing an image, press any key to continue...")
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
