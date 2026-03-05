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
    preprocessing()

def test_imports():
    """ Test that all libraries are available. """
    try:
        logger.debug("OpenCV version: %s", cv2.__version__)
        logger.debug("Numpy version: %s", np.__version__)
        logger.debug("Matplotlib version: %s", matplotlib.__version__)
    except Exception as e:
        logger.exception("Unable to get version info of all dependencies, are you sure they are all installed? Check the docs. %s", e)

def preprocessing():
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

    # Correct image rotation
    rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(
        center=(dpl_image.shape[1] / 2, dpl_image.shape[0] / 2), # slides state that there is a default value, which is not actually the case...
        angle=-9.43,
        scale=1.0
    )
    dpl_image_rotated: np.ndarray = cv2.warpAffine(
        dpl_image,
        rotation_matrix,
        (dpl_image.shape[1], dpl_image.shape[0])
    )
    show_image(dpl_image_rotated, "dpl_image_rotated")

    # Remove the black borders from the image using scaling

def show_image(image: np.ndarray, title: str = "Image"):
    """ Utility function to show an image using OpenCV. """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
