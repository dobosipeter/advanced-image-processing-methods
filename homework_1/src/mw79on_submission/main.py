import logging
import cv2
import numpy as np
import matplotlib

logger = logging.getLogger(__name__)

def main():
    """ The main method of the module, started when the module is started. """
    test_imports()
    preprocessing()

def test_imports():
    """ Test that all libraries are available. """
    try:
        print("OpenCV version:", cv2.__version__)
        print("Numpy version:", np.__version__)
        print("Matplotlib version:", matplotlib.__version__)
    except Exception as e:
        logger.exception("Unable to get version info of all dependencies, are you sure they are all installed? Check the docs. %s", e)


def preprocessing():
    dpl_image = cv2.imread("data/dobosi_peter_laszlo.jpg", flag=cv2.IMREAD_COLOR)
    cv2.imshow("dpl_image", dpl_image)
    cv2.waitkey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
