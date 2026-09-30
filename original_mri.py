"""Output 1: the original MRI, resized for display."""
import cv2
from common import DISPLAY_SIZE


def get_original(gray):
    """Return the MRI as an RGB uint8 image at display size."""
    img = cv2.resize(gray, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
