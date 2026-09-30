"""Shared helpers: constants, image loading, the 64x64 preprocessing used for
training, and trained-model loading. The four output modules import from here."""
import os
from functools import lru_cache

import cv2
import joblib
import numpy as np

IMG_SIZE = 64        # classifier input size (must match training)
WORK_SIZE = 128      # preprocessing working size (must match training)
DISPLAY_SIZE = 448   # size used for the picture panels
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
)


def load_gray(image_bytes):
    """Decode uploaded bytes into a grayscale uint8 array (None if unreadable)."""
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def skull_strip(gray_img):
    _, thresh = cv2.threshold(gray_img, 12, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    else:
        mask = thresh
    return cv2.bitwise_and(gray_img, gray_img, mask=mask)


def preprocess_small(gray):
    """Same steps as training: resize -> denoise -> strip -> CLAHE -> 64x64 -> [0,1]."""
    img = cv2.resize(gray, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.fastNlMeansDenoising(img, h=10)
    img = skull_strip(img)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


@lru_cache(maxsize=1)
def load_models():
    """Load the four trained pipeline pieces saved by the notebook."""
    names = ["pca", "ica", "selector", "svm"]
    missing = [n for n in names if not os.path.exists(os.path.join(MODEL_DIR, f"{n}.joblib"))]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in '{MODEL_DIR}'. Copy the 'models' folder your notebook "
            "created (pca/ica/selector/svm .joblib) next to app.py."
        )
    return {n: joblib.load(os.path.join(MODEL_DIR, f"{n}.joblib")) for n in names}
