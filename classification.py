"""Output 4: classification result (tumor type + confidence for each class)."""
import numpy as np
from common import CLASSES, load_models


def classify(norm_img):
    """norm_img: 64x64 float image from preprocess_small."""
    m = load_models()
    x = m["selector"].transform(m["ica"].transform(m["pca"].transform(norm_img.reshape(1, -1))))
    probs = m["svm"].predict_proba(x)[0]
    k = int(np.argmax(probs))
    return {
        "label": CLASSES[k],
        "confidence": float(probs[k]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
    }
