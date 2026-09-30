"""Output 3: ICA feature map - the image rebuilt from its independent components."""
import cv2
import numpy as np
from common import IMG_SIZE, DISPLAY_SIZE, load_models


def make_feature_map(norm_img):
    """norm_img: 64x64 float image from preprocess_small. Returns an RGB heatmap."""
    m = load_models()
    x_pca = m["pca"].transform(norm_img.reshape(1, -1))
    coeffs = m["ica"].transform(x_pca)
    recon = m["pca"].inverse_transform(m["ica"].inverse_transform(coeffs))
    fmap = recon.reshape(IMG_SIZE, IMG_SIZE)
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
    heat = cv2.applyColorMap((fmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.resize(heat, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC)
