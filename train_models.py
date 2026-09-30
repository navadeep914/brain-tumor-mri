"""Train and save the model pieces used by the Streamlit dashboard."""
import argparse
import os

import cv2
import joblib
import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

from common import CLASSES, preprocess_small


def load_dataset(data_dir):
    features = []
    labels = []
    for label_index, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for filename in sorted(os.listdir(class_dir)):
            path = os.path.join(class_dir, filename)
            if not os.path.isfile(path):
                continue
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            features.append(preprocess_small(image).reshape(-1))
            labels.append(label_index)
    if not features:
        raise ValueError(f"No readable images found in {data_dir}")
    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def train(data_dir, model_dir):
    x, y = load_dataset(data_dir)
    pca = PCA(n_components=128, whiten=True, random_state=42)
    x_pca = pca.fit_transform(x)
    ica = FastICA(n_components=64, random_state=42, max_iter=300, whiten="unit-variance")
    x_ica = ica.fit_transform(x_pca)
    selector = SelectKBest(f_classif, k=32)
    x_selected = selector.fit_transform(x_ica, y)
    svm = SVC(probability=True, class_weight="balanced", random_state=42)
    svm.fit(x_selected, y)

    os.makedirs(model_dir, exist_ok=True)
    for name, model in (("pca", pca), ("ica", ica), ("selector", selector), ("svm", svm)):
        joblib.dump(model, os.path.join(model_dir, f"{name}.joblib"))
    print(f"Saved models for {len(y)} images to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join("archive (1)", "Training"))
    parser.add_argument("--models", default="models")
    args = parser.parse_args()
    train(args.data, args.models)