"""
Kernel SVM (K-SVM) training utility for the Brain Tumor MRI Dashboard.

Dataset layout:
dataset/
  glioma/
  meningioma/
  notumor/
  pituitary/

Trains an RBF-kernel SVM using the project's existing PCA, ICA and selector
transformers, then saves models/ksvm.joblib.

IMPORTANT: Use the same preprocessing and data representation used to create
pca.joblib, ica.joblib and selector.joblib. This is a research/demo tool, not
a medically validated diagnostic system.
"""
from pathlib import Path
import argparse
import sys

import cv2
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from common import preprocess_small, CLASSES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_CLASSES = set(CLASSES)


def preprocess_image(path: Path) -> np.ndarray:
    """Convert an MRI image into a flattened 64x64 grayscale vector using project preprocessing."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read image: {path}")
    return preprocess_small(gray).reshape(-1)


def load_dataset(dataset_dir: Path):
    features, labels = [], []
    for folder in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        label = folder.name.lower().replace(" ", "").replace("_", "")
        if label not in ALLOWED_CLASSES:
            print(f"Skipping unknown class folder: {folder.name}")
            continue
        count = 0
        for image_path in folder.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                features.append(preprocess_image(image_path))
                labels.append(label)
                count += 1
            except Exception as exc:
                print(f"Skipping {image_path}: {exc}")
        print(f"{label}: loaded {count} images")
    if not features:
        raise ValueError("No images found. Check the dataset and class-folder names.")
    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def main():
    default_data = "dataset"
    if not Path(default_data).is_dir():
        fallback = Path("archive (1)") / "Training"
        if fallback.is_dir():
            default_data = str(fallback)

    parser = argparse.ArgumentParser(description="Train an RBF Kernel SVM for MRI classification")
    parser.add_argument("--data", default=default_data, help="Folder containing one subfolder per class")
    parser.add_argument("--models", default="models", help="Folder containing pca/ica/selector joblib files")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_dir, models_dir = Path(args.data), Path(args.models)
    if not data_dir.is_dir():
        sys.exit(f"Dataset folder not found: {data_dir}")
    required = ["pca.joblib", "ica.joblib", "selector.joblib"]
    missing = [name for name in required if not (models_dir / name).is_file()]
    if missing:
        sys.exit("Missing transformer files in " + str(models_dir) + ": " + ", ".join(missing))

    X, y = load_dataset(data_dir)
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        sys.exit("Need at least two classes and at least two images in each class.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pca = joblib.load(models_dir / "pca.joblib")
    ica = joblib.load(models_dir / "ica.joblib")
    selector = joblib.load(models_dir / "selector.joblib")
    try:
        X_train = selector.transform(ica.transform(pca.transform(X_train)))
        X_test = selector.transform(ica.transform(pca.transform(X_test)))
    except Exception as exc:
        sys.exit(
            f"Existing PCA/ICA/selector files are incompatible with this preprocessing: {exc}\n"
            "Use the exact preprocessing and feature format used to train those files."
        )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                    class_weight="balanced", random_state=args.random_state)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Holdout accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, zero_division=0))

    models_dir.mkdir(parents=True, exist_ok=True)
    output = models_dir / "ksvm.joblib"
    joblib.dump(model, output)
    print(f"Saved model: {output.resolve()}")
    print("At inference, apply PCA -> ICA -> selector, then call ksvm.predict_proba(features).")


if __name__ == "__main__":
    main()
