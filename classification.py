"""Output 4: classification result (tumor type + confidence for each class + K-SVM diagnostics)."""
import numpy as np
from common import CLASSES, load_models


def extract_ksvm_info(model, x_features):
    """Extract Kernel SVM diagnostics: support vectors, margin distances, RBF params."""
    svc = model.named_steps["svm"] if hasattr(model, "named_steps") else model
    kernel = getattr(svc, "kernel", "rbf")
    c_val = getattr(svc, "C", 1.0)
    gamma_val = getattr(svc, "gamma", "scale")
    n_supp = getattr(svc, "n_support_", None)
    total_supp = int(np.sum(n_supp)) if n_supp is not None else 0
    supp_dict = {c: int(n) for c, n in zip(CLASSES, n_supp)} if n_supp is not None else {}

    decision_scores = {}
    if hasattr(model, "decision_function"):
        df = model.decision_function(x_features)[0]
        if isinstance(df, np.ndarray) and len(df) == len(CLASSES):
            decision_scores = {c: float(round(v, 4)) for c, v in zip(CLASSES, df)}

    return {
        "kernel": kernel.upper() if isinstance(kernel, str) else "RBF",
        "kernel_formula": "K(x, x') = exp(-γ · ||x - x'||²)",
        "c_param": float(c_val),
        "gamma": str(gamma_val),
        "total_support_vectors": total_supp,
        "support_vectors_by_class": supp_dict,
        "decision_scores": decision_scores,
        "selected_features": [round(float(v), 4) for v in x_features[0][:32]],
    }


def classify(norm_img, model_name="ksvm"):
    """norm_img: 64x64 float image from preprocess_small.
    model_name: 'ksvm' or 'svm' (defaults to 'ksvm' if available, otherwise 'svm').
    """
    m = load_models()
    x = m["selector"].transform(m["ica"].transform(m["pca"].transform(norm_img.reshape(1, -1))))

    # Default to ksvm if available, otherwise fallback to svm
    active_key = model_name if model_name in m else ("ksvm" if "ksvm" in m else "svm")
    model = m[active_key]
    probs = model.predict_proba(x)[0]
    k = int(np.argmax(probs))

    # Evaluate comparison metrics across both models
    comparison = {}
    for key in ["ksvm", "svm"]:
        if key in m:
            km = m[key]
            kp = km.predict_proba(x)[0]
            kk = int(np.argmax(kp))
            comparison[key] = {
                "label": CLASSES[kk],
                "confidence": float(kp[kk]),
                "probabilities": {c: float(p) for c, p in zip(CLASSES, kp)},
                "model_name": "Scaled K-SVM Pipeline (C=10)" if key == "ksvm" else "Standard RBF K-SVM (C=1)",
            }

    ksvm_info = extract_ksvm_info(model, x)
    ksvm_info["active_model"] = "Scaled K-SVM Pipeline (C=10)" if active_key == "ksvm" else "Standard RBF K-SVM (C=1)"
    ksvm_info["active_key"] = active_key
    ksvm_info["has_ksvm_pipeline"] = "ksvm" in m

    return {
        "label": CLASSES[k],
        "confidence": float(probs[k]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
        "ksvm": ksvm_info,
        "comparison": comparison,
    }
