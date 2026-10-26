"""Dashboard: upload an MRI slice and see all four outputs.

Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from common import load_gray, preprocess_small, load_models
from original_mri import get_original
from tumor_highlight import make_highlight, make_segmentation
from feature_map import make_feature_map
from classification import classify

st.set_page_config(page_title="Brain Tumor MRI Dashboard", layout="wide")
st.title("Brain Tumor Detection & Classification")
st.caption("ICA feature extraction + Kernel SVM classifier (glioma / meningioma / pituitary / no tumor)")

with st.sidebar:
    st.header("Settings")
    st.markdown(
        "**About the highlight:** it is a brightness-based region finder, not a trained "
        "segmentation model. It works on bright/enhancing tumors and can miss dark or "
        "cystic ones. The classification is the reliable output."
    )
    st.divider()
    st.subheader("Model Configuration")
    model_choice = st.selectbox(
        "Classifier Engine",
        options=["ksvm", "svm"],
        index=0,
        format_func=lambda x: "Scaled K-SVM Pipeline (C=10, RBF)" if x == "ksvm" else "Standard RBF K-SVM (C=1)",
        help="Select between the Scaled K-SVM pipeline or standard RBF SVM."
    )

try:
    load_models()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

upload = st.file_uploader("Upload an MRI slice (jpg / png)", type=["jpg", "jpeg", "png"])
if upload is None:
    st.info("Upload an MRI image to see the outputs and K-SVM diagnostics.")
    st.stop()

gray = load_gray(upload.getvalue())
if gray is None:
    st.error("That file could not be read as an image.")
    st.stop()

norm = preprocess_small(gray)
result = classify(norm, model_name=model_choice)
original = get_original(gray)
fmap = make_feature_map(norm)
is_tumor = result["label"] != "notumor"
if is_tumor:
    highlighted, highlight_found = make_highlight(original)
    segmentation, segmentation_found = make_segmentation(original)
else:
    highlighted = original
    segmentation = original * 0
    highlight_found = False
    segmentation_found = False

row1 = st.columns(2)
row2 = st.columns(2)
row3 = st.columns(1)

with row1[0]:
    st.subheader("1. Original MRI")
    st.image(original)

with row1[1]:
    st.subheader("2. Tumor highlighting")
    st.image(highlighted)
    if not is_tumor:
        st.info("No tumor predicted, so no highlight is drawn.")
    elif not highlight_found:
        st.info("No highlight region found in this slice.")

with row2[0]:
    st.subheader("3. Tumor segmentation")
    st.image(segmentation)
    if is_tumor and not segmentation_found:
        st.info("No segmented mass found in this slice.")

with row2[1]:
    st.subheader("4. Feature map (ICA)")
    st.image(fmap)

ksvm_meta = result.get("ksvm", {})
decision_scores = ksvm_meta.get("decision_scores", {})
active_df = decision_scores.get(result["label"], 0.0)

with row3[0]:
    st.subheader("5. Classification & K-SVM Intelligence")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted class", result["label"].upper())
    m2.metric("Confidence", f"{result['confidence']*100:.1f}%")
    m3.metric("Tumor status", "POSITIVE" if is_tumor else "NEGATIVE")
    m4.metric("K-SVM Decision Margin", f"{active_df:+.2f}")

    col_prob, col_df = st.columns(2)
    with col_prob:
        st.caption("Class Probability Distribution")
        st.bar_chart(pd.Series(result["probabilities"], name="probability"))
    with col_df:
        st.caption("Hyperplane Decision Function Margin (Signed Distance)")
        if decision_scores:
            st.bar_chart(pd.Series(decision_scores, name="margin distance"))

    with st.expander("🔍 Kernel SVM (K-SVM) Diagnostics & Hyperplane Geometry", expanded=True):
        kd1, kd2, kd3, kd4 = st.columns(4)
        kd1.metric("Kernel Function", ksvm_meta.get("kernel", "RBF"))
        kd2.metric("Regularization (C)", f"{ksvm_meta.get('c_param', 1.0):.1f}")
        kd3.metric("Gamma (γ)", str(ksvm_meta.get("gamma", "scale")))
        kd4.metric("Support Vectors", f"{ksvm_meta.get('total_support_vectors', 0):,}")

        st.markdown(f"**Mathematical Formulation:** `{ksvm_meta.get('kernel_formula', '')}`")
        st.markdown(f"**Active Model:** `{ksvm_meta.get('active_model', '')}`")

        sv_dict = ksvm_meta.get("support_vectors_by_class", {})
        if sv_dict:
            st.caption("Support Vectors per Class")
            sv_cols = st.columns(len(sv_dict))
            for col, (cls_name, sv_count) in zip(sv_cols, sv_dict.items()):
                col.metric(cls_name.capitalize(), f"{sv_count:,}")

        comp = result.get("comparison", {})
        if comp and len(comp) > 1:
            st.divider()
            st.subheader("Model Comparison (Pipeline K-SVM vs Standard K-SVM)")
            comp_rows = []
            for k_name, c_data in comp.items():
                comp_rows.append({
                    "Engine": c_data["model_name"],
                    "Predicted": c_data["label"].upper(),
                    "Confidence": f"{c_data['confidence']*100:.1f}%",
                    **{f"P({c})": f"{c_data['probabilities'].get(c, 0)*100:.1f}%" for c in ["glioma", "meningioma", "pituitary", "notumor"]}
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

        features = ksvm_meta.get("selected_features", [])
        if features:
            st.caption("32 Selected FastICA Features Fed into K-SVM Kernel")
            st.line_chart(pd.Series(features, name="ICA Feature Value"))
