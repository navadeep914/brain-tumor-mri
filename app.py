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

try:
    load_models()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

upload = st.file_uploader("Upload an MRI slice (jpg / png)", type=["jpg", "jpeg", "png"])
if upload is None:
    st.info("Upload an MRI image to see the four outputs.")
    st.stop()

gray = load_gray(upload.getvalue())
if gray is None:
    st.error("That file could not be read as an image.")
    st.stop()

norm = preprocess_small(gray)
result = classify(norm)
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

with row3[0]:
    st.subheader("5. Classification result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted class", result["label"].upper())
    m2.metric("Confidence", f"{result['confidence']*100:.1f}%")
    m3.metric("Tumor status", "POSITIVE" if is_tumor else "NEGATIVE")
    st.bar_chart(pd.Series(result["probabilities"], name="probability"))
