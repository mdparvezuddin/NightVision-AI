import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="NightVision AI",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent

st.title("🌙 NightVision AI")

st.subheader(
    "Low-Light Image Enhancement and Zero-Shot Object Detection"
)

st.markdown(
    """
    An experimental computer vision system designed to study
    object detection performance under challenging low-light conditions.

    NightVision AI combines image enhancement techniques with
    YOLO26n-based object detection and provides a dedicated
    research results dashboard.
    """
)

st.divider()

st.header("Platform Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Research Results")
    st.write(
        """
        Explore the experimental evaluation and research findings.

        - Paired enhancement validation
        - Original vs CLAHE vs Gamma comparison
        - Full zero-shot evaluation
        - Lighting-wise analysis
        - Class-wise detection performance
        """
    )

    st.info("Open Research Results from the sidebar navigation.")

with col2:
    st.markdown("### 🔍 Live Detection Demo")
    st.write(
        """
        Test object detection on your own images.

        - Upload low-light images
        - Compare Original, CLAHE and Gamma
        - View annotated object detections
        - Inspect confidence scores
        - Export detection results as CSV
        """
    )

    st.info("Open Live Detection from the sidebar navigation.")

st.divider()

st.header("Experimental Configuration")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Detection Model", "YOLO26n")
col2.metric("Inference Device", "CPU")
col3.metric("Image Size", "640 × 640")
col4.metric("Enhancement Methods", "3")

st.divider()

st.header("Research Dataset")

col1, col2 = st.columns(2)

with col1:
    st.metric("Paired Validation Images", "1,799")

with col2:
    st.metric("Full Test Images", "2,563")

st.caption(
    "The paired enhancement validation and full zero-shot evaluation "
    "are independent experiments with separately reported metrics."
)

st.divider()

st.markdown(
    """
    **NightVision AI**

    Research and experimental demonstration platform

    Model: YOLO26n | Enhancement: CLAHE and Gamma | Device: CPU
    """
)
