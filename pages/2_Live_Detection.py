import io
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "yolo26n.pt"

IMAGE_SIZE = 640
GAMMA = 0.70
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

torch.set_num_threads(6)

st.set_page_config(
    page_title="NightVision AI",
    page_icon="🌙",
    layout="wide"
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


def apply_clahe(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )

    enhanced_y = clahe.apply(y)
    enhanced = cv2.merge((enhanced_y, cr, cb))

    return cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)


def apply_gamma(image):
    lut = np.array(
        [((i / 255.0) ** GAMMA) * 255 for i in range(256)],
        dtype=np.uint8
    )
    return cv2.LUT(image, lut)


def run_detection(model, image, confidence):
    start_time = time.perf_counter()

    result = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=confidence,
        device="cpu",
        half=False,
        verbose=False
    )[0]

    inference_time = time.perf_counter() - start_time

    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, class_id, score in zip(boxes, classes, confidences):
            detections.append({
                "Class": model.names[class_id],
                "Confidence": round(float(score), 4),
                "X1": round(float(box[0]), 2),
                "Y1": round(float(box[1]), 2),
                "X2": round(float(box[2]), 2),
                "Y2": round(float(box[3]), 2)
            })

    return annotated_rgb, detections, inference_time


st.title("🌙 NightVision AI")
st.subheader("Low-Light Image Enhancement and Object Detection")

st.markdown(
    "Compare object detection on an original low-light image "
    "against CLAHE and Gamma enhancement using YOLO26n."
)

st.info(
    "This demonstration uses a COCO-pretrained YOLO26n model. "
    "It compares inference outputs; it does not calculate detection "
    "accuracy without ground-truth annotations."
)

with st.sidebar:
    st.header("Detection Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.25,
        step=0.05
    )

    st.caption(f"Image size: {IMAGE_SIZE} × {IMAGE_SIZE}")
    st.caption("Inference device: CPU")

uploaded_file = st.file_uploader(
    "Upload a low-light image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_image)
        original_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        variants = {
            "Original": original_bgr,
            "CLAHE": apply_clahe(original_bgr),
            "Gamma": apply_gamma(original_bgr)
        }

        st.subheader("Input Image")
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

        if st.button("Run NightVision AI", type="primary", use_container_width=True):
            with st.spinner("Loading YOLO26n and running detection..."):
                model = load_model()

                outputs = {}
                all_rows = []

                for variant_name, variant_image in variants.items():
                    annotated, detections, elapsed = run_detection(
                        model,
                        variant_image,
                        confidence
                    )

                    outputs[variant_name] = {
                        "image": annotated,
                        "detections": detections,
                        "time": elapsed
                    }

                    for detection in detections:
                        all_rows.append({
                            "Variant": variant_name,
                            **detection
                        })

                st.session_state["nightvision_outputs"] = outputs
                st.session_state["nightvision_rows"] = all_rows

        outputs = st.session_state.get("nightvision_outputs")

        if outputs:
            st.divider()
            st.subheader("Detection Comparison")

            columns = st.columns(3)

            for column, (name, output) in zip(columns, outputs.items()):
                with column:
                    st.markdown(f"**{name}**")
                    st.image(
                        output["image"],
                        use_container_width=True
                    )
                    st.metric(
                        "Objects Detected",
                        len(output["detections"])
                    )
                    st.metric(
                        "Inference Time",
                        f"{output['time']:.3f} sec"
                    )

            st.divider()
            st.subheader("Detection Details")

            rows = st.session_state.get("nightvision_rows", [])
            if rows:
                detection_df = pd.DataFrame(rows)
                st.dataframe(
                    detection_df,
                    use_container_width=True,
                    hide_index=True
                )

                csv_data = detection_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Detection Results (CSV)",
                    data=csv_data,
                    file_name="nightvision_detection_results.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No objects detected at the selected confidence threshold.")

    except Exception as error:
        st.error(f"An error occurred: {error}")

else:
    st.caption("Upload an image to begin the comparison.")

