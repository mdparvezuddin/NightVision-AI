
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "exdark_audit_output"

PAIRED = OUTPUT / "paired_enhancement_validation"
FULL = OUTPUT / "zero_shot_full_test"

st.set_page_config(
    page_title="NightVision AI | Research Dashboard",
    page_icon="🌙",
    layout="wide"
)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path):
    return pd.read_csv(path)


def percent(value):
    return f"{value * 100:.2f}%"


# ---------------------------------------------------------
# DASHBOARD HEADER
# ---------------------------------------------------------

st.title("🌙 NightVision AI")
st.subheader("Research Results Dashboard")

st.markdown(
    "Experimental evaluation of zero-shot object detection "
    "under low-light conditions using YOLO26n."
)

st.info(
    "The paired enhancement validation and full zero-shot evaluation "
    "are separate experiments. Their metrics are presented independently."
)

overview, paired, full_test, lighting, classes = st.tabs([
    "Experiment Overview",
    "Enhancement Comparison",
    "Full Test Evaluation",
    "Lighting Analysis",
    "Class Analysis"
])


# ---------------------------------------------------------
# EXPERIMENT OVERVIEW
# ---------------------------------------------------------

with overview:

    st.header("Experimental Setup")

    paired_summary = load_json(
        PAIRED / "experiment_summary.json"
    )

    full_summary = load_json(
        FULL / "full_test_summary.json"
    )

    full_evaluation = load_json(
        FULL / "full_test_evaluation_summary.json"
    )

    st.markdown("### Model and configuration")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model", paired_summary["model"])
    col2.metric("Device", paired_summary["device"])
    col3.metric("Confidence", paired_summary["confidence_threshold"])
    col4.metric("Image size", paired_summary["image_size"])

    st.divider()

    st.markdown("### Experiment coverage")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Paired Enhancement Validation**")
        st.metric(
            "Images evaluated",
            f'{paired_summary["images"]:,}'
        )
        st.caption("Original vs CLAHE vs Gamma")

    with col2:
        st.markdown("**Full Zero-Shot Evaluation**")
        st.metric(
            "Images processed",
            f'{full_summary["images_processed"]:,}'
        )
        st.metric(
            "Total predictions",
            f'{full_summary["total_predictions"]:,}'
        )
        st.caption("Independent full-test detector evaluation")

    st.divider()

    st.markdown("### Enhancement parameters")

    st.json(paired_summary["enhancements"])

    st.markdown("### Evaluation configuration")

    st.write({
        "IoU threshold": paired_summary["iou_threshold"],
        "Confidence threshold": paired_summary["confidence_threshold"],
        "Inference device": paired_summary["device"],
        "Full-test inference time (seconds)": full_summary["inference_seconds"]
    })


# ---------------------------------------------------------
# PAIRED ENHANCEMENT COMPARISON
# ---------------------------------------------------------

with paired:

    st.header("Paired Enhancement Validation")
    st.caption("1,799 validation images | YOLO26n | IoU 0.50")

    summary = load_json(
        PAIRED / "experiment_summary.json"
    )

    variants = summary["variants"]

    rows = []

    for name, values in variants.items():

        rows.append({
            "Variant": name.capitalize(),
            "True Positives": values["TP"],
            "False Positives": values["FP"],
            "False Negatives": values["FN"],
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1-score": values["f1"]
        })

    metrics = pd.DataFrame(rows).set_index("Variant")

    c1, c2, c3 = st.columns(3)

    c1.metric("Images", f'{summary["images"]:,}')
    c2.metric("IoU threshold", summary["iou_threshold"])
    c3.metric(
        "Inference time",
        f'{summary["inference_seconds"]:.2f} sec'
    )

    # PERFORMANCE METRICS

    st.markdown("### Performance metrics")

    display_metrics = metrics[
        ["Precision", "Recall", "F1-score"]
    ].mul(100).round(2)

    st.dataframe(
        display_metrics,
        use_container_width=True
    )

    # METRIC COMPARISON CHART

    st.markdown("### Metric comparison")

    st.bar_chart(
        display_metrics,
        stack=None
    )

    # PAIRED IMAGE-LEVEL CHANGES

    st.markdown("### Paired image-level changes")

    comparison = summary["paired_comparison"]

    comparison_rows = []

    for variant, values in comparison.items():

        comparison_rows.append({
            "Enhancement": variant.upper(),
            "Images improved": values["images_improved"],
            "Images unchanged": values["images_unchanged"],
            "Images damaged": values["images_damaged"],
            "Mean miss-rate change (pp)": (
                f'{values["mean_miss_rate_change"] * 100:+.2f}'
            ),
            "Total FN change": values["total_FN_change"]
        })

    st.dataframe(
        pd.DataFrame(comparison_rows),
        use_container_width=True,
        hide_index=True
    )

    # BOOTSTRAP CONFIDENCE INTERVALS

    st.markdown("### Bootstrap confidence intervals")

    bootstrap = load_csv(
        PAIRED / "paired_bootstrap_confidence_intervals.csv"
    )

    selected_variant = st.selectbox(
        "Enhancement",
        ["clahe", "gamma"]
    )

    selected_metric = st.selectbox(
        "Metric",
        ["precision", "recall", "f1"]
    )

    filtered = bootstrap[
        (bootstrap["variant"] == selected_variant) &
        (bootstrap["metric"] == selected_metric)
    ].copy()

    if not filtered.empty:

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        overall = filtered[
            filtered["lighting"].isna()
        ]

        if not overall.empty:

            result = overall.iloc[0]

            st.markdown("**Overall bootstrap result**")

            a, b, c = st.columns(3)

            a.metric(
                "Metric change",
                f'{result["delta"] * 100:.3f} pp'
            )

            b.metric(
                "95% CI lower",
                f'{result["ci_2.5"] * 100:.3f} pp'
            )

            c.metric(
                "95% CI upper",
                f'{result["ci_97.5"] * 100:.3f} pp'
            )

    st.caption(
        "Confidence intervals and bootstrap probabilities are shown "
        "as stored in the original experiment outputs."
    )


# ---------------------------------------------------------
# FULL ZERO-SHOT EVALUATION
# ---------------------------------------------------------

with full_test:

    st.header("Full Zero-Shot Evaluation")
    st.caption("Separate full-test evaluation | 2,563 images")

    summary = load_json(
        FULL / "full_test_summary.json"
    )

    evaluation = load_json(
        FULL / "full_test_evaluation_summary.json"
    )

    a, b, c = st.columns(3)

    a.metric("Precision", percent(evaluation["precision"]))
    b.metric("Recall", percent(evaluation["recall"]))
    c.metric("F1-score", percent(evaluation["f1"]))

    st.markdown("### Detection statistics")

    a, b, c = st.columns(3)

    a.metric("True Positives", f'{evaluation["TP"]:,}')
    b.metric("False Positives", f'{evaluation["FP"]:,}')
    c.metric("False Negatives", f'{evaluation["FN"]:,}')

    st.markdown("### Dataset processing")

    a, b, c = st.columns(3)

    a.metric(
        "Images processed",
        f'{summary["images_processed"]:,}'
    )

    b.metric(
        "Images with predictions",
        f'{summary["images_with_predictions"]:,}'
    )

    c.metric(
        "Images without predictions",
        f'{summary["images_without_predictions"]:,}'
    )

    st.metric(
        "Inference time",
        f'{summary["inference_seconds"]:.2f} seconds'
    )

    st.caption(
        f'Ground-truth objects: {evaluation["ground_truth_objects"]:,} | '
        f'Predictions: {evaluation["predictions"]:,} | '
        f'IoU threshold: {evaluation["iou_threshold"]}'
    )


# ---------------------------------------------------------
# LIGHTING ANALYSIS
# ---------------------------------------------------------

with lighting:

    st.header("Lighting-wise Detection Analysis")

    lighting_df = load_csv(
        FULL / "metrics_by_lighting.csv"
    )

    st.dataframe(
        lighting_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### F1-score across lighting conditions")

    chart_df = lighting_df[
        ["lighting", "f1"]
    ].set_index("lighting")

    st.bar_chart(chart_df)

    st.markdown("### Recall across lighting conditions")

    recall_df = lighting_df[
        ["lighting", "recall"]
    ].set_index("lighting")

    st.bar_chart(recall_df)

    st.caption(
        "These are full-test lighting-wise metrics, not paired "
        "enhancement comparisons."
    )


# ---------------------------------------------------------
# CLASS ANALYSIS
# ---------------------------------------------------------

with classes:

    st.header("Class-wise Detection Analysis")

    class_df = load_csv(
        FULL / "metrics_by_class.csv"
    )

    st.dataframe(
        class_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Class-wise F1-score")

    f1_df = class_df[
        ["class_name", "f1"]
    ].set_index("class_name")

    st.bar_chart(f1_df)

    st.markdown("### Class-wise recall")

    recall_df = class_df[
        ["class_name", "recall"]
    ].set_index("class_name")

    st.bar_chart(recall_df)

    st.markdown("### Class-wise precision")

    precision_df = class_df[
        ["class_name", "precision"]
    ].set_index("class_name")

    st.bar_chart(precision_df)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "NightVision AI | Research dashboard | "
    "Results loaded from existing experiment files."
)

