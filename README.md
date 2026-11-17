# 🌙 NightVision AI
### Experimental Evaluation of Zero-Shot Object Detection Under Low-Light Conditions

NightVision AI is a computer vision research project that investigates how low-light conditions affect object detection performance and evaluates whether conventional image enhancement techniques can improve detection outcomes.

The project uses **YOLO26n** for object detection and the **ExDark (Exclusively Dark) dataset** for experimental evaluation. It compares original images against two image enhancement techniques: **CLAHE (Contrast Limited Adaptive Histogram Equalization)** and **Gamma Correction**.

The project also provides an interactive Streamlit application with research result visualizations and a live image detection interface.

---

## 🔍 Project Overview

Object detection systems often experience performance degradation in low-light environments because reduced illumination, poor contrast, noise, and loss of visual details make objects harder to identify.

NightVision AI explores this problem through two complementary experiments:

1. **Paired Enhancement Validation:** Evaluates how CLAHE and Gamma Correction affect detection performance compared with original images.
2. **Full Zero-Shot Evaluation:** Evaluates YOLO26n on the complete test set to establish its baseline performance under low-light conditions.

The objective is not simply to increase image brightness, but to understand how enhancement techniques influence detection accuracy, missed detections, and performance across different lighting conditions.

## 🎯 Research Objectives

- Investigate the impact of low-light conditions on object detection.
- Evaluate YOLO26n performance on the ExDark dataset.
- Compare original images with CLAHE-enhanced and Gamma-corrected images.
- Analyze changes in precision, recall, F1-score, and false negatives.
- Examine performance variations across lighting conditions and object categories.
- Investigate image-level improvements and degradation caused by enhancement.
- Present experimental findings through an interactive research dashboard.

---

## 🧠 Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Object Detection | YOLO26n |
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV |
| Image Processing | CLAHE, Gamma Correction |
| Data Processing | NumPy, Pandas |
| Visualization | Streamlit |
| Model Library | Ultralytics |
| Dataset | ExDark |
| Version Control | Git and GitHub |

---

## 📂 Dataset

This project uses the **ExDark (Exclusively Dark) dataset**, a benchmark dataset designed for research on object detection under low-light conditions.

The dataset contains images captured under different illumination conditions and includes 12 object categories.

### Object Categories

1. Bicycle
2. Boat
3. Bottle
4. Bus
5. Car
6. Cat
7. Chair
8. Cup
9. Dog
10. Motorbike
11. People
12. Table

The dataset provides bounding-box annotations and object category labels, allowing detection predictions to be compared against ground-truth annotations.

**Dataset source:** [ExDark Dataset – Exclusively Dark Image Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)

*Note: The original dataset is not included in this GitHub repository. Users must obtain it separately and follow its applicable usage terms.*

---

## ⚙️ Methodology

The project consists of two separate experimental evaluations.

### 1. Zero-Shot Object Detection

YOLO26n is used as the object detection model.

The model is evaluated on the ExDark test set without additional project-specific fine-tuning on that test set.

The evaluation compares predicted bounding boxes against ground-truth annotations using an IoU-based matching procedure.

**Evaluation metrics:**
- Precision
- Recall
- F1-score
- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)

### 2. Image Enhancement Techniques

Two conventional image enhancement methods are evaluated.

#### A. CLAHE

Contrast Limited Adaptive Histogram Equalization improves local image contrast by applying histogram equalization to smaller image regions.

Implementation configuration:

- Clip Limit: `2.0`
- Tile Grid Size: `(8, 8)`
- Color Space: YCrCb
- Enhancement applied to the luminance channel.

CLAHE attempts to improve visibility while limiting excessive contrast amplification.

#### B. Gamma Correction

Gamma correction adjusts image brightness through a nonlinear intensity transformation.

Implementation configuration:

- Gamma: `0.70`

A gamma value below 1 increases image brightness, potentially making darker regions more visible.

### 3. Paired Enhancement Comparison

The same images are evaluated under three conditions:

| Variant | Description |
|---|---|
| Original | Unmodified input image |
| CLAHE | Image enhanced using CLAHE |
| Gamma | Image enhanced using Gamma Correction |

The results are compared using detection metrics and image-level changes in missed detections.

This paired evaluation helps identify whether enhancement improves detection, has no meaningful effect, or negatively affects detection performance.

### 4. Statistical Analysis

Bootstrap confidence intervals are used to assess uncertainty in the observed differences between enhancement variants and original images.

The analysis also examines:
- Image-level improvement and degradation.
- Changes in missed detections.
- Differences in detection performance.
- Associations between image characteristics and enhancement outcomes.

---

## 📊 Experimental Results

### A. Full Zero-Shot Test Evaluation

The complete test evaluation produced the following results:

| Metric | Result |
|---|---:|
| Test Images | 2,563 |
| Ground-Truth Objects | 7,117 |
| Total Predictions | 6,279 |
| True Positives | 3,976 |
| False Positives | 2,303 |
| False Negatives | 3,141 |
| Precision | 63.32% |
| Recall | 55.87% |
| F1-score | 59.36% |
| Inference Time | 207.95 seconds |

These results provide the baseline performance of the evaluated YOLO26n configuration on the full test set.

### B. Paired Enhancement Validation

The paired enhancement experiment evaluated 1,799 images.

| Metric | Original | CLAHE | Gamma |
|---|---:|---:|---:|
| Precision | 54.93% | 54.24% | 53.57% |
| Recall | 55.29% | 55.72% | 55.45% |
| F1-score | 55.11% | 54.97% | 54.49% |

### C. Change in False Negatives

| Variant | Change in False Negatives |
|---|---:|
| CLAHE | -27 |
| Gamma | -10 |

A negative change indicates fewer missed objects compared with the original images.

### D. Image-Level Analysis

| Outcome | CLAHE | Gamma |
|---|---:|---:|
| Improved Images | 147 | 148 |
| Unchanged Images | 1,517 | 1,513 |
| Degraded Images | 135 | 138 |

The results indicate that enhancement does not uniformly improve detection performance. Some images benefit, some remain unchanged, and others experience degradation.

**Key observation:** Although both enhancement techniques reduce the total number of false negatives in this paired experiment, neither produces an overall improvement in F1-score. This highlights the importance of evaluating enhancement techniques using multiple metrics rather than brightness or missed detections alone.

*Important: The paired validation and full zero-shot test are separate experiments. Their metrics should not be directly combined or treated as measurements from the same evaluation population.*

---

## 🖥️ Interactive Streamlit Application

NightVision AI includes a web-based interface with three main sections.

### 1. Home

Provides the project introduction, research overview, objectives, and navigation to the other sections.

### 2. Research Results Dashboard

Provides interactive visualizations and experimental findings, including:

- Enhancement comparison.
- Full test evaluation.
- Lighting-condition analysis.
- Class-wise detection performance.
- Statistical analysis results.

### 3. Live Detection

Allows users to upload an image and compare object detection results under:

- Original image.
- CLAHE enhancement.
- Gamma correction.

The interface displays annotated detections and supports downloading detection results as a CSV file.

**Important:** Live Detection is an inference demonstration. It does not calculate precision, recall, or F1-score for arbitrary uploaded images because ground-truth annotations are not available for those images.

---

## 🗂️ Repository Structure

```text
NightVision-AI/
│
├── app.py
│
├── pages/
│   ├── 1_Research_Results.py
│   └── 2_Live_Detection.py
│
├── exdark_audit_output/
│   ├── paired_enhancement_validation/
│   │   ├── experiment_summary.json
│   │   └── paired_bootstrap_confidence_intervals.csv
│   │
│   └── zero_shot_full_test/
│       ├── full_test_summary.json
│       ├── full_test_evaluation_summary.json
│       ├── metrics_by_class.csv
│       └── metrics_by_lighting.csv
│
├── yolo26n.pt
├── requirements.txt
└── README.md
```

The repository contains the application, trained model weights used for inference, and selected experimental result files required by the dashboard.

The original dataset, intermediate files, and other local experimental artifacts are excluded from the public repository.

---

## 🚀 Run Locally

### Prerequisites

- Python 3.12 recommended.
- Git.
- Internet connection for installing dependencies.
- Sufficient RAM and CPU resources for model inference.

### Step 1: Clone the Repository

```bash
git clone https://github.com/mdparvezuddin/NightVision-AI.git
```

### Step 2: Navigate to the Project

```bash
cd NightVision-AI
```

### Step 3: Create a Virtual Environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## ☁️ Deployment

The application is configured for deployment using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Select the `NightVision-AI` repository.
5. Select the `main` branch.
6. Set the main file path to `app.py`.
7. Deploy the application.

Deployment resource availability may affect CPU-intensive inference operations.

---

## ⚠️ Limitations

- The project evaluates a specific YOLO26n configuration; findings should not automatically be generalized to all object detection models.
- Low-light image enhancement does not guarantee improved object detection.
- CLAHE and Gamma Correction may introduce visual changes that negatively affect some detections.
- The paired enhancement experiment and full zero-shot evaluation use separate experimental scopes.
- Live inference does not provide ground-truth-based accuracy metrics for user-uploaded images.
- CPU-based inference may be slower than GPU-based inference.
- Experimental findings are specific to the selected dataset, model configuration, and evaluation methodology.

---

## 🔬 Research Direction

The broader research direction associated with this project is:

**Degradation-Conditioned Selective Enhancement with Original-Feature Preservation for Robust Object Detection**

This research direction explores the possibility of making image enhancement conditional on image degradation characteristics rather than applying the same enhancement uniformly to every image.

The intended research ideas include:

- Identifying image degradation characteristics.
- Determining when enhancement is beneficial or harmful.
- Preserving useful information from the original image.
- Applying enhancement selectively.
- Improving robustness under varying illumination conditions.

These ideas represent the broader research direction and should be distinguished from the currently implemented CLAHE and Gamma Correction experiments.

---

## 👨‍💻 Author

**Md Parvezuddin**

B.Tech Computer Science and Engineering (AI & ML)

Research Interests:
- Computer Vision
- Artificial Intelligence
- Deep Learning
- Object Detection
- Image Processing
- Robust AI Systems

GitHub: [@mdparvezuddin](https://github.com/mdparvezuddin)

---

## 📄 License

This repository is intended for academic and research purposes.

A formal license should be added before permitting reuse, modification, or redistribution. Dataset and model usage remain subject to their respective licenses and terms.

---

## ⭐ Acknowledgements

- ExDark dataset contributors.
- Ultralytics for the YOLO object detection framework.
- PyTorch for deep learning infrastructure.
- OpenCV for image processing.
- Streamlit for the interactive application framework.

**If you find this project useful for research or learning, consider giving the repository a star!**
