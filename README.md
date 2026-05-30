# 🏥 Interpretable Medical Diagnosis using GlassBox Models

<div align="center">

<img src="assets/slide_01.jpg" width="700"/>

### *Glass-Box Models vs Black-Box Models — Accuracy & Interpretability in Breast Cancer Diagnosis*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reemzouhby/Breast_cancer_XAI/blob/main/breast_cancer_ebm__5_.ipynb)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange?logo=scikitlearn&logoColor=white)
![InterpretML](https://img.shields.io/badge/InterpretML-0.7.8-teal)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Motivation](#-motivation)
- [Dataset](#-dataset)
- [Models](#-models)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Visualizations](#-key-visualizations)
- [Discussion](#-discussion)
- [Conclusions](#-conclusions)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)

---

## 🎯 Project Overview

<div align="center">
<img src="assets/slide_02.jpg" width="700"/>
</div>

This project compares **Black-Box models** (Random Forest) with **Glass-Box models** (Explainable Boosting Machines — EBM) on the **Breast Cancer Wisconsin Diagnostic** dataset. The core question:

> **Can a fully interpretable model match or even outperform a black-box ensemble on a real medical dataset?**

---

## 💡 Motivation

<div align="center">
<img src="assets/slide_03.jpg" width="700"/>
</div>

In high-stakes domains like medicine, post-hoc explanations (SHAP, LIME) applied to black-box models carry a fundamental flaw — they **approximate** the model's behavior rather than exposing its actual logic. This distinction is critical when:

- A clinician needs to **trust and verify** a prediction
- Regulators require **auditable, legally defensible** decision logic (EU AI Act, FDA guidance)
- A false negative (missed cancer) could cost a patient their life

EBMs are **intrinsically interpretable** — their explanations are not approximations; they **are** the model.

---

## 📊 Dataset

<div align="center">
<img src="assets/slide_06.jpg" width="700"/>
</div>

The **Breast Cancer Wisconsin Diagnostic Dataset** from `sklearn.datasets`:

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 numerical |
| Malignant (0) | 212 (37.3%) |
| Benign (1) | 357 (62.7%) |
| Split | 80/20 stratified |

Features describe cell nucleus characteristics from fine needle aspirate (FNA) images: **radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension** — each as *mean*, *standard error*, and *worst*.

---

## 🤖 Models

### Model 1 — Random Forest (Black-Box 🔒)

- Ensemble of **200 decision trees** (bagging + random feature subsets)
- Predictions = majority vote across all trees
- Explanations via Gini Importance and Post-Hoc PDPs — **approximations only**

### Model 2 — Explainable Boosting Machine (Glass-Box 🏥)

<div align="center">
<img src="assets/slide_04.jpg" width="700"/>
</div>

EBM is a **Generalized Additive Model** trained with gradient boosting:

$$P(\text{malignant}) = \sigma\left(\beta_0 + f_1(x_1) + f_2(x_2) + \cdots + f_{ij}(x_i, x_j)\right)$$

Each $f_k$ is a **shape function** — the model's actual learned curve. No approximation: the model IS its explanation.

---

## 🔬 Methodology

<div align="center">
<img src="assets/slide_05.jpg" width="700"/>
</div>

| Step | Description |
|---|---|
| 1 | Load & explore — class distribution, feature histograms |
| 2 | Preprocessing — stratified split; StandardScaler for RF only |
| 3 | Train — RF on scaled data; EBM on raw data |
| 4 | Evaluate — Accuracy, AUC-ROC, 5-fold CV, confusion matrices |
| 5 | Global explanation — RF Gini vs EBM Mean Absolute Contribution |
| 6 | Shape functions — EBM native PDPs vs RF post-hoc PDPs |
| 7 | Local explanation — EBM single-patient waterfall |

### RF PDP vs EBM Shape Function

| Property | RF PDP | EBM Shape Function |
|---|---|---|
| Type | Post-hoc | Intrinsic |
| Fidelity | Approximate | Exact (= the model) |
| Feature units | Standardized (z-score) | Original units |
| Speed | Slow (recomputed) | Instant (from params) |
| Correlated features | Can mislead | Robust |
| Trust level | Medium | **High** |

---

## 📈 Results

<div align="center">
<img src="assets/slide_07.jpg" width="700"/>
</div>

| Metric | Random Forest | EBM | Winner |
|---|---|---|---|
| Test Accuracy | 95.61% | **96.49%** | 🏆 EBM |
| AUC-ROC | **99.32%** | 99.04% | RF (slightly) |
| CV Accuracy | 96.26% ± 1.8% | **96.92% ± 1.2%** | 🏆 EBM |
| Interpretable? | ❌ No | ✅ Yes (native) | 🏆 EBM |
| PDP Type | Post-hoc, approx. | Intrinsic, exact | 🏆 EBM |

> ⚠️ In medical diagnosis, **False Negatives** (predicting Benign when actually Malignant) are the most dangerous errors. EBM achieved **0 false negatives on benign patients**.

---

## 🖼️ Key Visualizations

### Feature Importance — EBM vs Random Forest

<div align="center">
<img src="assets/slide_08.jpg" width="700"/>
</div>

Both models broadly agree on the top features, but differ fundamentally:
- **RF Gini Importance** — rough rank, directionless, biased toward high-cardinality features
- **EBM Mean Absolute Contribution** — exact average impact in log-odds units, unbiased

Only **2/5 top features overlapped** between the two models, showing how RF importance can be misleading.

### EBM Shape Functions — Native Glass-Box PDPs

<div align="center">
<img src="assets/slide_09.jpg" width="700"/>
</div>

EBM's shape functions reveal the **exact nonlinear relationship** between each feature and malignancy risk — in original clinical units. These curves are not estimates; they are the model's actual learned parameters.

### Random Forest PDPs — Post-Hoc Approximation

<div align="center">
<img src="assets/slide_10.jpg" width="700"/>
</div>

RF PDPs show broadly similar shapes but use standardized z-scores (unreadable by clinicians) and are subject to extrapolation artifacts when features are correlated.

### Local Explanation — Single Patient

<div align="center">
<img src="assets/slide_11.jpg" width="700"/>
</div>

For **Patient #256** (Malignant, p=1.000), EBM shows exactly which features drove the prediction — ready for clinical records with no approximation involved.

---

## 🧠 Discussion

<div align="center">
<img src="assets/slide_12.jpg" width="700"/>
</div>

**1. The Accuracy–Interpretability Tradeoff is a Myth on structured data.**
EBM achieved higher accuracy AND full interpretability. On well-structured tabular medical data, glass-box models can match or exceed black-box ensembles.

**2. Glass-Box vs Post-Hoc: A Fundamental Difference.**
RF explanations are post-hoc approximations of an opaque model. EBM's shape functions are the model — mathematically derived, not statistically estimated. This matters when explanations must be legally defensible and clinically actionable.

**3. Medical AI Recommendation.**
For clinical decision support systems, EBMs should be the **default first choice**. They satisfy EU AI Act and FDA guidance on explainability, enable clinician trust through transparent logic, and provide patient-level explanations ready for medical records.

---

## ✅ Conclusions

<div align="center">
<img src="assets/slide_13.jpg" width="700"/>
</div>

1. **EBM ≥ RF Accuracy** — 96.49% vs 95.61%, proving interpretability does not require sacrificing predictive power.
2. **Interpretability is Real** — EBM shape functions are faithful and exact. What you see is what the model does.
3. **PDPs: Native vs Post-Hoc** — EBM's are instant, exact, and reliable. RF's are approximate, slow, and can mislead.
4. **Medical AI** — EBMs satisfy regulatory requirements and clinical trust without compromising accuracy.

---

## 🛠️ Installation & Usage

### Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn interpret
```

### Run the Notebook

```bash
jupyter notebook breast_cancer_ebm__5_.ipynb
```

Or click the **Open in Colab** badge at the top of this README.

---

## 📁 Project Structure

```
interpretable-medical-diagnosis/
│
├── breast_cancer_ebm__5_.ipynb   # Main analysis notebook
├── Ml_Master_XAI.pptx            # Presentation slides
├── README.md                     # This file
│
└── assets/                       # Slide images for README
    ├── slide_01.jpg  ...  slide_14.jpg
```

---

## 📚 References

- **InterpretML**: Nori et al. (2019). [InterpretML: A Unified Framework for Machine Learning Interpretability](https://arxiv.org/abs/1909.09223)
- **EBM (GA²M)**: Caruana et al. (2015). [Intelligible Models for HealthCare](https://dl.acm.org/doi/10.1145/2783258.2788613). KDD '15.
- **Dataset**: Wolberg, W.H. et al. UCI Machine Learning Repository — Breast Cancer Wisconsin.
- **EU AI Act**: European Commission (2024). Regulation on Artificial Intelligence.

---

<div align="center">

*Built for the ML Masters program — in medical AI, interpretability is not a luxury, it's a requirement.*

</div>
