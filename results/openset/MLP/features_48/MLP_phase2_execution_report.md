# Phase 2 Experiment Execution Report: Multi-Layer Perceptron (MLP) Classifier

**Date of Execution**: 2026-11-18  
**Classifier**: Multi-Layer Perceptron (`MLPModel` / `MLPClassifier`)  
**Feature Count**: 48 actual input features (NIDSPreprocessor)  
**Output Directory**: `results/openset/MLP/features_48/`  
**Execution Status**: Successfully Completed, Verified & Frozen  

---

## 1. Executive Summary

This report documents the end-to-end Phase 2 experimental evaluation of the **Multi-Layer Perceptron (MLP)** for the Explainable AI-Based Network Intrusion Detection System (XAI-NIDS). 

The evaluation includes:
1. Closed-set multiclass classification strictly on the 6 known classes using $\mathcal{D}_{\text{train\_known}}$ ($N=113,288$ flows) and validation on $\mathcal{D}_{\text{val\_known}}$ ($N=24,276$ flows).
2. Closed-set evaluation on the untouched official known test split ($N=77,831$ flows).
3. Three open-set zero-day attack detection pipelines:
   - **Maximum Softmax Probability (MSP)** baseline (Hendrycks & Gimpel, 2017).
   - **Canonical Free Energy** detector (Liu et al., NeurIPS 2020) using native pre-softmax neural network logits.
   - **Canonical OpenMax** detector (Bendale & Boult, CVPR 2016) using Class Mean Activation Vectors (MAVs) and Extreme Value Theory (EVT) Weibull tail modeling on native pre-softmax logits.
4. Independent threshold calibration ($\tau_{\text{FPR05}}$ and $\tau_{\text{Cov95}}$) exclusively on $\mathcal{D}_{\text{thresh\_cal}}$ ($N=12,139$ flows) and evaluation on the full official test set ($N=82,332$ flows: 77,831 known, 4,501 unknown).

All experimental artifacts have been persisted under `results/openset/MLP/features_48/`. No prior artifacts from LightGBM, Random Forest, XGBoost, Logistic Regression, or cross-model synthesis were modified or overwritten.

---

## 2. Model Architecture, Pre-Training Validation & Registry Integration

### 2.1 Neural Network Architecture
The MLP model is implemented in `src/models/mlp_model.py` as `MLPModel`, wrapping scikit-learn's `MLPClassifier`:
- **Input Dimension**: 48 standardized numerical and encoded categorical features.
- **Hidden Layers**: Two fully-connected dense layers: `(128, 64)`.
- **Activation Function**: Rectified Linear Unit ($\text{ReLU}(z) = \max(0, z)$).
- **Output Layer**: 6 linear neurons corresponding to the six known traffic classes:
  $$\text{logits} = h_2 W_{\text{out}} + b_{\text{out}} \in \mathbb{R}^6$$
- **Optimization**: Adam optimizer (`learning_rate_init: 0.001`, batch size: $\min(200, N)$, $\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$).
- **Early Stopping**: Enabled (`early_stopping: True`, validation fraction: 10%, `n_iter_no_change: 10`, `max_iter: 150`).
- **Class Ordering**: Strictly constrained to `[0, 1, 2, 3, 4, 5]` mapped to `["Normal", "Analysis", "Generic", "Exploits", "Fuzzers", "DoS"]`.

### 2.2 Mathematical Equivalence of Pre-Softmax Logits
For neural network energy and OpenMax detection, access to native pre-softmax activations is essential.
In `MLPModel.predict_logits(X)`, forward propagation computes:
$$h_1 = \text{ReLU}(X W_0 + b_0), \quad h_2 = \text{ReLU}(h_1 W_1 + b_1), \quad z(x) = h_2 W_{\text{out}} + b_{\text{out}}$$
We independently verified that:
$$\max_{x \in X_{\text{test}}} |\text{softmax}(z(x)) - \text{predict\_proba}(x)| = 0.00 \times 10^0$$
This confirms that `predict_logits(X)` returns exact native pre-softmax activations without probability reconstruction heuristics.

### 2.3 Pluggability Unit Tests
Before full training, `tests/test_openset_mlp_pluggability.py` was executed:
- `test_mlp_registry_instantiation`: PASSED (verified factory instantiation and alias resolution for `["mlp", "neural_net", "neural_network", "multilayer_perceptron", "MLP"]`).
- `test_mlp_fit_predict_properties`: PASSED (verified 6-class output, probability simplex $\sum p = 1.0$, pre-softmax logits, and non-zero variance).
- `test_mlp_serialization`: PASSED (verified save/load via `BaseNIDSModel.load()`).
- `test_mlp_detector_compatibility`: PASSED (verified MSP, Energy, and OpenMax detectors on MLP representations).
- `test_prior_model_reference_artifacts_untouched`: PASSED (verified non-mutation of all prior model artifacts).

---

## 3. Stage 1: Closed-Set Classification Results

### 3.1 High-Level Performance Metrics
Trained on $\mathcal{D}_{\text{train\_known}}$ ($N=113,288$ flows) and evaluated on $\mathcal{D}_{\text{test\_known}}$ ($N=77,831$ flows):

| Metric | Value | Reference Context across Models |
| :--- | :---: | :--- |
| **Test Accuracy** | **74.06%** | LGBM: 78.45%, RF: 78.48%, XGB: 78.24%, LR: 70.41% |
| **Test Macro F1** | **49.33%** | LGBM: 53.85%, RF: 53.20%, XGB: 53.29%, LR: 46.54% |
| **Test Weighted F1** | **75.37%** | LGBM: 79.35%, RF: 79.19%, XGB: 79.03%, LR: 71.67% |
| **Validation Accuracy** | **82.72%** | Closed-set validation split |
| **Validation Macro F1** | **61.88%** | Closed-set validation split |
| **Training Time** | **157.44 s** | Adam optimizer with early stopping |
| **Inference Latency** | **0.0200 ms/sample** | 2-layer forward matrix multiplication |

### 3.2 Per-Class Classification Report
From `classification_report.csv`:

| Class ID | Class Name | Precision | Recall | F1-Score | Support | Error Characteristics |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 0 | Normal | 0.9647 | 0.6791 | 0.7971 | 37,000 | 10,061 flows confused with Fuzzers; 1,183 with Exploits |
| 1 | Analysis | 0.0000 | 0.0000 | 0.0000 | 677 | Minority-class collapse (0 true positives out of 677) |
| 2 | Generic | 0.9990 | 0.9627 | 0.9805 | 18,871 | Highly separable; dominant precision and recall |
| 3 | Exploits | 0.6066 | 0.8816 | 0.7187 | 11,132 | High recall; absorbs Normal and DoS attack flows |
| 4 | Fuzzers | 0.2666 | 0.7202 | 0.3892 | 6,062 | High false alarms due to overlap with Normal traffic |
| 5 | DoS | 0.4185 | 0.0408 | 0.0744 | 4,089 | Severe under-prediction (3,431 misclassified as Exploits) |
| **Macro Avg** | — | **0.5426** | **0.5474** | **0.4933** | **77,831** | Unweighted class performance |
| **Weighted Avg** | — | **0.8303** | **0.7406** | **0.7537** | **77,831** | Dominated by Normal and Generic majority traffic |

### 3.3 6x6 Closed-Set Confusion Matrix
From `confusion_matrix_6x6.csv`:

```
True \ Pred      Normal  Analysis   Generic  Exploits   Fuzzers       DoS     Total
Normal           25,127       622         3     1,183    10,061         4    37,000
Analysis              6         0         0       400       240        31       677
Generic               6         0    18,167       498       191         9    18,871
Exploits             73        22         8     9,814     1,080       135    11,132
Fuzzers             789         0         0       854     4,366        53     6,062
DoS                  45         1         8     3,431       437       167     4,089
Total Pred       26,046       645    18,186    16,180    16,375       399    77,831
```

---

## 4. Stage 2: MSP Open-Set Evaluation

Maximum Softmax Probability (MSP) anomaly score: $S(x) = 1.0 - \max_k p_k(x)$.
Calibrated strictly on $\mathcal{D}_{\text{thresh\_cal}}$ ($N=12,139$ flows) and evaluated on official test set ($N=82,332$ flows).

- **AUROC:** **0.8189**
- **AUPR-Out:** **0.1907**
- **AUPR-In:** **0.9879**
- **OSCR AUC:** **0.6714**
- **Scoring Latency:** $0.0189\text{ ms/sample}$ | Preprocessing: $0.8648\text{ ms/sample}$ | End-to-End: $0.8837\text{ ms/sample}$

### Operating Points:
- **$\tau_{\text{FPR05}} = 0.475560$**:
  - Realized Test Normal FPR: **13.08%** ($4,840 / 37,000$ false alarms)
  - Known Coverage: **83.55%** ($65,029 / 77,831$)
  - Unknown Detection Rate: **50.79%** ($2,286 / 4,501$)
  - Open-Set Accuracy: **67.01%** | Macro F1: **43.67%** | Weighted F1: **70.52%**
- **$\tau_{\text{Cov95}} = 0.556201$**:
  - Realized Test Normal FPR: **3.82%** ($1,415 / 37,000$ false alarms)
  - Known Coverage: **94.99%** ($73,932 / 77,831$)
  - Unknown Detection Rate: **29.24%** ($1,316 / 4,501$)
  - Open-Set Accuracy: **70.06%** | Macro F1: **45.16%** | Weighted F1: **71.77%**

### Per-Category Unknown Detection:
| Category | Support | Detected ($\tau_{\text{FPR05}}$) | Recall ($\tau_{\text{FPR05}}$) | Detected ($\tau_{\text{Cov95}}$) | Recall ($\tau_{\text{Cov95}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Reconnaissance | 3,496 | 1,784 | 51.03% | 1,048 | 29.98% |
| Backdoor | 583 | 320 | 54.89% | 181 | 31.05% |
| Shellcode | 378 | 175 | 46.30% | 84 | 22.22% |
| Worms | 44 | 7 | 15.91% | 3 | 6.82% |
| **Total Unknowns** | **4,501** | **2,286** | **50.79%** | **1,316** | **29.24%** |

---

## 5. Stage 3: Energy-Based Open-Set Evaluation

Canonical Free Energy formulation (Liu et al., 2020) on native pre-softmax neural network logits:
$$E(x; T) = -z_{\max}(x) - T \cdot \log \sum_{c=0}^5 \exp\left(\frac{z_c(x) - z_{\max}(x)}{T}\right), \quad T = 1.0$$
Higher free energy indicates lower likelihood under known class training representations.

- **AUROC:** **0.8593**
- **AUPR-Out:** **0.2080**
- **AUPR-In:** **0.9909**
- **OSCR AUC:** **0.6906**
- **Scoring Latency:** $0.0190\text{ ms/sample}$ | Preprocessing: $1.0042\text{ ms/sample}$ | End-to-End: $1.0232\text{ ms/sample}$

### Operating Points:
- **$\tau_{\text{FPR05}} = -3.218229$**:
  - Realized Test Normal FPR: **15.62%** ($5,779 / 37,000$ false alarms)
  - Known Coverage: **83.81%** ($65,231 / 77,831$)
  - Unknown Detection Rate: **65.52%** ($2,949 / 4,501$)
  - Open-Set Accuracy: **68.66%** | Macro F1: **45.28%** | Weighted F1: **71.35%**
- **$\tau_{\text{Cov95}} = -2.799077$**:
  - Realized Test Normal FPR: **6.51%** ($2,410 / 37,000$ false alarms)
  - Known Coverage: **93.44%** ($72,725 / 77,831$)
  - Unknown Detection Rate: **34.84%** ($1,568 / 4,501$)
  - Open-Set Accuracy: **69.65%** | Macro F1: **45.17%** | Weighted F1: **71.34%**

### Per-Category Unknown Detection:
| Category | Support | Detected ($\tau_{\text{FPR05}}$) | Recall ($\tau_{\text{FPR05}}$) | Detected ($\tau_{\text{Cov95}}$) | Recall ($\tau_{\text{Cov95}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Reconnaissance | 3,496 | 2,284 | 65.33% | 1,216 | 34.78% |
| Backdoor | 583 | 388 | 66.55% | 223 | 38.25% |
| Shellcode | 378 | 270 | 71.43% | 126 | 33.33% |
| Worms | 44 | 7 | 15.91% | 3 | 6.82% |
| **Total Unknowns** | **4,501** | **2,949** | **65.52%** | **1,568** | **34.84%** |

---

## 6. Stage 4: OpenMax Open-Set Evaluation

Canonical OpenMax (Bendale & Boult, 2016) operating directly on the pre-softmax activation layer of the MLP neural network:
1. Mean Activation Vectors (MAVs) $\mu_c = \frac{1}{|D_{c,\text{correct}}|} \sum z(x)$ fitted on $\mathcal{D}_{\text{evt\_fit}}$ ($N=12,138$).
2. Upper-tail Euclidean distances $d(z(x), \mu_c)$ fitted with class-specific Weibull distributions ($\text{floc} = \min(\text{tail})$).
3. Top $\alpha=3$ classes recalibrated by Weibull CDF tail weights; Unknown activation $z_{\text{unknown}}$ synthesized.

### 6.1 Weibull Fitting Parameters & Tail Audit
From `mav_and_weibull_params.json`:

| Class ID | Class Name | Correct in $\mathcal{D}_{\text{evt\_fit}}$ | Tail Size ($\eta$) | Shape ($k$) | Scale ($\lambda$) | Location ($\text{loc}$) | Fallback Applied |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Normal | 3,665 | 20 | 1.1887 | 3.2973 | 35.8867 | None (`false`) |
| 1 | Analysis | 23 | 5 | 1.9967 | 7.8028 | 12.1181 | None (`false`) |
| 2 | Generic | 2,924 | 20 | 1.2415 | 1.4406 | 12.3361 | None (`false`) |
| 3 | Exploits | 2,302 | 20 | 0.7209 | 3.9068 | 13.7069 | None (`false`) |
| 4 | Fuzzers | 1,034 | 20 | 1.0318 | 2.3207 | 10.5654 | None (`false`) |
| 5 | DoS | 62 | 12 | 0.7392 | 2.3625 | 4.8271 | None (`false`) |

- **Analysis ($n_c = 23, \eta = 5$):** Out of 150 Analysis flows in $\mathcal{D}_{\text{evt\_fit}}$, MLP correctly classified 23 flows (vs. 9 for Logistic Regression). The tail size rule $\lfloor 0.2 \times 23 \rfloor = 4 < 5$ pinned $\eta = 5$. All 5 tail samples fitted cleanly without fallback.
- **Generic ($k = 1.2415, \lambda = 1.4406$):** Completely non-degenerate, smooth Weibull upper tail.
- **DoS ($n_c = 62, \eta = 12$):** Correctly scaled $\eta = \lfloor 0.2 \times 62 \rfloor = 12$.

### 6.2 Discrimination & Operating Points:
- **AUROC:** **0.8543**
- **AUPR-Out:** **0.1812**
- **AUPR-In:** **0.9906**
- **OSCR AUC:** **0.6770**
- **Scoring Latency:** $0.0145\text{ ms/sample}$ | Preprocessing: $0.9535\text{ ms/sample}$ | End-to-End: $0.9679\text{ ms/sample}$

- **$\tau_{\text{FPR05}} = 0.039363$**:
  - Realized Test Normal FPR: **15.51%** ($5,740 / 37,000$ false alarms)
  - Known Coverage: **81.91%** ($63,753 / 77,831$)
  - Unknown Detection Rate: **72.98%** ($3,285 / 4,501$)
  - Open-Set Accuracy: **67.65%** | Macro F1: **44.08%** | Weighted F1: **70.59%**
- **$\tau_{\text{Cov95}} = 0.078016$**:
  - Realized Test Normal FPR: **5.46%** ($2,020 / 37,000$ false alarms)
  - Known Coverage: **92.99%** ($72,377 / 77,831$)
  - Unknown Detection Rate: **30.86%** ($1,389 / 4,501$)
  - Open-Set Accuracy: **68.39%** | Macro F1: **43.80%** | Weighted F1: **70.43%**

### Per-Category Unknown Detection:
| Category | Support | Detected ($\tau_{\text{FPR05}}$) | Recall ($\tau_{\text{FPR05}}$) | Detected ($\tau_{\text{Cov95}}$) | Recall ($\tau_{\text{Cov95}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Reconnaissance | 3,496 | 2,571 | 73.54% | 1,139 | 32.58% |
| Backdoor | 583 | 406 | 69.64% | 147 | 25.21% |
| Shellcode | 378 | 294 | 77.78% | 95 | 25.13% |
| Worms | 44 | 14 | 31.82% | 8 | 18.18% |
| **Total Unknowns** | **4,501** | **3,285** | **72.98%** | **1,389** | **30.86%** |

---

## 7. Within-Model Methodological Synthesis (MLP)

| Metric | MSP | Free Energy ($T=1.0$) | OpenMax ($\alpha=3$) | Best Performing Detector |
| :--- | :---: | :---: | :---: | :--- |
| **Score Representation** | Normalized Softmax | Native Pre-Softmax Logits | Logit MAV Upper-Tail EVT | — |
| **AUROC** | 0.8189 | **0.8593** | 0.8543 | **Energy (+4.04% over MSP)** |
| **AUPR-Out** | 0.1907 | **0.2080** | 0.1812 | **Energy (+1.73% over MSP)** |
| **AUPR-In** | 0.9879 | **0.9909** | 0.9906 | **Energy (+0.30% over MSP)** |
| **OSCR AUC** | 0.6714 | **0.6906** | 0.6770 | **Energy (+1.92% over MSP)** |
| **Scoring Latency** | 0.0189 ms | 0.0190 ms | **0.0145 ms** | **OpenMax (Vectorized Matrix)** |
| **$\tau_{\text{FPR05}}$ Threshold** | 0.475560 | -3.218229 | 0.039363 | Calibrated on $\mathcal{D}_{\text{thresh\_cal}}$ |
| **$\tau_{\text{FPR05}}$ Normal FPR** | **13.08%** | 15.62% | 15.51% | **MSP (Lowest False Alarms)** |
| **$\tau_{\text{FPR05}}$ Known Coverage** | 83.55% | **83.81%** | 81.91% | **Energy** |
| **$\tau_{\text{FPR05}}$ Unknown Recall** | 50.79% | 65.52% | **72.98%** | **OpenMax (+22.19% over MSP)** |
| **$\tau_{\text{FPR05}}$ Open-Set Acc** | 67.01% | **68.66%** | 67.65% | **Energy** |
| **$\tau_{\text{FPR05}}$ Macro F1** | 43.67% | **45.28%** | 44.08% | **Energy** |
| **$\tau_{\text{Cov95}}$ Threshold** | 0.556201 | -2.799077 | 0.078016 | Calibrated on $\mathcal{D}_{\text{thresh\_cal}}$ |
| **$\tau_{\text{Cov95}}$ Normal FPR** | **3.82%** | 6.51% | 5.46% | **MSP (Lowest False Alarms)** |
| **$\tau_{\text{Cov95}}$ Known Coverage** | **94.99%** | 93.44% | 92.99% | **MSP ($\approx 95.0\%$ target)** |
| **$\tau_{\text{Cov95}}$ Unknown Recall** | 29.24% | **34.84%** | 30.86% | **Energy** |
| **$\tau_{\text{Cov95}}$ Open-Set Acc** | **70.06%** | 69.65% | 68.39% | **MSP** |
| **$\tau_{\text{Cov95}}$ Macro F1** | 45.16% | **45.17%** | 43.80% | **Energy** |

### Key Methodological Insights for MLP:
1. **Canonical Formulation Advantage:** Unlike tree ensembles where Energy and OpenMax rely on heuristic margins or tree leaf votes, MLP operates on genuine continuous logit activations ($z(x) \in \mathbb{R}^6$). This yields excellent OOD discrimination: **Energy AUROC = 0.8593** and **OpenMax AUROC = 0.8543**.
2. **OpenMax Unknown Recall Dominance at $\tau_{\text{FPR05}}$:** OpenMax detects **72.98%** of zero-day attacks (including $73.54\%$ of Reconnaissance and $77.78\%$ of Shellcode), significantly surpassing MSP ($50.79\%$).
3. **Energy Stability:** Free Energy achieves the highest AUROC ($0.8593$), highest AUPR-Out ($0.2080$), and highest OSCR AUC ($0.6906$) while maintaining stable, continuous log-sum-exp behavior.

---

## 8. Complete Artifact Inventory

All artifacts are persisted under `results/openset/MLP/features_48/`:

### Closed-Set Artifacts:
- Model Wrapper: `model.pkl`
- Preprocessor: `preprocessor.pkl`
- Configuration: `run_config.json`
- Feature Names: `feature_names.json`
- Class Mapping: `class_mapping.json`
- Metrics JSON: `metrics_closed_set.json`
- Tabular Report: `classification_report.csv`
- Confusion Matrix: `confusion_matrix_6x6.csv`
- Confusion Matrix Heatmap: `confusion_matrix_6x6.png`
- ROC Curves: `roc_curves_closed_set.png`
- PR Curves: `pr_curves_closed_set.png`
- Feature Importance: `feature_importance.png`

### MSP Open-Set Artifacts:
- Thresholds: `MSP/thresholds.json`
- Experiment Config: `MSP/experiment_config.json`
- Metrics JSON: `MSP/metrics_openset_msp.json`
- Metrics CSV: `MSP/metrics_openset_msp.csv`
- 7x7 CM $\tau_{\text{FPR05}}$: `MSP/confusion_matrix_7x7_fpr05.csv` & `.png`
- 7x7 CM $\tau_{\text{Cov95}}$: `MSP/confusion_matrix_7x7_cov95.csv` & `.png`
- Per-Category CSV: `MSP/per_category_unknown_detection.csv`
- Curves: `MSP/roc_pr_curves.png`, `MSP/oscr_curve.png`

### Energy Open-Set Artifacts:
- Thresholds: `Energy/thresholds.json`
- Experiment Config: `Energy/experiment_config.json`
- Metrics JSON: `Energy/metrics_openset_energy.json`
- Metrics CSV: `Energy/metrics_openset_energy.csv`
- 7x7 CM $\tau_{\text{FPR05}}$: `Energy/confusion_matrix_7x7_fpr05.csv` & `.png`
- 7x7 CM $\tau_{\text{Cov95}}$: `Energy/confusion_matrix_7x7_cov95.csv` & `.png`
- Per-Category CSV: `Energy/per_category_unknown_detection.csv`
- Curves: `Energy/roc_pr_curves.png`, `Energy/oscr_curve.png`

### OpenMax Open-Set Artifacts:
- Thresholds: `OpenMax/thresholds.json`
- Experiment Config: `OpenMax/experiment_config.json`
- Parameters: `OpenMax/mav_and_weibull_params.json`
- Metrics JSON: `OpenMax/metrics_openset_openmax.json`
- Metrics CSV: `OpenMax/metrics_openset_openmax.csv`
- 7x7 CM $\tau_{\text{FPR05}}$: `OpenMax/confusion_matrix_7x7_fpr05.csv` & `.png`
- 7x7 CM $\tau_{\text{Cov95}}$: `OpenMax/confusion_matrix_7x7_cov95.csv` & `.png`
- Per-Category CSV: `OpenMax/per_category_unknown_detection.csv`
- Curves: `OpenMax/roc_pr_curves.png`, `OpenMax/oscr_curve.png`

---

## 9. Non-Mutation & Data Integrity Verification

- **Prior Artifacts:** Confirmed zero modifications to LightGBM, Random Forest, XGBoost, Logistic Regression, and both cross-model synthesis directories (`results/openset/synthesis/`, `results/openset/synthesis_3models/`).
- **Data Partitions:** Verified identical split sizes ($N=113,288; 24,276; 12,138; 12,139; 77,831; 4,501$) and zero unknown attack contamination in training, validation, or calibration.
- **Stopping Condition:** All MLP experimental phases and artifacts are complete.
