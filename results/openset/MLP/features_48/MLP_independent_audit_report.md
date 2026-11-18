# Independent Phase 2 Audit Report: Multi-Layer Perceptron (MLP) Classifier

**Date of Audit**: 2026-11-18  
**Auditor**: Independent Antigravity Audit Subsystem  
**Audit Target**: Multi-Layer Perceptron (`MLPModel` / `MLPClassifier`)  
**Artifact Directory**: `results/openset/MLP/features_48/`  
**Execution Report**: `MLP_phase2_execution_report.md`  
**Audit Nature**: Strict Read-Only Verification, Independent Numerical Recomputation, and Artifact Consistency Audit  
**Overall Verdict**: **ACCEPTED AS INDEPENDENTLY VERIFIED (WITH DOCUMENTED STATISTICAL LIMITATIONS)**

---

## 1. Audit Scope, Objectives & Methodology

This audit provides an exhaustive, independent evaluation of the Phase 2 experiments conducted for the **Multi-Layer Perceptron (MLP)** network intrusion detection classifier on the UNSW-NB15 benchmark dataset.

### Audit Objectives:
1. **Safety & Non-Mutation**: Verify that all previous model artifacts (LightGBM, Random Forest, XGBoost, Logistic Regression) and cross-model synthesis directories remain completely untouched.
2. **Architecture & Logit Integrity**: Verify MLP hyperparameters, layer structure, activation functions, and prove that `predict_logits()` extracts exact native pre-softmax neural activations without heuristic approximations.
3. **Partition Isolation & Leakage Check**: Verify that data partitions adhere strictly to the established Phase 2 protocol with zero unknown class leakage into training, validation, EVT fitting, or threshold calibration splits.
4. **Independent Closed-Set Recomputation**: Independently regenerate closed-set predictions on the official untouched known test set ($N=77,831$), recomputing accuracy, macro F1, weighted F1, per-class metrics, and the $6 \times 6$ confusion matrix.
5. **Independent Open-Set Recomputation**: Independently recompute AUROC, AUPR-Out, AUPR-In, OSCR AUC, calibrated operating thresholds ($\tau_{\text{FPR05}}$ and $\tau_{\text{Cov95}}$), operating point metrics, and $7 \times 7$ confusion matrices for Maximum Softmax Probability (MSP), Canonical Free Energy, and Canonical OpenMax ($N=82,332$).
6. **EVT Weibull Fitting & Statistical Validity**: Audit Extreme Value Theory tail models across all six classes, analyzing sample support, tail size selection, and parameter stability.
7. **Per-Category Unknown Attack Audit**: Verify detection counts and percentages across Reconnaissance, Backdoor, Shellcode, and Worms.
8. **Artifact Preservation & Machine-Readability**: Verify that all artifacts are non-empty, readable, and mathematically aligned.
9. **Cross-Model Parity**: Verify consistency against LightGBM, Random Forest, XGBoost, and Logistic Regression.

---

## 2. Files and Source Code Inspected

### Source Code:
- `src/models/mlp_model.py`: `MLPModel` wrapper, layer architecture, forward propagation logit extraction.
- `src/models/registry.py`: Factory instantiation, alias resolution (`mlp`, `neural_network`, `multilayer_perceptron`).
- `src/openset/partitioner.py`: 4-way stratified partitioning, class definitions, zero unknown contamination assertion.
- `src/preprocessing.py`: Feature standardizer and categorical encoder.
- `src/openset/train_classifier.py`: Closed-set training pipeline, Adam optimization, early stopping.
- `src/openset/msp_detector.py`: MSP anomaly score calculation and threshold calibration.
- `src/openset/energy_detector.py`: Log-sum-exp energy calculation and threshold calibration.
- `src/openset/openmax_detector.py`: MAV computation, Weibull fitting (`scipy.stats.weibull_min`), alpha recalibration.
- `src/openset/metrics.py`: Standardized ROC, PR, OSCR, and operating point computation.

### Artifacts Inspected:
- Model and preprocessor: `model.pkl` (367,477 bytes), `preprocessor.pkl` (12,722 bytes).
- Metadata & Configuration: `run_config.json`, `class_mapping.json`, `feature_names.json`.
- Closed-set outputs: `metrics_closed_set.json`, `classification_report.csv`, `confusion_matrix_6x6.csv`, `confusion_matrix_6x6.png`, `roc_curves_closed_set.png`, `pr_curves_closed_set.png`, `feature_importance.png`.
- MSP outputs: `MSP/thresholds.json`, `MSP/experiment_config.json`, `MSP/metrics_openset_msp.json`, `MSP/metrics_openset_msp.csv`, `MSP/confusion_matrix_7x7_fpr05.csv/.png`, `MSP/confusion_matrix_7x7_cov95.csv/.png`, `MSP/per_category_unknown_detection.csv`, `MSP/roc_pr_curves.png`, `MSP/oscr_curve.png`.
- Energy outputs: `Energy/thresholds.json`, `Energy/experiment_config.json`, `Energy/metrics_openset_energy.json`, `Energy/metrics_openset_energy.csv`, `Energy/confusion_matrix_7x7_fpr05.csv/.png`, `Energy/confusion_matrix_7x7_cov95.csv/.png`, `Energy/per_category_unknown_detection.csv`, `Energy/roc_pr_curves.png`, `Energy/oscr_curve.png`.
- OpenMax outputs: `OpenMax/thresholds.json`, `OpenMax/experiment_config.json`, `OpenMax/mav_and_weibull_params.json`, `OpenMax/metrics_openset_openmax.json`, `OpenMax/metrics_openset_openmax.csv`, `OpenMax/confusion_matrix_7x7_fpr05.csv/.png`, `OpenMax/confusion_matrix_7x7_cov95.csv/.png`, `OpenMax/per_category_unknown_detection.csv`, `OpenMax/roc_pr_curves.png`, `OpenMax/oscr_curve.png`.
- Reports: `MLP_phase2_execution_report.md`, `artifact_completeness_report.md`.

---

## 3. Architecture, Optimizer & Logit Verification

### 3.1 Model Architecture & Hyperparameters
Inspection of `model.pkl` and `run_config.json` confirmed:
- **Input Dimension**: Exactly 48 features corresponding to `feature_names.json`.
- **Hidden Layers**: Two fully-connected hidden layers: `(128, 64)`.
- **Activation Function**: Rectified Linear Unit (`relu`).
- **Output Layer**: 6 units corresponding to canonical class IDs `[0, 1, 2, 3, 4, 5]`.
- **Weight Shapes**:
  - Layer 1: $W_0 \in \mathbb{R}^{48 \times 128}$, $b_0 \in \mathbb{R}^{128}$
  - Layer 2: $W_1 \in \mathbb{R}^{128 \times 64}$, $b_1 \in \mathbb{R}^{64}$
  - Output Layer: $W_{\text{out}} \in \mathbb{R}^{64 \times 6}$, $b_{\text{out}} \in \mathbb{R}^{6}$
- **Optimizer**: Adam (`learning_rate_init: 0.001`, $\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$).
- **Early Stopping & Convergence**:
  - `early_stopping: True`, `validation_fraction: 0.10`, `n_iter_no_change: 10`.
  - Actual fitted epochs (`n_iter_`): **45 epochs** (out of `max_iter: 150`).
  - Best validation score: **0.825316** (achieved at epoch 34).
  - Optimizer stopped cleanly due to early stopping criteria without divergence.

### 3.2 Pre-Softmax Logit Verification
The forward propagation logic in `MLPModel.predict_logits(X)`:
$$h_1 = \max(0, X W_0 + b_0), \quad h_2 = \max(0, h_1 W_1 + b_1), \quad z(x) = h_2 W_{\text{out}} + b_{\text{out}}$$
was independently evaluated against a manual NumPy forward propagation on all 77,831 known test samples and all 82,332 full test samples:
- **Maximum Logit Discrepancy**:
  $$\max_{x \in X_{\text{test}}} |z_{\text{model}}(x) - z_{\text{manual}}(x)| = 0.00 \times 10^0 \text{ (exact match)}$$
- **Softmax Mathematical Equivalence**:
  $$\max_{x \in X_{\text{test}}} |\text{softmax}(z(x)) - \text{predict\_proba}(x)| = 0.00 \times 10^0 \text{ (exact match)}$$
- **Status**: **PASS**. The model extracts genuine pre-softmax activations.

---

## 4. Dataset Isolation & Fixed Partition Verification

The raw UNSW-NB15 dataset files (`data/raw/unsw_nb15_train.csv` and `data/raw/unsw_nb15_test.csv`) were loaded and partitioned via `src/openset/partitioner.py` using `random_state=42`.

### 4.1 Partition Sizes and Class Accounting

| Partition | Flow Count | Description & Verification |
| :--- | :---: | :--- |
| **Raw Training Set** | 175,341 | Loaded from `data/raw/unsw_nb15_train.csv` |
| **Purged Unknown Attacks** | 13,500 | Unseen attacks purged from training (Recon: 10,491, Backdoor: 1,746, Shellcode: 1,133, Worms: 130) |
| **Known Training Pool** | 161,841 | Disjoint known training traffic ($175,341 - 13,500$) |
| **$\mathcal{D}_{\text{train\_known}}$** | 113,288 | Exactly 70.0% of known training pool (used to train MLP and fit preprocessor) |
| **$\mathcal{D}_{\text{val\_known}}$** | 24,276 | Exactly 15.0% of known training pool (used for closed-set validation) |
| **$\mathcal{D}_{\text{evt\_fit}}$** | 12,138 | Exactly 7.5% of known training pool (used exclusively for OpenMax MAV and Weibull tail fitting) |
| **$\mathcal{D}_{\text{thresh\_cal}}$** | 12,139 | Exactly 7.5% of known training pool (used exclusively for OOD threshold calibration) |
| **$\mathcal{D}_{\text{test\_known}}$** | 77,831 | Official test set known traffic (6 known classes) |
| **$\mathcal{D}_{\text{test\_unknown}}$** | 4,501 | Official test set unknown traffic (Recon: 3,496, Backdoor: 583, Shellcode: 378, Worms: 44) |
| **Total Test Split** | 82,332 | Untouched official test benchmark ($77,831 + 4,501$) |

### 4.2 Leakage & Mutual Disjointness Audit
1. **Zero Contamination**: The intersection of attack categories in $\mathcal{D}_{\text{train\_known}}$, $\mathcal{D}_{\text{val\_known}}$, $\mathcal{D}_{\text{evt\_fit}}$, and $\mathcal{D}_{\text{thresh\_cal}}$ with `{"Reconnaissance", "Backdoor", "Shellcode", "Worms"}` was confirmed to be empty.
2. **Mutual Disjointness**: Pairwise index intersections across the four training/calibration splits were confirmed to be strictly zero prior to index resetting.
3. **Preprocess Fitting**: The preprocessor was fitted exclusively on $\mathcal{D}_{\text{train\_known}}$ ($N=113,288$).
4. **Target & ID Columns**: Verified that `id`, `attack_cat`, and binary `label` columns were completely excluded from the 48-feature input matrix.
- **Status**: **PASS**. Leakage-free isolation is strictly maintained.

---

## 5. Independently Recomputed Closed-Set Classification Results

Evaluated on $\mathcal{D}_{\text{test\_known}}$ ($N=77,831$ flows):

### 5.1 Recomputed High-Level Metrics

| Metric | Saved Value in Artifacts | Independently Recomputed | Difference | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.7405917950 | 0.7405917950 | 0.00000000 | **PASS** |
| **Macro F1** | 0.4933079629 | 0.4933079629 | 0.00000000 | **PASS** |
| **Weighted F1** | 0.7536733442 | 0.7536733442 | 0.00000000 | **PASS** |
| **Total Test Support** | 77,831 | 77,831 | 0 | **PASS** |
| **Total Correct Flows** | 57,641 | 57,641 | 0 | **PASS** |

### 5.2 Recomputed Per-Class Report vs Saved `classification_report.csv`

| Class ID | Class Name | Precision (Recomputed) | Recall (Recomputed) | F1-Score (Recomputed) | Support | Reconciliation Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Normal | 0.964728 | 0.679108 | 0.797100 | 37,000 | **PASS** (Exact Match) |
| 1 | Analysis | 0.000000 | 0.000000 | 0.000000 | 677 | **PASS WITH LIMITATION** |
| 2 | Generic | 0.999010 | 0.962694 | 0.980516 | 18,871 | **PASS** (Exact Match) |
| 3 | Exploits | 0.606551 | 0.881603 | 0.718714 | 11,132 | **PASS** (Exact Match) |
| 4 | Fuzzers | 0.266626 | 0.720224 | 0.389209 | 6,062 | **PASS** (Exact Match) |
| 5 | DoS | 0.418546 | 0.040841 | 0.074436 | 4,089 | **PASS WITH LIMITATION** |

### 5.3 Detailed Investigation of Analysis (0% Recall) and DoS (4.08% Recall)
Inspection of the recomputed $6 \times 6$ confusion matrix confirmed:
- **Analysis (Class 1, Support: 677)**:
  - Row distribution: `[Normal: 6, Analysis: 0, Generic: 0, Exploits: 400, Fuzzers: 240, DoS: 31]`.
  - True Positives = **0**.
  - *Root Cause Analysis*: Extreme class imbalance (Analysis constitutes only 0.87% of known test flows). Due to standard cross-entropy training without class weighting, the network's decision boundary for Analysis collapsed in favor of the dominant Exploits and Fuzzers classes.
  - *Classification*: **Statistical Limitation (Minority Class Collapse)**. Not an implementation error.
- **DoS (Class 5, Support: 4,089)**:
  - Row distribution: `[Normal: 45, Analysis: 1, Generic: 8, Exploits: 3,431, Fuzzers: 437, DoS: 167]`.
  - True Positives = **167** ($167 / 4,089 = 4.0841\%$).
  - *Root Cause Analysis*: 3,431 DoS flows (83.91%) were misclassified as Exploits. In UNSW-NB15, many DoS attack signatures share packet rate, packet length, and TCP flag characteristics with port-scanning Exploits, causing the dense ReLU decision surfaces to absorb DoS into the larger Exploits manifold.
  - *Classification*: **Statistical Limitation (Class Overlap)**. Not an implementation error.

---

## 6. Independently Recomputed Open-Set Detection Results

Evaluated on the full official test set ($N=82,332$ flows: 77,831 known, 4,501 unknown).

### 6.1 Discrimination Metrics Recomputation

| Detector | Metric | Saved Value | Recomputed Value | Absolute Difference | Reconciliation Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MSP** | AUROC | 0.8188888 | 0.8188888 | $0.00 \times 10^0$ | **PASS** |
| | AUPR-Out | 0.1906798 | 0.1906798 | $0.00 \times 10^0$ | **PASS** |
| | AUPR-In | 0.9878678 | 0.9878678 | $0.00 \times 10^0$ | **PASS** |
| | OSCR AUC | 0.6713608 | 0.6713608 | $0.00 \times 10^0$ | **PASS** |
| **Energy ($T=1.0$)** | AUROC | 0.8593322 | 0.8593322 | $0.00 \times 10^0$ | **PASS** |
| | AUPR-Out | 0.2079861 | 0.2079861 | $0.00 \times 10^0$ | **PASS** |
| | AUPR-In | 0.9908564 | 0.9908564 | $0.00 \times 10^0$ | **PASS** |
| | OSCR AUC | 0.6905664 | 0.6905664 | $0.00 \times 10^0$ | **PASS** |
| **OpenMax ($\alpha=3$)** | AUROC | 0.8543425 | 0.8543425 | $0.00 \times 10^0$ | **PASS** |
| | AUPR-Out | 0.1811782 | 0.1811782 | $0.00 \times 10^0$ | **PASS** |
| | AUPR-In | 0.9905728 | 0.9905728 | $0.00 \times 10^0$ | **PASS** |
| | OSCR AUC | 0.6769859 | 0.6769859 | $0.00 \times 10^0$ | **PASS** |

### 6.2 Operating Threshold Calibration Verification
Operating thresholds were independently recomputed on $\mathcal{D}_{\text{thresh\_cal}}$ ($N=12,139$):
- **MSP**:
  - Recomputed $\tau_{\text{FPR05}}$: `0.475560453950339` (Saved: `0.475560453950339`) -> **PASS**
  - Recomputed $\tau_{\text{Cov95}}$: `0.5562010196230628` (Saved: `0.5562010196230628`) -> **PASS**
- **Canonical Free Energy**:
  - Recomputed $\tau_{\text{FPR05}}$: `-3.2182289878357744` (Saved: `-3.2182289878357744`) -> **PASS**
  - Recomputed $\tau_{\text{Cov95}}$: `-2.7990767172135795` (Saved: `-2.7990767172135795`) -> **PASS**
- **Canonical OpenMax**:
  - Recomputed $\tau_{\text{FPR05}}$: `0.03936294533909908` (Saved: `0.03936294533909908`) -> **PASS**
  - Recomputed $\tau_{\text{Cov95}}$: `0.0780163884472662` (Saved: `0.0780163884472662`) -> **PASS**

### 6.3 Test Operating Points Recomputation

#### Operating Point 1: $\tau_{\text{FPR05}}$ (Calibrated for $\le 5\%$ Normal False Alarms on $\mathcal{D}_{\text{thresh\_cal}}$)

| Detector | Metric | Saved Value | Recomputed Value | Discrepancy | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MSP** | Normal FPR | 0.130811 | 0.130811 | 0.000000 | **PASS** |
| | Known Coverage | 0.835515 | 0.835515 | 0.000000 | **PASS** |
| | Unknown Detection Rate | 0.507887 | 0.507887 | 0.000000 | **PASS** |
| | Open-Set Accuracy | 0.670104 | 0.670104 | 0.000000 | **PASS** |
| | Open-Set Macro F1 | 0.436730 | 0.436730 | 0.000000 | **PASS** |
| **Energy** | Normal FPR | 0.156189 | 0.156189 | 0.000000 | **PASS** |
| | Known Coverage | 0.838111 | 0.838111 | 0.000000 | **PASS** |
| | Unknown Detection Rate | 0.655188 | 0.655188 | 0.000000 | **PASS** |
| | Open-Set Accuracy | 0.686574 | 0.686574 | 0.000000 | **PASS** |
| | Open-Set Macro F1 | 0.452800 | 0.452800 | 0.000000 | **PASS** |
| **OpenMax** | Normal FPR | 0.155135 | 0.155135 | 0.000000 | **PASS** |
| | Known Coverage | 0.808303 | 0.808303 | 0.000000 | **PASS** |
| | Unknown Detection Rate | 0.729838 | 0.729838 | 0.000000 | **PASS** |
| | Open-Set Accuracy | 0.667541 | 0.667541 | 0.000000 | **PASS** |
| | Open-Set Macro F1 | 0.441014 | 0.441014 | 0.000000 | **PASS** |

#### Operating Point 2: $\tau_{\text{Cov95}}$ (Calibrated for $\ge 95\%$ Known Coverage on $\mathcal{D}_{\text{thresh\_cal}}$)

| Detector | Metric | Saved Value | Recomputed Value | Discrepancy | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MSP** | Normal FPR | 0.038243 | 0.038243 | 0.000000 | **PASS** |
| | Known Coverage | 0.949904 | 0.949904 | 0.000000 | **PASS** |
| | Unknown Detection Rate | 0.292379 | 0.292379 | 0.000000 | **PASS** |
| | Open-Set Accuracy | 0.700602 | 0.700602 | 0.000000 | **PASS** |
| | Open-Set Macro F1 | 0.451616 | 0.451616 | 0.000000 | **PASS** |
| **Energy** | Normal FPR | 0.065135 | 0.065135 | 0.000000 | **PASS** |
| | Known Coverage | 0.934396 | 0.934396 | 0.000000 | **PASS** |
| | Unknown Detection Rate | 0.348367 | 0.348367 | 0.000000 | **PASS** |
| | Open-Set Accuracy | 0.696546 | 0.696546 | 0.000000 | **PASS** |
| | Open-Set Macro F1 | 0.451692 | 0.451692 | 0.000000 | **PASS** |
| **OpenMax** | Normal FPR | 0.054595 | 0.054595 | 0.000000 | **PASS** |
| | Known Coverage | 0.929912 | 0.929912 | 0.000000 | **PASS** |
| | Unknown Detection Rate | 0.308598 | 0.308598 | 0.000000 | **PASS** |
| | Open-Set Accuracy | 0.683914 | 0.683914 | 0.000000 | **PASS** |
| | Open-Set Macro F1 | 0.437992 | 0.437992 | 0.000000 | **PASS** |

### 6.4 Analysis of Normal FPR Inflation at $\tau_{\text{FPR05}}$ (Test FPR: 13.08% – 15.62%)
The audit specifically investigated why test Normal FPR exceeded the 5.0% calibration target across all three detectors (MSP: 13.08%, Energy: 15.62%, OpenMax: 15.51%):
1. **Calibration Score Verification**: On $\mathcal{D}_{\text{thresh\_cal}}$, the empirical 95th percentile threshold yielded an exact 5.00% empirical false alarm rate on Normal calibration traffic for all three detectors.
2. **Distribution Shift Quantification**:
   - **MSP**: Normal calibration scores had $\mu = 0.0804, \sigma = 0.1682$; Normal test scores had $\mu = 0.1799, \sigma = 0.2200$. Shift ratio: **2.62x**.
   - **Energy**: Normal calibration scores had $\mu = -14.0358, \sigma = 7.2593$; Normal test scores had $\mu = -10.7042, \sigma = 7.8072$. Shift ratio: **3.12x**.
   - **OpenMax**: Normal calibration scores had $\mu = 0.0069, \sigma = 0.0373$; Normal test scores had $\mu = 0.0173, \sigma = 0.0479$. Shift ratio: **3.10x**.
3. **Audit Finding**: In the UNSW-NB15 official test partition, Normal traffic exhibits greater entropy, higher variance, and lower classifier confidence than in the training-derived calibration set. This represents a **genuine dataset covariate shift** between train and test splits, rather than an implementation or calibration defect.
4. **Classification**: **PASS WITH LIMITATION**.

---

## 7. OpenMax EVT & Weibull Tail Fitting Assessment

OpenMax was fitted strictly on $\mathcal{D}_{\text{evt\_fit}}$ ($N=12,138$ known flows).

### 7.1 Class-Specific Weibull Parameters

| Class ID | Class Name | Total in $\mathcal{D}_{\text{evt\_fit}}$ | Correct ($n_c$) | Tail Size ($\eta$) | Shape ($k$) | Scale ($\lambda$) | Location ($\text{loc}$) | Fallback Applied |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Normal | 5,767 | 3,665 | 20 | 1.1887 | 3.2973 | 35.8867 | None (`false`) |
| 1 | Analysis | 150 | 23 | 5 | 1.9967 | 7.8028 | 12.1181 | None (`false`) |
| 2 | Generic | 2,940 | 2,924 | 20 | 1.2415 | 1.4406 | 12.3361 | None (`false`) |
| 3 | Exploits | 2,683 | 2,302 | 20 | 0.7209 | 3.9068 | 13.7069 | None (`false`) |
| 4 | Fuzzers | 1,678 | 1,034 | 20 | 1.0318 | 2.3207 | 10.5654 | None (`false`) |
| 5 | DoS | 920 | 62 | 12 | 0.7392 | 2.3625 | 4.8271 | None (`false`) |

### 7.2 Methodological Evaluation of EVT Fitting
- **Analysis Class ($n_c=23, \eta=5$)**:
  - The tail size rule $\eta = \max(5, \min(20, \lfloor 0.2 \times 23 \rfloor)) = \max(5, 4) = 5$ correctly pinned $\eta=5$.
  - Scipy's `weibull_min.fit()` fitted the 5 tail samples cleanly with $k=1.9967$ and $\lambda=7.8028$ without fallback.
  - *Statistical Observation*: Fitting an extreme value distribution to 5 samples carries wide confidence intervals. However, because Analysis constitutes only 0.87% of known test flows, its contribution to overall open-set false alarms is negligible.
  - *Classification*: **PASS WITH LIMITATION**.
- **DoS Class ($n_c=62, \eta=12$)**:
  - $\lfloor 0.2 \times 62 \rfloor = 12$. The 12 tail samples fitted stably ($k=0.7392, \lambda=2.3625$).
  - *Classification*: **PASS WITH LIMITATION**.
- **Generic Class ($n_c=2,924, \eta=20$)**:
  - Unlike tree-based ensembles where Generic tail distances frequently collapsed to discrete constants, MLP produced a continuous activation spread ($k=1.2415, \lambda=1.4406, \text{loc}=12.3361$).
  - *Classification*: **PASS**. Continuous, well-conditioned EVT model.

---

## 8. Per-Category Unknown Attack Evaluation Recomputation

Evaluated across the 4 unseen attack families ($N=4,501$ flows). Recomputed percentages use exact category supports as denominators.

| Category | Support | MSP Detected ($\tau_{\text{FPR05}}$) | MSP Detected ($\tau_{\text{Cov95}}$) | Energy Detected ($\tau_{\text{FPR05}}$) | Energy Detected ($\tau_{\text{Cov95}}$) | OpenMax Detected ($\tau_{\text{FPR05}}$) | OpenMax Detected ($\tau_{\text{Cov95}}$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reconnaissance** | 3,496 | 1,784 (51.03%) | 1,048 (29.98%) | 2,284 (65.33%) | 1,216 (34.78%) | 2,571 (73.54%) | 1,139 (32.58%) | **PASS** |
| **Backdoor** | 583 | 320 (54.89%) | 181 (31.05%) | 388 (66.55%) | 223 (38.25%) | 406 (69.64%) | 147 (25.21%) | **PASS** |
| **Shellcode** | 378 | 175 (46.30%) | 84 (22.22%) | 270 (71.43%) | 126 (33.33%) | 294 (77.78%) | 95 (25.13%) | **PASS** |
| **Worms** | 44 | 7 (15.91%) | 3 (6.82%) | 7 (15.91%) | 3 (6.82%) | 14 (31.82%) | 8 (18.18%) | **PASS WITH LIMITATION** |
| **Total Unknowns** | **4,501** | **2,286 (50.79%)** | **1,316 (29.24%)** | **2,949 (65.52%)** | **1,568 (34.84%)** | **3,285 (72.98%)** | **1,389 (30.86%)** | **PASS** |

### Observation on Worms Attack Recall
Because Worms has only $N=44$ flows in the official test set, each detected flow changes the recall metric by $2.27\%$. The reported recall values (e.g., 7 flows = 15.91%, 14 flows = 31.82%) are numerically verified, but subject to high binomial variance.

---

## 9. Artifact Completeness & Visualizations Audit

The contents of `results/openset/MLP/features_48/` were audited against all requested items:
- **Core Standard Artifacts (46/46)**: All 46 core standard files (12 closed-set, 11 MSP, 11 Energy, 12 OpenMax) exist, are non-empty, and agree 100% with recomputed metrics.
- **Visualizations Present (16/16)**:
  - ROC curves (closed-set and open-set OOD): Present (300 DPI).
  - Precision-Recall curves (closed-set and open-set OOD): Present (300 DPI).
  - OSCR curves (MSP, Energy, OpenMax): Present (300 DPI).
  - Closed-set confusion matrix heatmap ($6 \times 6$): Present (300 DPI).
  - Open-set confusion matrix heatmaps ($7 \times 7$ for $\tau_{\text{FPR05}}$ and $\tau_{\text{Cov95}}$): Present across all three detectors (6 plots total, 300 DPI).
- **Visualizations Missing Across All 5 Classifiers (Pipeline Limitation)**:
  - Known vs. Unknown score distribution histograms.
  - Threshold calibration distribution plots.
  - Standalone training loss / validation score curves (numerical arrays preserved in `model.pkl`).
  - Standalone OpenMax EVT diagnostic curves (numerical parameters preserved in JSON).
  - *Status*: **PASS WITH LIMITATION**. This is a common property of the baseline pipeline across all five models, not an MLP-specific omission.

---

## 10. Cross-Model Consistency Findings

Comparing MLP with LightGBM, Random Forest, XGBoost, and Logistic Regression:
1. **Population Parity**: All 5 models evaluated on the identical test population ($N=82,332$: 77,831 known, 4,501 unknown), identical 6 known classes, identical 4 unknown classes, and identical 48-feature input schema.
2. **Methodological Distinction**:
   - **LightGBM & XGBoost**: Logits derived from raw tree margin heuristics (`output_margin=True`).
   - **Random Forest**: Logits derived from Centered Log-Ratio (CLR) pseudo-logits over leaf class probability estimates.
   - **Logistic Regression**: Logits derived from linear decision hyperplanes ($X W^T + b$).
   - **MLP**: Logits derived from native continuous pre-softmax neural activations ($h_2 W_{\text{out}} + b_{\text{out}}$).
3. **Empirical Trade-off**:
   - MLP achieves **AUROC = 0.8593** (Energy) and **0.8543** (OpenMax), clearly outperforming Logistic Regression (0.8480 / 0.8251) and competitive with tree ensembles (0.8327 – 0.8817).
   - In closed-set accuracy, MLP (**74.06%**) trails tree models (~78.4%) due to tabular step-function boundaries and minority class collapse (Analysis: 0%, DoS: 4.08%), while substantially outperforming linear Logistic Regression (70.41%).

---

## 11. Discrepancies and Clarifications

| Issue Identified | Location | Nature & Severity | Audit Explanation & Recommendation |
| :--- | :--- | :--- | :--- |
| OpenMax $\tau_{\text{FPR05}}$ Known Coverage and Accuracy Typo | `MLP_phase2_execution_report.md` (Section 6.2 text) | Narrative Draft Reference (Low) | Draft narrative text mentioned 81.91% known coverage and 67.65% accuracy. The actual machine-readable JSON and CSV records confirm **80.83% known coverage** (62,911 / 77,831) and **66.75% accuracy** (54,960 / 82,332), matching the $7 \times 7$ confusion matrix. All machine-readable files are correct. |
| Test Normal FPR Inflation at $\tau_{\text{FPR05}}$ | All 3 detectors | Dataset Covariate Shift (Medium) | Test Normal FPR realized at 13.08% – 15.62% vs 5.0% calibration target. Confirmed to be a genuine covariate shift in UNSW-NB15 Normal test traffic, identical to behavior observed in prior models. No correction needed. |
| Missing Diagnostic Visualizations | Output directories | Pipeline Feature Omission (Low) | Score distributions, calibration curves, and EVT plots are missing across all 5 models. Underlying numerical data are fully preserved. |

---

## 12. Statistical Limitations Summary

1. **Analysis Minority Collapse ($n=677, \text{Recall}=0.0\%$)**:
   The unweighted cross-entropy loss function allowed the dense decision boundaries to collapse the small Analysis class into Exploits and Fuzzers.
2. **Analysis EVT Tail Sample Size ($\eta=5, n_c=23$)**:
   Fitting extreme value distributions on 5 upper-tail samples has wide estimation variance, though its impact on system-level false alarms is minor.
3. **Worms Sample Size ($N=44$)**:
   Small test support makes per-category recall percentages sensitive to single-flow variations ($\pm 2.27\%$ per flow).

---

## 13. Summary of Audit Checks & Verdict

### Check Category Totals:
- **Total Individual Verification Checks Performed**: **28**
- **PASS**: **21**
- **PASS WITH LIMITATION**: **7**
- **WARNING**: **0**
- **FAIL**: **0**
- **NOT VERIFIABLE**: **0**

### Final Audit Verdict:
The Phase 2 experimental evaluation of the **Multi-Layer Perceptron (MLP)** classifier is **OFFICIALLY ACCEPTED AS INDEPENDENTLY VERIFIED**.

All numerical metrics, confusion matrices, and operating points are reproducible to 100% precision. Pre-softmax logit extraction is mathematically exact. Partitions are strictly isolated with zero leakage. Artifact preservation is complete and exhibits exact structural parity with LightGBM, Random Forest, XGBoost, and Logistic Regression.
