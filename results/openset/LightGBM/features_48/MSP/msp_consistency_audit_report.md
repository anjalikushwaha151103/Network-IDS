# MSP Results Consistency Audit Report

This report presents the mathematical verification and consistency audit of the Maximum Softmax Probability (MSP) open-set evaluation artifacts saved under [`results/openset/LightGBM/features_48/MSP/`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/).

---

## 1. Executive Summary & Verification Findings

1. **Total Sample Count & Support Integrity**:
   - Both saved confusion matrices ([`confusion_matrix_7x7_fpr05.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/confusion_matrix_7x7_fpr05.csv) and [`confusion_matrix_7x7_cov95.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/confusion_matrix_7x7_cov95.csv)) sum to exactly **82,332** flows.
   - Row sums (supports) across both matrices match the official UNSW-NB15 test set categories to the single flow:
     - Normal: **37,000**
     - Analysis: **677**
     - Generic: **18,871**
     - Exploits: **11,132**
     - Fuzzers: **6,062**
     - DoS: **4,089**
     - Unknown (*Reconnaissance + Backdoor + Shellcode + Worms*): **4,501**
2. **Arithmetic Consistency**:
   - Recalculating Accuracy, Macro F1, Weighted F1, Normal FPR, Known Coverage, and Unknown Recall directly from matrix counts yields values that **identically match** [`metrics_openset_msp.json`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/metrics_openset_msp.json) across all decimal places (absolute difference $< 10^{-14}$).
3. **Source of Previous Discrepancy**:
   - The discrepancy identified arose from **chat response text drafting**: in the summary table of the previous response message, accuracy and weighted F1 were transcribed as `70.92%` and `72.63%` (for FPR05) instead of the true JSON values `70.75%` and `74.16%`.
   - The original CSV ([`metrics_openset_msp.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/metrics_openset_msp.csv)) contained the rates and macro F1, but omitted explicit rows for overall accuracy and weighted F1.
   - **No model was retrained, no thresholds were altered, and no underlying predictions were modified.**
   - A versioned audited summary has been persisted as [`metrics_openset_msp_audited.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/metrics_openset_msp_audited.csv) while preserving the original.

---

## 2. Mathematical Recalculations from Confusion Matrix Counts

### A. Operating Point 1: `tau_FPR05 = 0.448034`

From [`confusion_matrix_7x7_fpr05.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/confusion_matrix_7x7_fpr05.csv):

$$\text{Total Elements } N = 82,332$$
$$\text{Diagonal Elements (Correct Predictions)} = 26,215 + 54 + 18,271 + 7,691 + 2,974 + 241 + 2,804 = \mathbf{58,250}$$

1. **Open-Set 7-Class Accuracy**:
   $$\text{Accuracy} = \frac{58,250}{82,332} = \mathbf{0.707501336054...} \quad (\mathbf{70.7501\%})$$
2. **False Positive Rate on Normal Traffic**:
   $$\text{Normal Support} = 37,000, \quad \text{Normal Flagged as Unknown} = 4,653$$
   $$\text{FPR}_{\text{Normal}} = \frac{4,653}{37,000} = \mathbf{0.1257567567...} \quad (\mathbf{12.5757\%})$$
3. **Known-Traffic Rejection Rate & Coverage**:
   $$\text{Known Support} = 77,831, \quad \text{Known Flagged as Unknown} = 4,653 + 557 + 244 + 2,842 + 2,349 + 2,436 = 13,085$$
   $$\text{Rejection Rate}_{\text{Known}} = \frac{13,085}{77,831} = \mathbf{0.1680692783...} \quad (\mathbf{16.8069\%})$$
   $$\text{Known Coverage} = 1 - 0.16806928 = \mathbf{0.8319307217...} \quad (\mathbf{83.1931\%})$$
4. **Unknown Attack Detection Rate (Recall on Unknowns)**:
   $$\text{Unknown Support} = 4,501, \quad \text{Unknown Flagged as Unknown} = 2,804$$
   $$\text{Recall}_{\text{Unknown}} = \frac{2,804}{4,501} = \mathbf{0.6229726727...} \quad (\mathbf{62.2973\%})$$
5. **Per-Class F1 and Macro / Weighted F1**:
   - Normal F1: $0.82312065$
   - Analysis F1: $0.04940531$
   - Generic F1: $0.98334812$
   - Exploits F1: $0.71747796$
   - Fuzzers F1: $0.38670655$
   - DoS F1: $0.10932184$
   - Unknown F1: $0.27506386$
   - **Macro F1**: $\frac{1}{7} \sum \text{F1}_i = \mathbf{0.4777689407...} \quad (\mathbf{47.7769\%})$
   - **Weighted F1**: $\frac{1}{82332} \sum (\text{F1}_i \times N_i) = \mathbf{0.7416237626...} \quad (\mathbf{74.1624\%})$

---

### B. Operating Point 2: `tau_Cov95 = 0.526492`

From [`confusion_matrix_7x7_cov95.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/confusion_matrix_7x7_cov95.csv):

$$\text{Total Elements } N = 82,332$$
$$\text{Diagonal Elements (Correct Predictions)} = 28,080 + 60 + 18,303 + 9,558 + 3,544 + 283 + 2,058 = \mathbf{61,887}$$

1. **Open-Set 7-Class Accuracy**:
   $$\text{Accuracy} = \frac{61,887}{82,332} = \mathbf{0.7516639945...} \quad (\mathbf{75.1664\%})$$
2. **False Positive Rate on Normal Traffic**:
   $$\text{Normal Support} = 37,000, \quad \text{Normal Flagged as Unknown} = 799$$
   $$\text{FPR}_{\text{Normal}} = \frac{799}{37,000} = \mathbf{0.0215945945...} \quad (\mathbf{2.1595\%})$$
3. **Known-Traffic Rejection Rate & Coverage**:
   $$\text{Known Support} = 77,831, \quad \text{Known Flagged as Unknown} = 799 + 161 + 121 + 865 + 717 + 712 = 3,375$$
   $$\text{Rejection Rate}_{\text{Known}} = \frac{3,375}{77,831} = \mathbf{0.0433631843...} \quad (\mathbf{4.3363\%})$$
   $$\text{Known Coverage} = 1 - 0.04336318 = \mathbf{0.9566368157...} \quad (\mathbf{95.6637\%})$$
4. **Unknown Attack Detection Rate (Recall on Unknowns)**:
   $$\text{Unknown Support} = 4,501, \quad \text{Unknown Flagged as Unknown} = 2,058$$
   $$\text{Recall}_{\text{Unknown}} = \frac{2,058}{4,501} = \mathbf{0.4572317262...} \quad (\mathbf{45.7232\%})$$
5. **Per-Class F1 and Macro / Weighted F1**:
   - Normal F1: $0.85243886$
   - Analysis F1: $0.05063291$
   - Generic F1: $0.98403226$
   - Exploits F1: $0.71171804$
   - Fuzzers F1: $0.39575656$
   - DoS F1: $0.12569403$
   - Unknown F1: $0.41425121$
   - **Macro F1**: $\frac{1}{7} \sum \text{F1}_i = \mathbf{0.5049243886...} \quad (\mathbf{50.4924\%})$
   - **Weighted F1**: $\frac{1}{82332} \sum (\text{F1}_i \times N_i) = \mathbf{0.7632825640...} \quad (\mathbf{76.3283\%})$

---

## 3. Side-by-Side Arithmetic Audit Table

| Metric | Direct Matrix Count | `metrics_openset_msp.json` | `metrics_openset_msp.csv` | Match Status |
| :--- | :---: | :---: | :---: | :---: |
| **tau_FPR05 Threshold** | `0.448033758661` | `0.448033758661` | `0.448033758661` | **EXACT MATCH** |
| **tau_FPR05 Accuracy** | **70.7501%** | **70.7501%** | *(added to audited CSV)* | **EXACT MATCH** |
| **tau_FPR05 Macro F1** | **47.7769%** | **47.7769%** | `0.477768940739` | **EXACT MATCH** |
| **tau_FPR05 Weighted F1**| **74.1624%** | **74.1624%** | *(added to audited CSV)* | **EXACT MATCH** |
| **tau_FPR05 Normal FPR** | **12.5757%** | **12.5757%** | `0.125756756757` | **EXACT MATCH** |
| **tau_FPR05 Known Rejection** | **16.8069%** | **16.8069%** | *(derived from cov)* | **EXACT MATCH** |
| **tau_FPR05 Known Coverage** | **83.1931%** | **83.1931%** | `0.831930721692` | **EXACT MATCH** |
| **tau_FPR05 Unknown Recall** | **62.2973%** | **62.2973%** | `0.622972672739` | **EXACT MATCH** |
| **tau_Cov95 Threshold** | `0.526491895397` | `0.526491895397` | `0.526491895397` | **EXACT MATCH** |
| **tau_Cov95 Accuracy** | **75.1664%** | **75.1664%** | *(added to audited CSV)* | **EXACT MATCH** |
| **tau_Cov95 Macro F1** | **50.4924%** | **50.4924%** | `0.504924388645` | **EXACT MATCH** |
| **tau_Cov95 Weighted F1**| **76.3283%** | **76.3283%** | *(added to audited CSV)* | **EXACT MATCH** |
| **tau_Cov95 Normal FPR** | **2.1595%** | **2.1595%** | `0.021594594595` | **EXACT MATCH** |
| **tau_Cov95 Known Rejection** | **4.3363%** | **4.3363%** | *(derived from cov)* | **EXACT MATCH** |
| **tau_Cov95 Known Coverage** | **95.6637%** | **95.6637%** | `0.956636815665` | **EXACT MATCH** |
| **tau_Cov95 Unknown Recall** | **45.7232%** | **45.7232%** | `0.457231726283` | **EXACT MATCH** |

---

## 4. Test Verification Status

The automated consistency test [`test_confusion_matrix_metrics_consistency`](file:///d:/Network%20IDS/tests/test_openset_msp.py#L174-L244) was executed via:

```powershell
python -m pytest tests/test_openset_msp.py -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.13.2, pytest-8.4.2, pluggy-1.6.0 -- C:\Python313\python.exe
cachedir: .pytest_cache
rootdir: D:\Network IDS
plugins: anyio-4.14.2, platformdirs-4.12.2
collecting ... collected 6 items

tests/test_openset_msp.py::test_msp_scores_bounded_and_monotonic PASSED        [ 16%]
tests/test_openset_msp.py::test_thresholds_calculated_exclusively_from_d_threshold_cal PASSED [ 33%]
tests/test_openset_msp.py::test_predicted_labels_restricted_to_0_through_6 PASSED [ 50%]
tests/test_openset_msp.py::test_both_thresholds_produce_separate_evaluations PASSED [ 66%]
tests/test_openset_msp.py::test_saved_results_reproducible_and_loadable PASSED [ 83%]
tests/test_openset_msp.py::test_confusion_matrix_metrics_consistency PASSED   [100%]

======================== 6 passed in 83.37s (0:01:23) =========================
```

The audit confirms full mathematical agreement across all metrics and confusion matrices. All original artifacts remain preserved.
