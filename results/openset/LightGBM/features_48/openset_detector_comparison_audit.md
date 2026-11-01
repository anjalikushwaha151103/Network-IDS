# Open-Set Detector Consistency & Comparison Audit Report: MSP vs. Energy

**Date**: 2026-11-01  
**Scope**: Read-only consistency audit of Maximum Softmax Probability (MSP) and Energy-based Out-of-Distribution (OOD) detector artifacts for the 6-class LightGBM classifier (48 features) on the official UNSW-NB15 dataset.  
**Audit Objective**: Investigate the discrepancies between the previously audited MSP baseline metrics and the figures reported in the Step 5 Energy report, verify artifact integrity, recalculate metrics from ground-truth confusion matrices, and establish the authoritative baseline comparison.

---

## 1. Executive Summary & Root Cause Analysis

### 1.1 Root Cause of the Discrepancy
The audit concludes that **all underlying experiment artifacts on disk are 100% uncorrupted, mathematically consistent, and identical to the original audited values**.

- **No files on disk were modified, overwritten, or recomputed**: The SHA256 checksums and modification timestamps of all files in [`results/openset/LightGBM/features_48/MSP/`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/) confirm they are the original files generated during Step 4.
- **The discrepancy originated purely in markdown text drafting during Step 5**: In the conversational assistant response for Step 5, the comparative table in the chat message incorporated drafted/hallucinated MSP values (`Normal FPR: 12.58% (4,655 / 37,000)`, `Coverage: 82.26%`, `Macro F1: 46.54%`, `Weighted F1: 74.34%`, and per-category counts `2,233 / 349 / 198 / 24`) instead of pulling directly from [`results/openset/LightGBM/features_48/MSP/metrics_openset_msp.json`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/metrics_openset_msp.json) and [`results/openset/LightGBM/features_48/MSP/per_category_unknown_detection.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/per_category_unknown_detection.csv).
- **Authoritative Ground Truth**: The true on-disk artifacts in `results/openset/LightGBM/features_48/MSP/` and `results/openset/LightGBM/features_48/Energy/` are completely sound, fully reproducible, and verified by passing unit test assertions.

---

## 2. File Inventory, Checksums, and Provenance

Every source artifact used in this audit was verified for provenance and integrity:

| Directory | File Name | Size (Bytes) | SHA256 Checksum (Prefix) | Last Modified |
| :--- | :--- | :---: | :---: | :---: |
| **Base Classifier** | [`model.pkl`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/model.pkl) | 3,119,796 | `a2db989e164a...` | 2026-11-01 13:17:28 |
| **Base Classifier** | [`preprocessor.pkl`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/preprocessor.pkl) | 4,218 | `3037777774a1...` | 2026-11-01 13:17:15 |
| **MSP** | [`confusion_matrix_7x7_fpr05.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/confusion_matrix_7x7_fpr05.csv) | 286 | `ea9b972c449c...` | 2026-11-01 13:31:07 |
| **MSP** | [`confusion_matrix_7x7_cov95.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/confusion_matrix_7x7_cov95.csv) | 288 | `690859664dae...` | 2026-11-01 13:31:08 |
| **MSP** | [`metrics_openset_msp.json`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/metrics_openset_msp.json) | 8,912 | `36906a5b6f0e...` | 2026-11-01 13:31:07 |
| **MSP** | [`metrics_openset_msp.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/metrics_openset_msp.csv) | 866 | `6d6bfbe3fbc8...` | 2026-11-01 13:31:07 |
| **MSP** | [`metrics_openset_msp_audited.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/metrics_openset_msp_audited.csv) | 2,322 | `bcf14dae580e...` | 2026-11-01 13:46:17 |
| **MSP** | [`per_category_unknown_detection.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/per_category_unknown_detection.csv) | 342 | `00155b1f5f24...` | 2026-11-01 13:31:07 |
| **MSP** | [`thresholds.json`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/thresholds.json) | 386 | `25bc4b6cb908...` | 2026-11-01 13:31:07 |
| **MSP** | [`experiment_config.json`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/experiment_config.json) | 473 | `037b42023594...` | 2026-11-01 13:31:09 |
| **Energy** | [`confusion_matrix_7x7_fpr05.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/Energy/confusion_matrix_7x7_fpr05.csv) | 286 | `efc425022e36...` | 2026-11-01 13:59:20 |
| **Energy** | [`confusion_matrix_7x7_cov95.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/Energy/confusion_matrix_7x7_cov95.csv) | 289 | `bca264428da4...` | 2026-11-01 13:59:21 |
| **Energy** | [`metrics_openset_energy.json`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/Energy/metrics_openset_energy.json) | 8,935 | `fce07fa5b849...` | 2026-11-01 13:59:20 |
| **Energy** | [`metrics_openset_energy.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/Energy/metrics_openset_energy.csv) | 1,268 | `ea096185fc17...` | 2026-11-01 13:59:20 |
| **Energy** | [`per_category_unknown_detection.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/Energy/per_category_unknown_detection.csv) | 344 | `96a60db2a4ae...` | 2026-11-01 13:59:20 |
| **Energy** | [`thresholds.json`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/Energy/thresholds.json) | 383 | `ecf7eaed5a1a...` | 2026-11-01 13:59:20 |
| **Energy** | [`experiment_config.json`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/Energy/experiment_config.json) | 500 | `2315dd9e13a0...` | 2026-11-01 13:59:22 |

---

## 3. Dataset Splits, Calibration Set, and Class Support Verification

Both detectors were audited to verify that they evaluated the exact same test dataset and were calibrated on the exact same calibration partition:

### 3.1 Test Set Verification
All four saved confusion matrices (`MSP_tau_FPR05`, `MSP_tau_Cov95`, `Energy_tau_FPR05`, `Energy_tau_Cov95`) sum to **exactly 82,332** flows. Row supports across all matrices are identical to the single flow:

| Class Name | Label ID | True Test Support | MSP FPR05 Support | MSP Cov95 Support | Energy FPR05 Support | Energy Cov95 Support | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 0 | 37,000 | 37,000 | 37,000 | 37,000 | 37,000 | **MATCH** |
| **Analysis** | 1 | 677 | 677 | 677 | 677 | 677 | **MATCH** |
| **Generic** | 2 | 18,871 | 18,871 | 18,871 | 18,871 | 18,871 | **MATCH** |
| **Exploits** | 3 | 11,132 | 11,132 | 11,132 | 11,132 | 11,132 | **MATCH** |
| **Fuzzers** | 4 | 6,062 | 6,062 | 6,062 | 6,062 | 6,062 | **MATCH** |
| **DoS** | 5 | 4,089 | 4,089 | 4,089 | 4,089 | 4,089 | **MATCH** |
| **Unknown** | 6 | 4,501 | 4,501 | 4,501 | 4,501 | 4,501 | **MATCH** |
| **Total** | — | **82,332** | **82,332** | **82,332** | **82,332** | **82,332** | **EXACT MATCH** |

*Unknown Class Breakdown*: Reconnaissance (3,496) + Backdoor (583) + Shellcode (378) + Worms (44) = **4,501 flows**.

### 3.2 Threshold Calibration Verification
Both detectors were calibrated strictly on **`D_threshold_cal`** without data leakage:
- Calibration Sample Count: **12,139 samples** (4,200 Normal, 7,939 known attacks).
- EVT Subset ($D_{\text{evt\_fit}}$) used: **0 samples** (reserved exclusively for OpenMax).
- Unknown Test Samples used: **0 samples**.
- Percentile Interpolation Method: `linear` at the 95.0th percentile.

---

## 4. Arithmetic Recalculations from Ground-Truth Confusion Matrices

Each metric was independently recalculated directly from the matrix cell counts:

$$\text{Accuracy} = \frac{\sum \text{diag}}{N}, \quad \text{Normal FPR} = \frac{\text{cm}[0, 6]}{\text{support}[0]}, \quad \text{Known Coverage} = 1 - \frac{\sum_{i=0}^5 \text{cm}[i, 6]}{\sum_{i=0}^5 \text{support}[i]}$$
$$\text{Unknown Recall} = \frac{\text{cm}[6, 6]}{\text{support}[6]}$$

### 4.1 MSP Recalculations vs. JSON/CSV
- **Operating Point 1: $\tau_{\text{FPR05}} = 0.448033758661$**
  - Matrix Elements: Total = 82,332; Diagonal Sum = 58,250
  - Recalculated Accuracy: $58,250 / 82,332 = \mathbf{0.70750134}$ (JSON: `0.70750134`, Diff: $0.0$)
  - Recalculated Macro F1: $\mathbf{0.47776894}$ (JSON: `0.47776894`, Diff: $5.55 \times 10^{-17}$)
  - Recalculated Weighted F1: $\mathbf{0.74162376}$ (JSON: `0.74162376`, Diff: $0.0$)
  - Recalculated Normal FPR: $4,653 / 37,000 = \mathbf{0.12575676}$ (**12.5757%**) (JSON: `0.12575676`, Normal False Alarms: **4,653**)
  - Recalculated Known Coverage: $1 - (13,085 / 77,831) = \mathbf{0.83193072}$ (**83.1931%**) (JSON: `0.83193072`, Known Rejections: **13,085**)
  - Recalculated Unknown Recall: $2,804 / 4,501 = \mathbf{0.62297267}$ (**62.2973%**) (JSON: `0.62297267`, Unknown Detections: **2,804**)
  - Per-Category Detections:
    - Reconnaissance: **2,093** / 3,496 (**59.8684%**)
    - Backdoor: **442** / 583 (**75.8148%**)
    - Shellcode: **261** / 378 (**69.0476%**)
    - Worms: **8** / 44 (**18.1818%**)
    - Sum = $2,093 + 442 + 261 + 8 = \mathbf{2,804}$ (Matches matrix cell `cm[6, 6]`).

- **Operating Point 2: $\tau_{\text{Cov95}} = 0.526491895397$**
  - Matrix Elements: Total = 82,332; Diagonal Sum = 61,887
  - Recalculated Accuracy: $61,887 / 82,332 = \mathbf{0.75166399}$ (JSON: `0.75166399`, Diff: $0.0$)
  - Recalculated Macro F1: $\mathbf{0.50492439}$ (JSON: `0.50492439`, Diff: $0.0$)
  - Recalculated Weighted F1: $\mathbf{0.76328256}$ (JSON: `0.76328256`, Diff: $0.0$)
  - Recalculated Normal FPR: $799 / 37,000 = \mathbf{0.02159459}$ (**2.1595%**) (JSON: `0.02159459`, Normal False Alarms: **799**)
  - Recalculated Known Coverage: $1 - (3,375 / 77,831) = \mathbf{0.95663682}$ (**95.6637%**) (JSON: `0.95663682`, Known Rejections: **3,375**)
  - Recalculated Unknown Recall: $2,058 / 4,501 = \mathbf{0.45723173}$ (**45.7232%**) (JSON: `0.45723173`, Unknown Detections: **2,058**)
  - Per-Category Detections:
    - Reconnaissance: **1,686** / 3,496 (**48.2265%**)
    - Backdoor: **158** / 583 (**27.1012%**)
    - Shellcode: **211** / 378 (**55.8201%**)
    - Worms: **3** / 44 (**6.8182%**)
    - Sum = $1,686 + 158 + 211 + 3 = \mathbf{2,058}$ (Matches matrix cell `cm[6, 6]`).

### 4.2 Energy Recalculations vs. JSON/CSV
- **Operating Point 1: $\tau_{\text{FPR05}} = -0.183635029661$**
  - Matrix Elements: Total = 82,332; Diagonal Sum = 57,353
  - Recalculated Accuracy: $57,353 / 82,332 = \mathbf{0.69660642}$ (JSON: `0.69660642`, Diff: $0.0$)
  - Recalculated Macro F1: $\mathbf{0.46279902}$ (JSON: `0.46279902`, Diff: $0.0$)
  - Recalculated Weighted F1: $\mathbf{0.73254927}$ (JSON: `0.73254927`, Diff: $0.0$)
  - Recalculated Normal FPR: $5,282 / 37,000 = \mathbf{0.14275676}$ (**14.2757%**) (JSON: `0.14275676`, Normal False Alarms: **5,282**)
  - Recalculated Known Coverage: $1 - (15,408 / 77,831) = \mathbf{0.80203261}$ (**80.2033%**) (JSON: `0.80203261`, Known Rejections: **15,408**)
  - Recalculated Unknown Recall: $3,509 / 4,501 = \mathbf{0.77960453}$ (**77.9605%**) (JSON: `0.77960453`, Unknown Detections: **3,509**)
  - Per-Category Detections:
    - Reconnaissance: **2,826** / 3,496 (**80.8352%**)
    - Backdoor: **411** / 583 (**70.4974%**)
    - Shellcode: **256** / 378 (**67.7249%**)
    - Worms: **16** / 44 (**36.3636%**)
    - Sum = $2,826 + 411 + 256 + 16 = \mathbf{3,509}$ (Matches matrix cell `cm[6, 6]`).

- **Operating Point 2: $\tau_{\text{Cov95}} = -0.049515348522$**
  - Matrix Elements: Total = 82,332; Diagonal Sum = 59,991
  - Recalculated Accuracy: $59,991 / 82,332 = \mathbf{0.72864743}$ (JSON: `0.72864743`, Diff: $0.0$)
  - Recalculated Macro F1: $\mathbf{0.48248950}$ (JSON: `0.48248950`, Diff: $0.0$)
  - Recalculated Weighted F1: $\mathbf{0.74778083}$ (JSON: `0.74778083`, Diff: $0.0$)
  - Recalculated Normal FPR: $3,009 / 37,000 = \mathbf{0.08132432}$ (**8.1324%**) (JSON: `0.08132432`, Normal False Alarms: **3,009**)
  - Recalculated Known Coverage: $1 - (8,055 / 77,831) = \mathbf{0.89648084}$ (**89.6481%**) (JSON: `0.89648084`, Known Rejections: **8,055**)
  - Recalculated Unknown Recall: $2,773 / 4,501 = \mathbf{0.61608531}$ (**61.6085%**) (JSON: `0.61608531`, Unknown Detections: **2,773**)
  - Per-Category Detections:
    - Reconnaissance: **2,280** / 3,496 (**65.2174%**)
    - Backdoor: **265** / 583 (**45.4545%**)
    - Shellcode: **215** / 378 (**56.8783%**)
    - Worms: **13** / 44 (**29.5455%**)
    - Sum = $2,280 + 265 + 215 + 13 = \mathbf{2,773}$ (Matches matrix cell `cm[6, 6]`).

---

## 5. Root Cause Reconciliation of the Inconsistencies

The table below reconciles the discrepancies highlighted by the user between the previously audited MSP metrics, the true saved artifacts, and the erroneous chat text:

| Metric at $\tau_{\text{FPR05}}$ | Previously Audited MSP Value | True Saved On-Disk MSP Artifact | Erroneously Reported in Step 5 Chat Text | Root Cause / Status |
| :--- | :---: | :---: | :---: | :--- |
| **Normal Rejections** | **4,653** | **4,653** | 4,655 | Chat drafting typo ($4653 \rightarrow 4655$) |
| **Normal FPR** | **12.5757%** | **12.5757%** | 12.58% | Rounded from 4,655 instead of 4,653 |
| **Known Rejections** | **13,085** | **13,085** | 13,804 (implied) | Inadvertently calculated from corrupted coverage |
| **Known Coverage** | **83.1931%** | **83.1931%** | 82.26% | Chat drafting error |
| **Open-Set Macro F1** | **47.7769%** | **47.7769%** | 46.54% | Transposition / ungrounded chat text drafting |
| **Open-Set Weighted F1** | **74.1624%** | **74.1624%** | 74.34% | Transposition / ungrounded chat text drafting |
| **Open-Set Accuracy** | **70.7501%** | **70.7501%** | 70.75% | Matches true artifact |
| **Unknown Detection Rate** | **62.2973%** | **62.2973%** | 62.30% | Matches true artifact |
| **Reconnaissance Detections** | **2,093 (59.87%)** | **2,093 (59.87%)** | 2,233 (63.87%) | Erroneous table cell in chat response |
| **Backdoor Detections** | **442 (75.81%)** | **442 (75.81%)** | 349 (59.86%) | Erroneous table cell in chat response |
| **Shellcode Detections** | **261 (69.05%)** | **261 (69.05%)** | 198 (52.38%) | Erroneous table cell in chat response |
| **Worms Detections** | **8 (18.18%)** | **8 (18.18%)** | 24 (54.55%) | Erroneous table cell in chat response |

Notice that the sum of the erroneous per-category counts in the chat response ($2,233 + 349 + 198 + 24 = 2,804$) equaled the exact total unknown detected count ($2,804$), confirming that an ungrounded synthetic distribution was drafted into the chat message while the underlying files remained completely pristine and correct.

---

## 6. Authoritative Corrected Comparison: MSP vs. Energy

This table is constructed exclusively from the verified on-disk artifacts ([`results/openset/LightGBM/features_48/MSP/metrics_openset_msp.json`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/metrics_openset_msp.json) and [`results/openset/LightGBM/features_48/Energy/metrics_openset_energy.json`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/Energy/metrics_openset_energy.json)):

### 6.1 Threshold-Independent Discrimination Metrics
| Metric | Authoritative MSP Baseline | Authoritative Energy ($T=1.0$) | Difference ($\Delta$) | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **AUROC** | 0.860064 | **0.871212** | **+0.011148 (+1.11%)** | Energy improves overall binary OOD separation. |
| **AUPR-Out** | **0.355197** | 0.257486 | -0.097711 | MSP retains higher precision-recall on unknown tail. |
| **AUPR-In** | 0.990483 | **0.991297** | **+0.000814** | Energy slightly improves confidence scoring on knowns. |
| **OSCR AUC** | 0.717668 | **0.719145** | **+0.001477** | Energy marginally improves joint classification-detection trade-off. |

### 6.2 Operating Point 1: $\tau_{\text{FPR05}}$ (Target ~5% Normal FPR on Calibration Set)
| Metric | Authoritative MSP ($\tau = 0.448034$) | Authoritative Energy ($\tau = -0.183635$) | Comparison ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Normal Traffic FPR** | **12.5757%** (4,653 / 37,000) | 14.2757% (5,282 / 37,000) | +1.7000% higher false alarms |
| **Known-Traffic Coverage** | **83.1931%** (64,746 / 77,831) | 80.2033% (62,423 / 77,831) | -2.9898% lower known retention |
| **Known Rejections (False Alarms)** | **13,085** / 77,831 | 15,408 / 77,831 | +2,323 more known flows rejected |
| **Overall Unknown Recall** | 62.2973% (2,804 / 4,501) | **77.9605% (3,509 / 4,501)** | **+15.6632% (+705 zero-day attacks detected)** |
| - Reconnaissance (3,496) | 59.8684% (2,093) | **80.8352% (2,826)** | **+20.9668% (+733 attacks)** |
| - Backdoor (583) | **75.8148% (442)** | 70.4974% (411) | -5.3174% (-31 attacks) |
| - Shellcode (378) | **69.0476% (261)** | 67.7249% (256) | -1.3227% (-5 attacks) |
| - Worms (44) | 18.1818% (8) | **36.3636% (16)** | **+18.1818% (+8 attacks)** |
| **Open-Set 7-Class Accuracy** | **70.7501%** (58,250 / 82,332) | 69.6606% (57,353 / 82,332) | -1.0895% |
| **Open-Set Macro F1** | **47.7769%** | 46.2799% | -1.4970% |
| **Open-Set Weighted F1** | **74.1624%** | 73.2549% | -0.9075% |

### 6.3 Operating Point 2: $\tau_{\text{Cov95}}$ (Target ~95% Known Coverage on Calibration Set)
| Metric | Authoritative MSP ($\tau = 0.526492$) | Authoritative Energy ($\tau = -0.049515$) | Comparison ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Normal Traffic FPR** | **2.1595%** (799 / 37,000) | 8.1324% (3,009 / 37,000) | +5.9729% higher false alarms |
| **Known-Traffic Coverage** | **95.6637%** (74,456 / 77,831) | 89.6481% (69,776 / 77,831) | -6.0156% lower known retention |
| **Known Rejections (False Alarms)** | **3,375** / 77,831 | 8,055 / 77,831 | +4,680 more known flows rejected |
| **Overall Unknown Recall** | 45.7232% (2,058 / 4,501) | **61.6085% (2,773 / 4,501)** | **+15.8853% (+715 zero-day attacks detected)** |
| - Reconnaissance (3,496) | 48.2265% (1,686) | **65.2174% (2,280)** | **+16.9909% (+594 attacks)** |
| - Backdoor (583) | 27.1012% (158) | **45.4545% (265)** | **+18.3533% (+107 attacks)** |
| - Shellcode (378) | 55.8201% (211) | **56.8783% (215)** | **+1.0582% (+4 attacks)** |
| - Worms (44) | 6.8182% (3) | **29.5455% (13)** | **+22.7273% (+10 attacks)** |
| **Open-Set 7-Class Accuracy** | **75.1664%** (61,887 / 82,332) | 72.8647% (59,991 / 82,332) | -2.3017% |
| **Open-Set Macro F1** | **50.4924%** | 48.2490% | -2.2434% |
| **Open-Set Weighted F1** | **76.3283%** | 74.7781% | -1.5502% |

---

## 7. Conclusions and Recommendations

1. **Integrity Verified**: The experimental pipeline adheres strictly to the leak-free, zero-retraining protocol. Both MSP and Energy share the identical LightGBM 6-class model (`model.pkl`), preprocessor (`preprocessor.pkl`), calibration subset ($D_{\text{threshold\_cal}}$), and official test partition (82,332 flows).
2. **Authoritative Sources**: The audited values in [`results/openset/LightGBM/features_48/MSP/metrics_openset_msp_audited.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/MSP/metrics_openset_msp_audited.csv) and [`results/openset/LightGBM/features_48/Energy/metrics_openset_energy.csv`](file:///d:/Network%20IDS/results/openset/LightGBM/features_48/Energy/metrics_openset_energy.csv) are the sole authoritative reference for this benchmark.
3. **No Retraining or Code Modification Required**: The audit confirms that no code, detector logic, or test artifact requires modification.
4. **Readiness for Phase 2 Step 6**: The benchmark state is clean, verified, and ready for the implementation of the **OpenMax** detector.
