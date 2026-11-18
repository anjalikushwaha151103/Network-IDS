# Three-Model Consolidated Open-Set Comparison & Architectural Synthesis
## LightGBM vs. Random Forest vs. XGBoost

**Project:** Explainable AI-Based Network Intrusion Detection System (XAI-NIDS) with Open-Set / Zero-Day Attack Detection  
**Repository:** `AshwaniArya2005/Network-Intrusion-Detection-System`  
**Dataset:** UNSW-NB15 (48 actual network features, identical pipeline)  
**Evaluation Set:** Official untouched test split ($N = 82,332$: 77,831 Known, 4,501 Unknown across 4 zero-day classes)  
**Evaluated Classifiers:** LightGBM vs. Random Forest vs. XGBoost  
**Evaluated Detectors:** Maximum Softmax Probability (MSP), Energy-Based Model (EBM), Extreme Value Theory OpenMax  
**Synthesis Directory:** `results/openset/synthesis_3models/`  
**Date:** November 2026  

---

## 1. Executive Summary

This report extends the Phase 2 research synthesis by incorporating **XGBoost** into a unified, three-way architectural benchmark alongside **LightGBM** and **Random Forest**. By evaluating these three distinct tree-based ensemble paradigms—leaf-wise gradient boosting (LightGBM), bagging ensemble voting (Random Forest), and depth-wise gradient boosting (XGBoost)—under identical experimental conditions, we isolate the fundamental interactions between ensemble tree structures, activation representations (raw additive margins vs. probability-derived pseudo-logits), and open-set detection algorithms.

### Master Key Findings:
1. **Closed-Set Parity:** All three models achieve virtually identical closed-set test accuracy ($78.24\% - 78.48\%$) and weighted F1 ($79.03\% - 79.35\%$). LightGBM maintains the highest closed-set Macro F1 ($53.85\%$) due to slightly better sensitivity on minority attack classes (Analysis recall $12.26\%$ vs. $8.42\%$ in XGBoost and $5.91\%$ in RF).
2. **Open-Set Discrimination Hierarchy (AUROC):**
   $$\text{LightGBM OpenMax } (0.8817) > \text{LightGBM Energy } (0.8712) \approx \text{RF Energy } (0.8699) > \text{XGBoost OpenMax } (0.8614) > \text{LightGBM MSP } (0.8601) > \text{XGBoost MSP } (0.8546) > \text{RF MSP } (0.8523) > \text{XGBoost Energy } (0.8327) \gg \text{RF OpenMax } (0.8050)$$
3. **The GBDT Continuous Margin Validation:** XGBoost confirms our central hypothesis from Step 6: **gradient boosted trees with continuous raw additive margins (`output_margin=True`) provide statistically sound Class Activation Vectors for EVT OpenMax**. Unlike Random Forest (whose discrete voting probabilities caused Weibull shape collapse to $k = 0.0294$), XGBoost produced well-behaved continuous Weibull distributions across all 6 classes ($0.9289 \le k \le 1.5169$) with **zero fallbacks**, achieving $\text{AUROC} = 0.8614$ and $\text{OSCR} = 0.7200$.
4. **Energy Dynamics Divergence:** While Free Energy scored near-identically on LightGBM ($0.8712$) and RF ($0.8699$), it yielded lower discrimination on XGBoost ($0.8327$). In XGBoost, calibration energy scores were tightly packed near the boundary ($\tau_{\text{FPR05}} = -2.2678$ vs. $\tau_{\text{Cov95}} = -2.2500$, $\Delta = 0.0178$), showing that depth-wise tree margins concentrate distributional energy differently than leaf-wise margins.
5. **Operational Trade-Offs at $\tau_{\text{Cov95}}$:**
   - **LightGBM MSP:** Maximizes known coverage ($95.66\%$) and minimizes Normal false alarms ($2.16\%$), but detects only $45.72\%$ of zero-days.
   - **XGBoost OpenMax:** Delivers a balanced operational point, preserving $86.71\%$ known coverage and $9.08\%$ Normal FPR while detecting **$61.50\%$** of zero-day flows.
   - **LightGBM OpenMax:** Delivers the overall best trade-off, preserving $91.26\%$ known coverage and $4.63\%$ Normal FPR while detecting $57.92\%$ of zero-days with the highest open-set Macro F1 ($50.29\%$).

---

## 2. Closed-Set Classifier Comparison

All three classifiers were trained on $\mathcal{D}_{\text{train\_known}}$ ($N = 113,288$) using identical 48-feature preprocessors fitted exclusively on training data. Closed-set performance on $\mathcal{D}_{\text{test\_known}}$ ($N = 77,831$) is summarized below:

| Metric | LightGBM | Random Forest | XGBoost | Best Model | Source Artifact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Test Accuracy** | $78.45\%$ | **$78.48\%$** | $78.24\%$ | Random Forest (+0.03%) | `metrics_closed_set.json` |
| **Test Macro F1** | **$53.85\%$** | $53.20\%$ | $53.29\%$ | LightGBM (+0.56%) | `metrics_closed_set.json` |
| **Test Weighted F1** | **$79.35\%$** | $79.19\%$ | $79.03\%$ | LightGBM (+0.16%) | `metrics_closed_set.json` |
| **Validation Accuracy** | **$85.21\%$** | $85.07\%$ | $85.02\%$ | LightGBM (+0.14%) | `metrics_closed_set.json` |
| **Validation Macro F1** | **$65.68\%$** | $65.18\%$ | $65.30\%$ | LightGBM (+0.38%) | `metrics_closed_set.json` |
| **Training Duration** | $13.65\text{ s}$ | **$9.14\text{ s}$** | $19.95\text{ s}$ | Random Forest (Fastest) | `metrics_closed_set.json` |
| **Batch Scoring Latency** | $0.0374\text{ ms}$ | **$0.0108\text{ ms}$** | $0.0175\text{ ms}$ | Random Forest (3.5x faster) | `metrics_closed_set.json` |

### Per-Class Test Performance Matrix

```
                      LightGBM                         Random Forest                          XGBoost
Class         Prec.   Recall   F1-Score         Prec.   Recall   F1-Score         Prec.   Recall   F1-Score
Normal        0.9687  0.7658    0.8553          0.9655  0.7696    0.8565          0.9683  0.7607    0.8521
Analysis      0.0456  0.1226    0.0665          0.0342  0.0591    0.0433          0.0376  0.0842    0.0520
Generic       0.9977  0.9708    0.9841          0.9990  0.9675    0.9830          0.9952  0.9717    0.9833
Exploits      0.6249  0.9111    0.7413          0.6124  0.9233    0.7364          0.6156  0.9196    0.7375
Fuzzers       0.3287  0.6320    0.4325          0.3185  0.6049    0.4173          0.3239  0.6245    0.4266
DoS           0.6757  0.0851    0.1512          0.6067  0.0890    0.1553          0.6634  0.0819    0.1458
```

---

## 3. Master Consolidated Benchmark Table (9 Configurations)

All open-set metrics evaluated on the full, untouched official UNSW-NB15 test set ($N = 82,332$: 77,831 known flows, 4,501 unknown flows across Reconnaissance, Backdoor, Shellcode, Worms). Calibrated on $\mathcal{D}_{\text{thresh\_cal}}$ ($N = 12,139$).

| Model | Method | AUROC | AUPR-Out | AUPR-In | OSCR | $\tau_{\text{FPR05}}$ Norm FPR | $\tau_{\text{FPR05}}$ Known Cov | $\tau_{\text{FPR05}}$ Unk Det | $\tau_{\text{Cov95}}$ Norm FPR | $\tau_{\text{Cov95}}$ Known Cov | $\tau_{\text{Cov95}}$ Unk Det | $\tau_{\text{Cov95}}$ OpenSet Acc | $\tau_{\text{Cov95}}$ Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **MSP** | 0.8601 | 0.3552 | 0.9905 | 0.7177 | $12.58\%$ | $83.19\%$ | $62.30\%$ | **$2.16\%$** | **$95.66\%$** | $45.72\%$ | **$75.17\%$** | **$50.49\%$** |
| **LightGBM** | **Energy** | 0.8712 | 0.2575 | 0.9913 | 0.7191 | $14.28\%$ | $80.20\%$ | $77.96\%$ | $8.13\%$ | $89.65\%$ | $61.61\%$ | $72.86\%$ | $48.25\%$ |
| **LightGBM** | **OpenMax** | **0.8817** | 0.2717 | **0.9922** | **0.7299** | $13.09\%$ | $79.15\%$ | $81.67\%$ | $4.63\%$ | $91.26\%$ | $57.92\%$ | $73.98\%$ | $50.29\%$ |
| **RandomForest** | **MSP** | 0.8523 | 0.2988 | 0.9900 | 0.7158 | $12.87\%$ | $81.55\%$ | $64.59\%$ | $2.98\%$ | $91.20\%$ | $52.37\%$ | $74.02\%$ | $48.63\%$ |
| **RandomForest** | **Energy** | 0.8699 | 0.2294 | 0.9916 | 0.7184 | $15.65\%$ | $71.76\%$ | $90.87\%$ | $7.04\%$ | $90.19\%$ | $55.61\%$ | $72.47\%$ | $47.88\%$ |
| **RandomForest** | **OpenMax** | 0.8050 | 0.1184 | 0.9877 | 0.6621 | $16.72\%$ | $65.70\%$ | **$91.05\%$** | $9.72\%$ | $79.46\%$ | **$65.45\%$** | $65.66\%$ | $43.34\%$ |
| **XGBoost** | **MSP** | 0.8546 | **0.3591** | 0.9902 | 0.7153 | $11.36\%$ | $85.13\%$ | $56.34\%$ | **$2.13\%$** | $95.22\%$ | $42.46\%$ | $74.69\%$ | $49.23\%$ |
| **XGBoost** | **Energy** | 0.8327 | 0.1760 | 0.9888 | 0.7000 | $10.77\%$ | $87.25\%$ | $54.37\%$ | $9.41\%$ | $88.56\%$ | $50.52\%$ | $71.66\%$ | $46.38\%$ |
| **XGBoost** | **OpenMax** | 0.8614 | 0.1833 | 0.9910 | 0.7200 | $13.86\%$ | $77.20\%$ | $85.45\%$ | $9.08\%$ | $86.71\%$ | $61.50\%$ | $72.00\%$ | $46.42\%$ |

*Persisted Artifact Reference:* [results/openset/synthesis_3models/consolidated_benchmark_table.csv](file:///D:/Network%20IDS/results/openset/synthesis_3models/consolidated_benchmark_table.csv).

---

## 4. Per-Category Unknown Detection Synthesis

Zero-day detection recall across the 4,501 unknown flows at the operational threshold $\tau_{\text{Cov95}}$:

| Attack Category | Test Support | LGB MSP | LGB Energy | LGB OpenMax | RF MSP | RF Energy | RF OpenMax | XGB MSP | XGB Energy | XGB OpenMax |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reconnaissance** | 3,496 | $48.23\%$ | $65.22\%$ | $61.18\%$ | $48.48\%$ | $52.52\%$ | $64.30\%$ | $45.65\%$ | $48.40\%$ | **$61.58\%$** |
| **Backdoor** | 583 | $27.10\%$ | $45.45\%$ | $34.99\%$ | **$80.96\%$** | $77.36\%$ | $80.62\%$ | $25.56\%$ | $74.96\%$ | $59.01\%$ |
| **Shellcode** | 378 | $55.82\%$ | $56.88\%$ | $67.20\%$ | $49.21\%$ | $55.82\%$ | $57.14\%$ | $42.33\%$ | $34.92\%$ | **$67.99\%$** |
| **Worms** | 44 | $6.82\%$ | $29.55\%$ | $22.73\%$ | $9.09\%$ | $11.36\%$ | $27.27\%$ | $13.64\%$ | $29.55\%$ | **$31.82\%$** |
| **Overall Unknowns** | **4,501** | **$45.72\%$** | **$61.61\%$** | **$57.92\%$** | **$52.37\%$** | **$55.61\%$** | **$65.45\%$** | **$42.46\%$** | **$50.52\%$** | **$61.50\%$** |

*Persisted Artifact Reference:* [results/openset/synthesis_3models/per_category_unknown_comparison.csv](file:///D:/Network%20IDS/results/openset/synthesis_3models/per_category_unknown_comparison.csv).

### Cross-Model Attack Insights:
1. **Backdoor Detection Anomaly:** Both Random Forest ($77\% - 81\%$) and XGBoost Energy ($74.96\%$) detect Backdoors substantially better than LightGBM ($27\% - 45\%$). Bagged tree randomization and depth-constrained splits expand the uncertainty volume around stealthy, low-packet interactive sessions.
2. **Shellcode Exploits:** OpenMax is the superior detector for Shellcode payloads across all gradient boosted architectures ($67.20\%$ in LightGBM, $67.99\%$ in XGBoost), proving that EVT distance attenuation is highly effective against high-byte-rate buffer overflow patterns.
3. **Worms (Ultra-Rare):** With only 44 test instances, baseline MSP fails ($6.8\% - 13.6\%$). XGBoost OpenMax achieved the single highest Worm detection rate across all nine evaluations (**$31.82\%$**, 14/44 flows detected).

---

## 5. Architectural Deep-Dive: GBDT vs. Bagging in Open-Set Modeling

The addition of XGBoost resolves the architectural puzzle identified during the Random Forest audit:

```
        LightGBM (Leaf-Wise GBDT)           XGBoost (Depth-Wise GBDT)          Random Forest (Bagging)
Logits: Raw Additive Leaf Margins           Raw Additive Leaf Margins          Centered Log-Ratio (CLR)
EVT:    Smooth Continuous Tail              Smooth Continuous Tail             Discrete Probability Spikes
OpenMax: AUROC = 0.8817 (Top Rank)          AUROC = 0.8614 (Healthy)           AUROC = 0.8050 (Collapsed)
```

### 1. Proof of the Continuous Margin Hypothesis
In Step 6, Random Forest OpenMax failed because probability estimates $p_c(x)$ are discrete rational fractions ($m / 100$). For saturated classes like Generic, $70\%$ of EVT tail samples collapsed onto identical coordinates, causing maximum likelihood estimation to yield a degenerate Weibull fit ($k = 0.0294, \lambda \approx 10^{-23}$) and triggering mass false rejections (5,397 Generic flows rejected).

XGBoost completely vindicates this analysis:
- XGBoost extracts raw margins via `output_margin=True` (sums of continuous leaf weights).
- Even for Generic ($n=2,938$ correct EVT samples), the Euclidean distances to the MAV possess smooth continuous variance:
  $$\text{Generic Weibull: } \text{shape } k = 1.5169, \quad \text{scale } \lambda = 4.0058, \quad \text{fallback } = \text{None}$$
- OpenMax achieved an AUROC of **$0.8614$** and OSCR of **$0.7200$**, with known coverage at $\tau_{\text{Cov95}}$ remaining stable at **$86.71\%$** (versus RF's collapsed $79.46\%$).

### 2. Methodological Observation: Analysis Tail Sizing in XGBoost ($\eta = 5$)
In XGBoost OpenMax, Class 1 (Analysis) utilized a tail size of $\eta = 5$, whereas all other classes utilized $\eta = 20$:
- In $\mathcal{D}_{\text{evt\_fit}}$, Analysis is an extreme minority class ($150$ total flows).
- Due to low closed-set recall ($8.4\%$), only $n_1 = 27$ Analysis flows were classified correctly.
- The tail sizing formula $\eta = \min(20, \max(5, \lfloor 0.2 \cdot n_c \rfloor))$ evaluates to $\min(20, \max(5, 5)) = 5$.
- Because $n_1 = 27 \ge 5$, the Weibull fit converged smoothly ($k = 1.0000, \lambda = 1.0500$) without triggering synthetic fallbacks.

### 3. Methodological Observation: XGBoost Energy Threshold Proximity
In XGBoost Energy, the calibrated thresholds on $\mathcal{D}_{\text{thresh\_cal}}$ were:
$$\tau_{\text{FPR05}} = -2.2678, \quad \tau_{\text{Cov95}} = -2.2500 \quad (\Delta = 0.0178)$$
This sharp threshold proximity indicates that XGBoost's depth-wise margin distributions produce a steep cliff in energy score space near the decision boundary. Consequently, operating performance at $\tau_{\text{FPR05}}$ and $\tau_{\text{Cov95}}$ is very similar ($54.37\%$ vs. $50.52\%$ unknown detection, $10.77\%$ vs. $9.41\%$ Normal FPR).

---

## 6. Latency, Throughput, and SOC Operational Realities

Detailed timing decomposition across all three architectures on the official test set ($N = 82,332$):

| Model | Open-Set Scoring Latency | Preprocessing Latency | End-to-End Latency | Maximum Throughput |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM** | $0.0374\text{ ms/sample}$ | $\sim 0.80 - 1.05\text{ ms}$ | $\sim 0.84 - 1.09\text{ ms}$ | $\sim 920 - 1,190\text{ flows/sec}$ |
| **Random Forest** | **$0.0075 - 0.0193\text{ ms/sample}$** | $\sim 0.80 - 1.06\text{ ms}$ | $\sim 0.81 - 1.08\text{ ms}$ | $\sim 925 - 1,230\text{ flows/sec}$ |
| **XGBoost** | **$0.0089 - 0.0129\text{ ms/sample}$** | $\sim 1.14 - 1.20\text{ ms}$ | $\sim 1.15 - 1.21\text{ ms}$ | $\sim 825 - 870\text{ flows/sec}$ |

### Operational SOC Conclusions:
1. **The Preprocessing Bottleneck:** Tabular transformations (Pandas, One-Hot Encoding, MinMax scaling) account for **$>98\%$ of total pipeline latency**. Model scoring is extremely fast across all three engines ($< 0.04\text{ ms}$).
2. **False Alarm Burden at Scale:**
   In an enterprise environment processing $1,000,000$ daily normal flows:
   - **XGBoost MSP at $\tau_{\text{Cov95}}$ ($2.13\%$ FPR):** Generates $\sim 21,300$ daily false alarms while catching $42.5\%$ of zero-days.
   - **LightGBM OpenMax at $\tau_{\text{Cov95}}$ ($4.63\%$ FPR):** Generates $\sim 46,300$ daily false alarms while catching $57.9\%$ of zero-days.
   - **XGBoost OpenMax at $\tau_{\text{Cov95}}$ ($9.08\%$ FPR):** Generates $\sim 90,800$ daily false alarms while catching $61.5\%$ of zero-days.
   - **RF OpenMax at $\tau_{\text{Cov95}}$ ($9.72\%$ FPR):** Generates $\sim 97,200$ daily false alarms while rejecting $20.5\%$ of legitimate known traffic.

---

## 7. Limitations & Threats to Validity

1. **Benchmark Grounding:** Evaluated strictly on UNSW-NB15 (48 actual features). Cross-dataset generalization (e.g., CIC-IDS2017) remains subject to domain shift.
2. **Margin Scaling Disparities:** Depth-wise boosting (XGBoost) and leaf-wise boosting (LightGBM) generate different margin dynamic ranges, impacting energy thresholds and temperature scaling.
3. **Static Threshold Calibration:** All thresholds were statically calibrated on $\mathcal{D}_{\text{thresh\_cal}}$. Real-world production networks experience non-stationary traffic concept drift.

---

## 8. Final Recommendations: Optimal Model-Method Pairings

### Top Recommended Pairings:
1. **Primary Recommendation: LightGBM + OpenMax**
   - **Score:** $\text{AUROC} = 0.8817$, $\text{OSCR} = 0.7299$.
   - **Operational:** $91.26\%$ known coverage, $4.63\%$ Normal FPR, $57.92\%$ zero-day detection at $\tau_{\text{Cov95}}$.
   - **Why:** Best overall balance of closed-set classification, high open-set discrimination, and low false alarms.

2. **Secondary Recommendation: XGBoost + OpenMax**
   - **Score:** $\text{AUROC} = 0.8614$, $\text{OSCR} = 0.7200$.
   - **Operational:** $86.71\%$ known coverage, $9.08\%$ Normal FPR, $61.50\%$ zero-day detection at $\tau_{\text{Cov95}}$.
   - **Why:** Strongest single-model catch rate for Worms ($31.8\%$) and Shellcode ($68.0\%$) with healthy continuous Weibull tails.

3. **Conservative High-Throughput / Low-Alarm Pairing: XGBoost / LightGBM + MSP**
   - **Operational:** $>95.2\%$ known coverage, $\sim 2.1\%$ Normal FPR, $42.5\% - 45.7\%$ zero-day detection at $\tau_{\text{Cov95}}$.
   - **Why:** Essential when SOC analyst capacity is severely constrained and false positives must be minimized above all else.

4. **Bagging Alternative: Random Forest + Energy**
   - **Score:** $\text{AUROC} = 0.8699$, $\text{OSCR} = 0.7184$.
   - **Operational:** $90.19\%$ known coverage, $7.04\%$ Normal FPR, $55.61\%$ zero-day detection at $\tau_{\text{Cov95}}$.
   - **Why:** Bypasses discrete probability collapse in bagging ensembles while maintaining solid zero-day sensitivity.
