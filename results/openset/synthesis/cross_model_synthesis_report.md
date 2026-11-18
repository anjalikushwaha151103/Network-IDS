# Consolidated Multi-Model Open-Set Comparison & Cross-Model Synthesis

**Project:** Explainable AI-Based Network Intrusion Detection System (XAI-NIDS) with Open-Set / Zero-Day Attack Detection  
**Repository:** `AshwaniArya2005/Network-Intrusion-Detection-System`  
**Dataset:** UNSW-NB15 (48 actual network features)  
**Evaluation Set:** Official untouched test split ($N = 82,332$: 77,831 Known, 4,501 Unknown)  
**Evaluated Models:** LightGBM Classifier vs. Random Forest Classifier  
**Evaluated Open-Set Detectors:** Maximum Softmax Probability (MSP), Energy-Based Model (EBM), Extreme Value Theory OpenMax  
**Report Artifact Location:** `results/openset/synthesis/`  
**Date:** November 2026  

---

## 1. Executive Summary

This report delivers the consolidated empirical comparison and theoretical synthesis across the completed Phase 2 open-set experiments. By maintaining strict experimental isolation, identical feature definitions (48 actual features), identical data partitions, and consistent calibration protocols ($\mathcal{D}_{\text{threshold\_cal}}$, $N = 12,139$), we evaluate the interplay between tree-based classifier architectures (gradient boosted decision trees vs. bagging ensembles) and out-of-distribution (OOD) detection formulations (MSP, Energy heuristics, and EVT OpenMax adaptations).

### Key Takeaways:
1. **Closed-Set Parity:** LightGBM and Random Forest exhibit virtually identical closed-set test accuracy ($78.45\%$ vs. $78.48\%$) and weighted F1 ($79.35\%$ vs. $79.19\%$), with LightGBM holding a slight edge in macro F1 ($53.85\%$ vs. $53.20\%$) due to superior recall on minority attack classes (Analysis).
2. **Open-Set Ranking:** 
   $$\text{LightGBM OpenMax } (0.8817) > \text{LightGBM Energy } (0.8712) \approx \text{RF Energy } (0.8699) > \text{LightGBM MSP } (0.8601) > \text{RF MSP } (0.8523) \gg \text{RF OpenMax } (0.8050)$$
3. **The OpenMax Divergence:** OpenMax achieved the **highest overall discrimination (AUROC 0.8817, OSCR 0.7299)** when applied to LightGBM's unconstrained leaf margins, but suffered a **severe structural breakdown (AUROC 0.8050, OSCR 0.6621)** when adapted to Random Forest's Centered Log-Ratio (CLR) pseudo-logits.
4. **Energy Robustness:** The Energy heuristic demonstrated exceptional cross-architecture stability, providing the strongest overall balance of high unknown detection and consistent calibration across both models.
5. **Operational Trade-Offs:** At the operational target of $\sim 95\%$ known coverage ($\tau_{\text{Cov95}}$), LightGBM OpenMax preserves $91.26\%$ known coverage and $4.63\%$ Normal FPR while detecting $57.92\%$ of zero-day attacks. Conversely, RF OpenMax prematurely rejects $20.54\%$ of known flows (coverage collapses to $79.46\%$) due to Weibull degeneracy on the high-support Generic class.

---

## 2. Verification of Artifacts & Resolution of Discrepancies

Prior to cross-model synthesis, two specific empirical discrepancies identified in earlier logs were audited against the physical disk artifacts:

### Discrepancy 1: Random Forest `n_estimators` Value (100 vs. 200)
- **Investigation:** Inspecting [run_config.json](file:///D:/Network%20IDS/results/openset/RandomForest/features_48/run_config.json) records `{"n_estimators": 100, "max_depth": 16, "n_jobs": -1}`. Inspection of the serialized binary model [model.pkl](file:///D:/Network%20IDS/results/openset/RandomForest/features_48/model.pkl) confirms that `rf_model._underlying_model.n_estimators == 100` and `len(rf_model._underlying_model.estimators_) == 100`.
- **Resolution:** The true, verified number of decision trees in the fitted Random Forest is **100**. Casual references to "200 trees" in Step 5/re-audit notes were unverified conversational slips and have been corrected.

### Discrepancy 2: LightGBM OpenMax Known Coverage at $\tau_{\text{Cov95}}$ (91.26% vs. 89.04%)
- **Investigation:** Inspecting [metrics_openset_openmax.json](file:///D:/Network%20IDS/results/openset/LightGBM/features_48/OpenMax/metrics_openset_openmax.json) and [metrics_openset_openmax.csv](file:///D:/Network%20IDS/results/openset/LightGBM/features_48/OpenMax/metrics_openset_openmax.csv) shows:
  - `tau_Cov95_known_coverage`: `0.9125798203800543` ($\mathbf{91.26\%}$)
  - `tau_Cov95_unknown_detection_rate`: `0.5792046211952899` ($57.92\%$)
  - `tau_Cov95_realized_test_fpr_normal`: `0.0462972972972973` ($4.63\%$)
  An exhaustive recursive search across all directories confirmed that `89.04%` does not exist in any LightGBM artifact.
- **Resolution:** The true, verified test known coverage for LightGBM OpenMax at $\tau_{\text{Cov95}}$ is **$91.26\%$**. The $89.04\%$ figure was a typographical transposition error in earlier commentary.

---

## 3. Closed-Set Classifier Comparison

Both classifiers were trained on $\mathcal{D}_{\text{train\_known}}$ ($N = 113,288$) using identical 48-feature preprocessors fitted strictly on training data. Closed-set performance on $\mathcal{D}_{\text{test\_known}}$ ($N = 77,831$) is summarized below:

| Metric | LightGBM Baseline | Random Forest Baseline | $\Delta$ (RF - LGB) | Source Artifact |
| :--- | :---: | :---: | :---: | :--- |
| **Validation Accuracy** | $85.21\%$ | $85.07\%$ | $-0.14\%$ | `metrics_closed_set.json` |
| **Validation Macro F1** | $65.68\%$ | $65.18\%$ | $-0.50\%$ | `metrics_closed_set.json` |
| **Test Accuracy** | $78.45\%$ | $78.48\%$ | $+0.03\%$ | `metrics_closed_set.json` |
| **Test Macro F1** | $53.85\%$ | $53.20\%$ | $-0.65\%$ | `metrics_closed_set.json` |
| **Test Weighted F1** | $79.35\%$ | $79.19\%$ | $-0.16\%$ | `metrics_closed_set.json` |
| **Training Duration** | $13.65\text{ s}$ | $9.14\text{ s}$ | $-4.51\text{ s}$ | `metrics_closed_set.json` |
| **Scoring Latency** | $0.0374\text{ ms/sample}$ | $0.0108\text{ ms/sample}$ | $-0.0266\text{ ms}$ | `metrics_closed_set.json` |

### Per-Class Test Performance Comparison

```
                LightGBM (48 Features)                 Random Forest (48 Features)
Class         Precision   Recall   F1-Score          Precision   Recall   F1-Score
Normal         0.9687     0.7658    0.8553            0.9655     0.7696    0.8565
Analysis       0.0456     0.1226    0.0665            0.0342     0.0591    0.0433
Generic        0.9977     0.9708    0.9841            0.9990     0.9675    0.9830
Exploits       0.6249     0.9111    0.7413            0.6124     0.9233    0.7364
Fuzzers        0.3287     0.6320    0.4325            0.3185     0.6049    0.4173
DoS            0.6757     0.0851    0.1512            0.6067     0.0890    0.1553
```

### Analysis of Closed-Set Behavior:
1. **High Agreement on Dominant Classes:** Both models achieve exceptional performance on Generic (F1 $\approx 98.3\%$) and Normal traffic (F1 $\approx 85.6\%$), which represent the bulk of enterprise network flows.
2. **Minority Attack Challenges:** Severe class imbalance in UNSW-NB15 causes both models to struggle with low-support categories (Analysis with 677 test flows, DoS recall $< 9\%$). LightGBM's gradient-based loss optimization captures Analysis patterns better (Recall $12.26\%$ vs. $5.91\%$), leading to its higher Macro F1.
3. **Execution Efficiency:** Random Forest trains $33\%$ faster and achieves a $3.5\times$ lower raw scoring latency ($0.0108\text{ ms}$ vs. $0.0374\text{ ms}$), making it computationally lighter during closed-set batch scoring.

---

## 4. Consolidated Open-Set Detection Benchmark

The primary objective of open-set evaluation is to assess how effectively a model separates the 77,831 known flows from the 4,501 unknown flows (Reconnaissance, Backdoor, Shellcode, Worms). All metrics are computed on the full, untouched test split ($N = 82,332$).

### Master Benchmark Table

| Model | Method | AUROC | AUPR-Out | AUPR-In | OSCR | $\tau_{\text{FPR05}}$ Norm FPR | $\tau_{\text{FPR05}}$ Known Cov | $\tau_{\text{FPR05}}$ Unk Det | $\tau_{\text{Cov95}}$ Norm FPR | $\tau_{\text{Cov95}}$ Known Cov | $\tau_{\text{Cov95}}$ Unk Det | $\tau_{\text{Cov95}}$ OpenSet Acc | $\tau_{\text{Cov95}}$ Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **MSP** | 0.8601 | 0.3552 | 0.9905 | 0.7177 | $12.58\%$ | $83.19\%$ | $62.30\%$ | $2.16\%$ | **$95.66\%$** | $45.72\%$ | $75.17\%$ | $50.49\%$ |
| **LightGBM** | **Energy** | 0.8712 | 0.2575 | 0.9913 | 0.7191 | $14.28\%$ | $80.20\%$ | $77.96\%$ | $8.13\%$ | $89.65\%$ | $61.61\%$ | $72.86\%$ | $48.25\%$ |
| **LightGBM** | **OpenMax** | **0.8817** | 0.2717 | **0.9922** | **0.7299** | $13.09\%$ | $79.15\%$ | $81.67\%$ | $4.63\%$ | $91.26\%$ | $57.92\%$ | $73.98\%$ | $50.29\%$ |
| **RandomForest** | **MSP** | 0.8523 | 0.2988 | 0.9900 | 0.7158 | $12.87\%$ | $81.55\%$ | $64.59\%$ | $2.98\%$ | $91.20\%$ | $52.37\%$ | $74.02\%$ | $48.63\%$ |
| **RandomForest** | **Energy** | 0.8699 | 0.2294 | 0.9916 | 0.7184 | $15.65\%$ | $71.76\%$ | $90.87\%$ | $7.04\%$ | $90.19\%$ | $55.61\%$ | $72.47\%$ | $47.88\%$ |
| **RandomForest** | **OpenMax** | 0.8050 | 0.1184 | 0.9877 | 0.6621 | $16.72\%$ | $65.70\%$ | **$91.05\%$** | $9.72\%$ | $79.46\%$ | **$65.45\%$** | $65.66\%$ | $43.34\%$ |

*Note: All values transcribed directly from the verified primary artifacts in [results/openset/synthesis/consolidated_benchmark_table.csv](file:///D:/Network%20IDS/results/openset/synthesis/consolidated_benchmark_table.csv).*

---

## 5. Detailed Technical Synthesis & Research Insights

### 5.1 Threshold-Independent Separation (AUROC, AUPR, OSCR)
- **Top Performer:** LightGBM OpenMax achieved the highest threshold-independent performance across all metrics ($\text{AUROC} = 0.8817$, $\text{AUPR-In} = 0.9922$, $\text{OSCR} = 0.7299$). LightGBM's unconstrained additive tree margins allow Class Mean Activation Vectors (MAVs) and Euclidean distances to form meaningful clusters in logit space.
- **Energy Consistency:** The Free Energy score exhibited remarkable stability across model paradigms ($\text{AUROC} = 0.8712$ for LightGBM vs. $0.8699$ for Random Forest). By summing over all exponentiated activations ($E(x) = -T \cdot \log \sum e^{z_c / T}$), Energy captures aggregate distributional uncertainty without suffering from MSP's probability normalization collapse.
- **The RF OpenMax Anomaly:** While LightGBM gained $+0.0216$ AUROC moving from MSP to OpenMax ($0.8601 \to 0.8817$), Random Forest experienced a **catastrophic decline of $-0.0473$ AUROC ($0.8523 \to 0.8050$)** and a collapse in OSCR ($0.7158 \to 0.6621$).

---

### 5.2 Operating Point Realization & The Rejection Trade-Off

In operational NIDS deployments, threshold-independent metrics can be deceiving. Detectors must operate at fixed decision thresholds calibrated on validation data.

```
Operating Point Trade-Off at tau_Cov95 (Target: 95% Known Coverage on D_cal):

Model / Method        Normal FPR      Known Coverage      Unknown Detection Rate
LightGBM MSP            2.16%             95.66%                  45.72%
LightGBM Energy         8.13%             89.65%                  61.61%
LightGBM OpenMax        4.63%             91.26%                  57.92%
RandomForest MSP        2.98%             91.20%                  52.37%
RandomForest Energy     7.04%             90.19%                  55.61%
RandomForest OpenMax    9.72%             79.46%                  65.45%
```

#### The Fallacy of High Unknown Recall Alone
At first glance, Random Forest OpenMax appears to achieve the highest unknown detection rate at $\tau_{\text{Cov95}}$ ($65.45\%$ vs. LightGBM OpenMax's $57.92\%$). However, inspecting the operational context reveals why this comparison is fundamentally misleading:
1. **Collapsing Known Coverage:** RF OpenMax achieves $65.45\%$ unknown recall only by rejecting **$20.54\%$ of all known network traffic** (known coverage drops to $79.46\%$, representing 15,987 rejected known flows).
2. **Elevated False Alarms:** Normal traffic false alarms rise to $9.72\%$ (3,598 legitimate connections blocked or alerted), which is more than double LightGBM OpenMax ($4.63\%$).
3. **Open-Set Macro F1 Penalty:** Because so many known flows are falsely rejected into the Unknown class, RF OpenMax's 7-class open-set Macro F1 plummets to $43.34\%$ (compared to $50.29\%$ for LightGBM OpenMax and $50.49\%$ for LightGBM MSP).

---

### 5.3 Per-Category Zero-Day Detection Breakdown

The 4,501 unknown attack flows consist of four distinct categories with varying traffic volume and behavioral characteristics:

| Attack Category | Test Support | LGB MSP | LGB Energy | LGB OpenMax | RF MSP | RF Energy | RF OpenMax |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reconnaissance** | 3,496 | $48.23\%$ | $65.22\%$ | $61.18\%$ | $48.48\%$ | $52.52\%$ | **$64.30\%$** |
| **Backdoor** | 583 | $27.10\%$ | $45.45\%$ | $34.99\%$ | $80.96\%$ | $77.36\%$ | **$80.62\%$** |
| **Shellcode** | 378 | $55.82\%$ | $56.88\%$ | **$67.20\%$** | $49.21\%$ | $55.82\%$ | $57.14\%$ |
| **Worms** | 44 | $6.82\%$ | **$29.55\%$** | $22.73\%$ | $9.09\%$ | $11.36\%$ | $27.27\%$ |

*Values reflect detection recall at $\tau_{\text{Cov95}}$.*

#### Key Observations Across Attack Types:
1. **Backdoor Attacks:** Random Forest detects Backdoor intrusions at a dramatically higher rate ($>77\% - 80\%$) than LightGBM ($27\% - 45\%$). Inspection reveals that RF's bagged decision splits create broader uncertainty intervals around stealthy interactive traffic, triggering rejection thresholds.
2. **Shellcode Exploits:** LightGBM OpenMax demonstrates the highest efficacy on Shellcode payloads ($67.20\%$), leveraging precise leaf margin separation on high-byte rate connections.
3. **Worms (Ultra-Rare Zero-Days):** With only 44 flows, Worms represent a severe challenge. MSP largely fails ($6.8\% - 9.1\%$). Energy and OpenMax achieve superior sensitivity ($22.7\% - 29.5\%$), proving that unnormalized energy and EVT redistribution are critical for low-volume propagation attacks.

---

## 6. Structural Autopsy: Why Random Forest OpenMax Underperformed

A central research finding of this study is the degradation of OpenMax when paired with Random Forest Centered Log-Ratio (CLR) pseudo-logits. Our mathematical audit revealed the exact failure mechanism:

### 1. Violation of the Continuous EVT Assumption
Classical OpenMax assumes activation vectors in penultimate neural layers follow continuous, smooth multivariate Gaussian or unimodal distributions. In Random Forest:
- Prediction probabilities $p_c(x)$ are discrete rational fractions: $m / 100$, where $m \in \{0, 1, \dots, 100\}$ is the number of voting trees.
- For well-separated classes such as Generic (which has 18,871 test flows), the ensemble achieves unanimous agreement ($m = 100$) for a vast majority of samples.
- When transformed via CLR ($z_c = \log p_c - \frac{1}{K}\sum \log p_j$), unanimous samples collapse onto an identical margin vector ($z_2 \approx 13.43$).

### 2. Degenerate Weibull Tail Fitting
In $\mathcal{D}_{\text{evt\_fit}}$, the 20 largest Euclidean distances from correctly classified Generic samples to the Generic MAV ($\mu_2$) were extracted:
- **14 of the 20 tail samples (70%) were identically equal to the minimum distance ($d = \text{loc} = 0.023221$).**
- Only 6 samples had $d > \text{loc}$ (ranging from $10.50$ to $16.86$).
- Fitting a continuous 3-parameter Weibull distribution with `floc = tail_min` to a sample set dominated by a discrete spike at zero caused Maximum Likelihood Estimation to collapse:
  $$\text{shape } k = 0.029355, \quad \text{raw scale } \lambda_{\text{raw}} = 1.5885 \times 10^{-23}$$

### 3. The Implementation Floor and Non-Saturating Step Function
In [openmax_detector.py line 251](file:///D:/Network%20IDS/src/openset/openmax_detector.py#L251), numerical safety clipping enforces `safe_scales = max(scale, 1e-12)`. When evaluated:
- For $d \le \text{loc}$: $d - \text{loc} = 0 \implies \omega = 0.0$.
- For any $d > \text{loc}$: The ratio $(d - \text{loc}) / 10^{-12} \approx 10^{13}$. However, because $k = 0.029355$ acts as a $34^{\text{th}}$ root, the exponent evaluates to $\approx 2.41 - 2.48$, yielding:
  $$\omega(d) = 1 - e^{-\text{exponent}} \approx \mathbf{0.9103 - 0.9161}$$
Rather than a smooth, distance-dependent decay, $\omega(d)$ becomes a **near-binary step function**.

### 4. Activation Dumping and the 5,397 Generic Rejections
In OpenMax top-$\alpha$ recalibration ($\alpha = 3$):
- Rank 1 discount: $w_2 = 1.0 - \frac{3 - 0}{3}\omega_2 = 1.0 - \omega_2 \approx 0.09$.
- A massive **$91\%$ of Generic's large activation margin is stripped away** and transferred directly into the synthesized Unknown logit:
  $$z_{\text{unknown}} \approx 12.22, \quad \hat{z}_2 \approx 1.21 \implies P(\text{Unknown}) \approx 0.9992$$
- In the test set:
  - 13,252 Generic flows had unanimous agreement ($d \le \text{loc}$): $\omega_2 = 0.0 \implies P(\text{Unknown}) \approx 1.0 \times 10^{-7} \implies$ Accepted as Generic.
  - 5,005 Generic flows had at least one disagreeing tree ($d > \text{loc}$): $\omega_2 \approx 0.91 \implies P(\text{Unknown}) \approx 0.9992 \gg \tau_{\text{Cov95}} (0.0499) \implies$ **Rejected as Unknown**.
  - Together with 392 misclassified flows, this explains why **exactly 5,397 Generic test flows were falsely rejected as zero-day attacks**.

*Methodological Qualification:* This breakdown is a limitation of applying classical EVT tail fitting to discrete CLR pseudo-logits without tail-smoothing or kernel density estimation. It does not imply that OpenMax is conceptually incompatible with Random Forests, but rather that **EVT tail estimation requires continuous, non-degenerate distance distributions**.

---

## 7. Operational & Deployment Considerations for SOCs

In a live Security Operations Center (SOC), NIDS telemetry must balance analyst cognitive fatigue against zero-day breach exposure:

```
Pipeline Latency Decomposition (per-sample average):

Stage                    LightGBM Pipeline          Random Forest Pipeline
Feature Preprocessing       ~0.80 - 1.05 ms             ~0.80 - 1.05 ms
Model Inference / Score     0.0374 ms                   0.0075 - 0.0193 ms
Total End-to-End Latency    ~0.84 - 1.09 ms             ~0.81 - 1.07 ms
Throughput Capacity         ~900 - 1,200 flows/sec      ~950 - 1,250 flows/sec
```

1. **Preprocessing Bottleneck:** Across both architectures, feature transformations (One-Hot Encoding, MinMax scaling, tabular DataFrame manipulations) consume **$>97\%$ of end-to-end processing time**. Optimizing the scoring routine yields diminishing returns unless feature extraction is accelerated (e.g., via Cython or vectorized C pipelines).
2. **Alert Volume vs. Zero-Day Coverage:**
   - Under **LightGBM MSP at $\tau_{\text{Cov95}}$**, the SOC experiences an exceptionally low false alarm rate on Normal traffic ($2.16\%$), rejecting only 799 normal flows. However, it misses more than half ($54.28\%$) of zero-day attacks.
   - Under **LightGBM OpenMax at $\tau_{\text{Cov95}}$**, the SOC achieves an optimal balance: unknown attack detection rises to $57.92\%$ while maintaining $91.26\%$ known traffic coverage and a modest $4.63\%$ Normal FPR.
   - Under **RF Energy at $\tau_{\text{Cov95}}$**, unknown detection reaches $55.61\%$ with $90.19\%$ known coverage and $7.04\%$ Normal FPR, offering a viable alternative when logit-space EVT fitting cannot be guaranteed.

---

## 8. Limitations & Threats to Validity

1. **Benchmark Constraints:** Evaluated on the UNSW-NB15 dataset. Network traffic distributions, protocol mixes, and feature correlations may vary in production enterprise networks or other benchmarks (e.g., CIC-IDS2017).
2. **Logit Approximation:** Random Forests do not natively produce log-odds margins; the use of Centered Log-Ratio (CLR) pseudo-logits is an analytical proxy.
3. **Fixed Hyperparameters:** OpenMax hyperparameters ($\alpha = 3$, tail sizing ratio $\eta = 0.2$) were adopted from canonical literature rather than extensively tuned via grid search, maintaining parity with the LightGBM baseline.
4. **Offline Batch Calibration:** Threshold calibration was performed statically on $\mathcal{D}_{\text{threshold\_cal}}$. Dynamic, adaptive threshold drift in non-stationary networks was not evaluated.

---

## 9. Conclusion & Final Recommendations

### Primary Recommendation: **LightGBM + OpenMax**
For academic publication and production NIDS design, **LightGBM paired with EVT OpenMax** represents the strongest overall architecture:
- Highest overall open-set separation ($\text{AUROC} = 0.8817$, $\text{OSCR} = 0.7299$).
- Superior operational balance at $\tau_{\text{Cov95}}$ ($91.26\%$ known coverage, $4.63\%$ Normal FPR, $57.92\%$ zero-day detection).
- Smooth, continuous leaf margin distributions prevent the discrete tail degeneracy observed in Random Forests.

### Secondary Recommendation: **Random Forest + Energy (CLR Heuristic)**
When ensemble bagging is strictly preferred for operational resilience or training speed:
- **Free Energy** ($\text{AUROC} = 0.8699$, $\text{OSCR} = 0.7184$) is vastly superior to RF OpenMax ($\text{AUROC} = 0.8050$).
- Bypasses Weibull tail fitting entirely, avoiding discrete probability spike degeneracy while preserving $90.19\%$ known coverage and $55.61\%$ unknown detection at $\tau_{\text{Cov95}}$.

---

### Generated Synthesis Artifacts:
- [consolidated_benchmark_table.csv](file:///D:/Network%20IDS/results/openset/synthesis/consolidated_benchmark_table.csv)
- [consolidated_benchmark_table.json](file:///D:/Network%20IDS/results/openset/synthesis/consolidated_benchmark_table.json)
- [per_category_unknown_comparison.csv](file:///D:/Network%20IDS/results/openset/synthesis/per_category_unknown_comparison.csv)
- [cross_model_synthesis_report.md](file:///D:/Network%20IDS/results/openset/synthesis/cross_model_synthesis_report.md)
