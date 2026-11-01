# Final UNSW-NB15 Dataset & Preprocessing Verification Report

**Date of Verification**: 2026-11-01  
**Project Workspace**: `D:\Network IDS`  
**Evaluation Scope**: Verification of Feature Accounting, Duplicates/Conflicts, Random Forest CLR Scores, Class Distribution Metrics, and Experiment Artifacts.  

---

## 1. Resolution of the 49-Feature Discrepancy

### 1.1 Complete Inventory of the 48 Model Input Features
The pipeline extracts exactly **48 features** entering the machine learning models:

| # | Feature Name | Source | Description |
| :-: | :--- | :--- | :--- |
| **1** | `proto` | Raw Categorical | Transaction protocol (tcp, udp, arp, ospf, etc.) |
| **2** | `service` | Raw Categorical | Application service (http, ftp, smtp, ssh, dns, etc.) |
| **3** | `state` | Raw Categorical | Connection state (FIN, CON, INT, REQ, etc.) |
| **4** | `ackdat` | Raw Numerical | TCP setup SYN-ACK to ACK time (s) |
| **5** | `avg_pkt_size` | Engineered | Interaction: `total_bytes / (total_pkts + 1.0)` |
| **6** | `byte_ratio` | Engineered | Interaction: `sbytes / (dbytes + 1.0)` |
| **7** | `ct_dst_ltm` | Raw Numerical | Connections to destination IP in 100-flow window |
| **8** | `ct_dst_sport_ltm` | Raw Numerical | Connections to destination from source port |
| **9** | `ct_dst_src_ltm` | Raw Numerical | Connections between same source & destination |
| **10** | `ct_flw_http_mthd` | Raw Numerical | Count of HTTP methods (GET/POST) in flow |
| **11** | `ct_ftp_cmd` | Raw Numerical | Count of FTP commands in session |
| **12** | `ct_src_dport_ltm` | Raw Numerical | Connections from source to destination port |
| **13** | `ct_src_ltm` | Raw Numerical | Connections from source IP in 100-flow window |
| **14** | `ct_srv_dst` | Raw Numerical | Connections to service and destination IP |
| **15** | `ct_srv_src` | Raw Numerical | Connections to service from source IP |
| **16** | `ct_state_ttl` | Raw Numerical | Connections with same state & TTL range |
| **17** | `dbytes` | Raw Numerical | Destination to source transaction bytes |
| **18** | `dinpkt` | Raw Numerical | Destination interpacket arrival time (ms) |
| **19** | `djit` | Raw Numerical | Destination jitter (ms) |
| **20** | `dload` | Raw Numerical | Destination bits per second |
| **21** | `dloss` | Raw Numerical | Destination dropped/retransmitted packets |
| **22** | `dmean` | Raw Numerical | Mean packet size transmitted by destination |
| **23** | `dpkts` | Raw Numerical | Destination to source packet count |
| **24** | `dtcpb` | Raw Numerical | Destination TCP base sequence number |
| **25** | `dttl` | Raw Numerical | Destination to source time to live |
| **26** | `dur` | Raw Numerical | Flow duration (seconds) |
| **27** | `duration_log` | Engineered | Non-linear transform: `log1p(max(0, dur))` |
| **28** | `dwin` | Raw Numerical | Destination TCP window advertisement |
| **29** | `is_ftp_login` | Raw Numerical | Binary flag: 1 if FTP login authenticated |
| **30** | `is_sm_ips_ports` | Raw Numerical | Binary flag: 1 if src & dst IP/port match (LAND) |
| **31** | `pkt_ratio` | Engineered | Interaction: `spkts / (dpkts + 1.0)` |
| **32** | `rate` | Raw Numerical | Packets per second |
| **33** | `response_body_len` | Raw Numerical | HTTP response payload content length |
| **34** | `sbytes` | Raw Numerical | Source to destination transaction bytes |
| **35** | `sinpkt` | Raw Numerical | Source interpacket arrival time (ms) |
| **36** | `sjit` | Raw Numerical | Source jitter (ms) |
| **37** | `sload` | Raw Numerical | Source bits per second |
| **38** | `sloss` | Raw Numerical | Source dropped/retransmitted packets |
| **39** | `smean` | Raw Numerical | Mean packet size transmitted by source |
| **40** | `spkts` | Raw Numerical | Source to destination packet count |
| **41** | `stcpb` | Raw Numerical | Source TCP base sequence number |
| **42** | `sttl` | Raw Numerical | Source to destination time to live |
| **43** | `swin` | Raw Numerical | Source TCP window advertisement |
| **44** | `synack` | Raw Numerical | TCP setup SYN to SYN-ACK time (s) |
| **45** | `tcprtt` | Raw Numerical | TCP round trip time (`synack + ackdat`) |
| **46** | `total_bytes` | Engineered | Interaction: `sbytes + dbytes` |
| **47** | `total_pkts` | Engineered | Interaction: `spkts + dpkts` |
| **48** | `trans_depth` | Raw Numerical | Pipelined HTTP request depth |

### 1.2 Why the Original Specification Refers to "49 Features"
1. **Academic Origin**: The original PCAP flow generator for UNSW-NB15 (Moustafa & Slay, 2015) extracted **49 total columns**:
   - 41 flow attributes
   - 4 network identifiers (`srcip`, `dstip`, `sport`, `dsport`)
   - 2 timestamps (`stime`, `ltime`)
   - 2 ground truth targets (`attack_cat`, `label`)
2. **Official Benchmark Pruning**: When creating `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`, the authors stripped the 6 non-generalizable network identifiers to prevent trivial IP memorization, added an artificial sequential row index `id`, and added `smean`/`dmean`, leaving **45 columns** (1 `id` + 42 flow features + 2 target labels).
3. **Audit Confirmation on Columns**:
   - Only **3 columns** in the raw CSV are not used as model input features:
     - `id`: An artificial sequential row counter (`1, 2, 3... 175341`). Retaining `id` would cause fatal data leakage.
     - `label`: Binary ground truth target (0 = Normal, 1 = Attack).
     - `attack_cat`: Multiclass ground truth target (10 categories).
   - **No legitimate predictive feature was unnecessarily removed**.
4. **Resolution for Tier 49**:
   - In our configuration, tier **49** represents the **Full Available Feature Pool** (all 48 features). When `features=49` is selected, `get_tier_features(49)` takes `min(49, 48) = 48` features cleanly without introducing artificial synthetic noise.

---

## 2. Duplicate Records and Label Conflicts Investigation

### 2.1 Empirical Findings

| Duplicate Category | Training Set (175,341 rows) | Testing Set (82,332 rows) | Description |
| :--- | :---: | :---: | :--- |
| **Full Row Duplicates (with `id`)** | **0 (0.00%)** | **0 (0.00%)** | The sequential `id` column makes every row unique. |
| **All Features + Labels (excluding `id`)** | **67,601 (38.55%)** | **26,387 (32.05%)** | Exact identical network flows with identical ground truth. |
| **Raw Feature-Only Duplicates** | **74,301 (42.38%)** | **28,386 (34.48%)** | Flows with identical features, regardless of label. |
| **Post-Feature-Engineering Duplicates** | **74,301 (42.38%)** | **28,386 (34.48%)** | Adding 6 engineered features leaves the count unchanged (deterministic linear/log combinations). |
| **Cross-Split Shared Flows** | **1,302 flows (1.58% of test set)** | Flows sharing identical feature vectors between train and test. |
| **Conflicting Binary Labels in Train** | **229 feature vectors (940 rows)** | Identical features labeled as both Normal and Attack. |
| **Conflicting Multiclass Labels in Train** | **1,772 feature vectors** | Identical features labeled as different attack categories. |

### 2.2 Breakdown of Conflicting Binary Flows
Of the 940 training rows involved in binary label conflict:
- `label = 0 (Normal)`: **396 rows**
- `label = 1 (Fuzzers)`: **543 rows**
- `label = 1 (Shellcode)`: **1 row**

### 2.3 Scientific Assessment: Benchmark Property vs Bug
- **Conclusion**: This is an **inherent, documented property of the UNSW-NB15 benchmark dataset**, not a pipeline or code error.
- **Root Cause**: The synthetic traffic generation testbed (IXIA PerfectStorm) generated hundreds of automated TCP SYN packets and probe pings with fixed packet sizes (e.g. 44 bytes, `dur=0.000009s`, `sbytes=264`, `swin=255`). Furthermore, Fuzzers generate malformed protocol headers that look identical in basic flow metrics to aborted normal handshakes.
- **Recommendation**: In accordance with published benchmark protocols (Chiba et al., Ring et al.), **do not unilaterally deduplicate or alter the official benchmark splits**. Deduplication would break reproducibility and prevent direct performance comparison with published academic baselines.

---

## 3. Random Forest Score Generation & Centered Log-Ratio (CLR) Validation

### 3.1 Mathematical Distinction Between CLR and Genuine Logits

1. **Neural Networks & Logistic Regression Logits**:
   $$f(x) = XW^T + b \in \mathbb{R}^K$$
   Unbounded real outputs representing unnormalized log-posteriors up to an arbitrary sample-dependent additive constant.
2. **Gradient Boosters (LightGBM / XGBoost) Margins**:
   $$f_c(x) = \sum_{m=1}^M \gamma_{m,c} h_{m,c}(x)$$
   Additive functional tree margins directly optimizing the negative log-likelihood objective in function space.
3. **Random Forest Leaf Probabilities & CLR**:
   Random Forests output empirical leaf-voting posteriors $p_c(x) = \frac{1}{M}\sum_{m=1}^M p_{m,c}(x) \in [0, 1]$ where $\sum_{c=1}^K p_c(x) = 1$.
   The Centered Log-Ratio (Aitchison, 1986) projects probabilities from the simplex $\mathcal{S}^K$ to Euclidean space:
   $$\text{clr}(p)_c = \log\left(\max(p_c, \epsilon)\right) - \frac{1}{K} \sum_{j=1}^K \log\left(\max(p_j, \epsilon)\right)$$
   **Crucial Mathematical Invariant**:
   $$\sum_{c=1}^K \text{clr}(p)_c \equiv 0 \quad (\forall x)$$
   Unlike LightGBM/XGBoost margins, RF CLR scores are strictly zero-sum.

### 3.2 Energy-Based OOD Discrimination Audit

Free Energy is computed via:
$$E(x) = -T \cdot \log \sum_{c=1}^K \exp\left( \frac{f_c(x)}{T} \right)$$

Evaluating on 1,500 test samples partitioned into Known Classes (*Normal, Generic, Exploits, Fuzzers, DoS*) vs Held-Out Unseen Classes (*Reconnaissance, Backdoor, Shellcode, Worms*):

| Metric | LightGBM Raw Margins | Random Forest CLR Scores |
| :--- | :---: | :---: |
| **Output Shape** | $(N, 10)$ | $(N, 10)$ |
| **Dynamic Range** | $[-15.03, +6.27]$ | $[-12.10, +14.51]$ |
| **Known Classes Mean Energy** | **-2.63** ($\pm 2.59$) | **-10.47** ($\pm 3.32$) |
| **Unseen Classes Mean Energy** | **-1.49** ($\pm 1.52$) | **-9.58** ($\pm 3.09$) |
| **Energy Separation ($\Delta E$)** | **+1.14** (higher energy for OOD) | **+0.89** (higher energy for OOD) |

- **Conclusion**: Both LightGBM and Random Forest exhibit the expected physical property: **unseen/OOD classes produce higher free energy (less negative) than known classes**.
- **Sensitivity to $\epsilon$**:
  - $\epsilon = 10^{-4}$: Mean Energy = -6.51
  - $\epsilon = 10^{-6}$: Mean Energy = -9.10
  - $\epsilon = 10^{-7}$ (Active): Mean Energy = -10.39, Std = 3.33
  - $\epsilon = 10^{-10}$: Mean Energy = -14.26
- **Mathematical Limitation**: While CLR preserves the monotonicity needed for OOD ranking, it is an empirical posterior projection rather than an unconstrained likelihood margin. For research purity, LightGBM and MLP remain the gold standard for unconstrained Energy OOD scoring.

---

## 4. Class Distribution & Evaluation Audit

### 4.1 Class Imbalance Across Benchmark Splits

| Category | Train Count | Train % | Test Count | Test % | Shift ($\Delta$) | Threat Level |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Normal** | 56,000 | 31.94% | 37,000 | 44.94% | **+13.00%** | Benign Baseline |
| **Generic** | 40,000 | 22.81% | 18,871 | 22.92% | +0.11% | High-Volume |
| **Exploits** | 33,393 | 19.05% | 11,132 | 13.52% | -5.53% | High-Volume |
| **Fuzzers** | 18,184 | 10.37% | 6,062 | 7.36% | -3.01% | Moderate-Volume |
| **DoS** | 12,264 | 6.99% | 4,089 | 4.97% | -2.02% | High-Impact |
| **Reconnaissance** | 10,491 | 5.98% | 3,496 | 4.25% | -1.73% | Scan/Probe |
| **Analysis** | 2,000 | 1.14% | 677 | 0.82% | -0.32% | Low-Frequency |
| **Backdoor** | 1,746 | 1.00% | 583 | 0.71% | -0.29% | Stealth Infiltration |
| **Shellcode** | 1,133 | 0.65% | 378 | 0.46% | -0.19% | Payload Execution |
| **Worms** | 130 | 0.07% | 44 | 0.05% | -0.02% | Self-Propagating |

### 4.2 Full-Dataset Evaluation Results (175,341 Train / 82,332 Test)

Model: **LightGBM** (30 features, multiclass)

| Metric | Overall Score |
| :--- | :---: |
| **Overall Accuracy** | **75.63%** |
| **Weighted F1-Score** | **76.69%** |
| **Macro F1-Score** | **47.87%** |
| **Macro Precision** | **50.15%** |
| **Macro Recall** | **56.67%** |

#### Per-Class Performance Breakdown:
| Category | Support (Test) | Precision | Recall | F1-Score | Performance Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Generic** | 18,871 | 99.76% | 88.33% | **93.70%** | Near-perfect detection; highly distinctive flow signatures |
| **Reconnaissance** | 3,496 | 91.64% | 80.89% | **85.93%** | High detection rate; port scan metrics are strong drivers |
| **Normal** | 37,000 | 92.66% | 76.31% | **83.69%** | High precision; false alarms predominantly on Fuzzers |
| **Exploits** | 11,132 | 59.41% | 88.51% | **71.10%** | High recall (88.5%); moderate precision |
| **Shellcode** | 378 | 34.77% | 78.84% | **48.26%** | Good recall on small sample; low precision |
| **Fuzzers** | 6,062 | 32.55% | 62.55% | **42.82%** | High confusion with Normal traffic |
| **Worms** | 44 | 9.79% | 52.27% | **16.49%** | Extreme minority class (44 samples in test set) |
| **DoS** | 4,089 | 62.60% | 9.17% | **16.00%** | Low recall; confused with high-rate Normal flows |
| **Analysis** | 677 | 9.14% | 23.34% | **13.14%** | Rare stealth attack; confused with Normal/Exploits |
| **Backdoor** | 583 | 9.20% | 6.52% | **7.63%** | Lowest detection; mimics standard HTTP/DNS traffic |

### 4.3 Why Accuracy Alone Is Insufficient (The Accuracy Paradox)
- **The Paradox**: The model achieves an overall **Accuracy of 75.63%**, which appears respectable.
- **The Reality**: The model misses **90.8% of DoS attacks**, **93.5% of Backdoors**, and **76.7% of Analysis probes**.
- **Why**: The top 3 classes (Normal, Generic, Exploits) account for **81.4% of all test flows**. A naive model that only classifies these three classes will achieve >75% accuracy while failing completely against critical advanced persistent threats.
- **Conclusion**: **Macro F1 (47.87%)** and **Per-Class Recall** must be prioritized over raw accuracy in all subsequent research novelties.

---

## 5. Experiment Artifacts Integrity Audit

Inspected all directories in `results/` to confirm artifact integrity:

| Directory | Dataset Scope | Requested Tier | Actual Features | Accuracy | Weighted F1 | Fit Time | Artifacts Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `results/LightGBM/features_15` | Sampled (10k tr / 5k ts) | 15 | 15 | 77.50% | 0.7803 | 6.82s | CM, ROC, PR, FI, ReportCSV |
| `results/LightGBM/features_20` | Sampled (10k tr / 5k ts) | 20 | 20 | 77.74% | 0.7849 | 5.88s | CM, ROC, PR, FI, ReportCSV |
| `results/LightGBM/features_30` | **FULL DATASET** (175k tr / 82k ts) | 30 | 30 | 75.63% | 0.7669 | 18.36s | CM, ROC, PR, FI, ReportCSV |
| `results/LightGBM/features_49` | Sampled (10k tr / 5k ts) | 49 | 48 | 76.88% | 0.7797 | 3.87s | CM, ROC, PR, FI, ReportCSV |

- **Comparison Report Check**: `results/comparisons/model_comparison.csv` and `final_report.md` match the underlying `metrics.json` records.
- **Timing Metrics Check**: `fit_time_seconds` accurately logs training duration (18.36s for full 175k rows), and `eval_time_seconds` logs test inference (2.57s for 82k rows = **0.0312 ms/sample**).

---

## 6. Synthesis & Final Review

### 6.1 Confirmed Findings
1. The raw dataset files in `data/raw/` are authentic, complete official UNSW-NB15 benchmark splits.
2. The model input space has **48 distinct features** (42 raw flow features + 6 domain engineered features). No legitimate predictive feature was improperly dropped.
3. Feature engineering provides numerical stability without data leakage.
4. Duplicate records (42.38% feature duplicates, 229 binary conflicts) are verified, inherent benchmark testbed characteristics.
5. The pipeline handles full dataset execution (175,341 training flows, 82,332 testing flows) in under 20 seconds.
6. All experiment artifacts (300 DPI plots, CSV reports, metrics JSONs) are saved systematically without collision.

### 6.2 Unresolved Concerns
1. **Severe Imbalance on Stealth Attacks**: Backdoors (583 test samples, F1=7.6%) and DoS (4,089 test samples, F1=16.0%) suffer from heavy class confusion with Normal traffic.
2. **Random Forest OOD Limitation**: Centered Log-Ratio scores provide viable OOD ranking, but sensitivity to $\epsilon$ requires fixed calibration. Gradient Boosters (LightGBM/XGBoost) and MLP provide more principled log-likelihood margins.

### 6.3 Recommended Corrections / Enhancements
1. **Naming Alignment**: In the final paper and dashboard, formally document tier 49 as `"Tier 49 (All 48 Features)"` to preserve academic transparency.
2. **Cost-Sensitive Weighting**: In multiclass training, consider applying class weights (`class_weight='balanced'`) to boost recall on rare attack families (Backdoor, Shellcode, Worms).
3. **Zero-Day Priority**: In Phase 2 Novelty 1, use LightGBM and MLP as the primary benchmark models for Energy OOD detection, utilizing Random Forest CLR as an empirical baseline.

---

### Final Verdict: **READY FOR RESEARCH EXPERIMENTS**
The data foundation, feature space, preprocessing pipeline, and results storage architecture are fully verified, robust, and mathematically sound. We are approved to proceed to **Phase 2: Research Novelties**.
