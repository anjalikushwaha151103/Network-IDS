"""Inspect and Extract All Verified Logistic Regression Results (Phase 2)."""

import json
from pathlib import Path
import pandas as pd
import numpy as np

base_dir = Path("results/openset/LogisticRegression/features_48")

print("=================================================================")
print("LOGISTIC REGRESSION RESULTS AUDIT & EXTRACTION")
print("=================================================================")

# 1. Closed-set metrics
with open(base_dir / "metrics_closed_set.json", "r") as f:
    closed_set = json.load(f)

print("\n--- 1. CLOSED-SET CLASSIFICATION ---")
print(f"Accuracy:            {closed_set['accuracy']:.4f} ({closed_set['accuracy']*100:.2f}%)")
print(f"Macro F1:            {closed_set['macro_f1']:.4f}")
print(f"Weighted F1:         {closed_set['weighted_f1']:.4f}")
print(f"Validation Acc:      {closed_set['validation_accuracy']:.4f}")
print(f"Validation Macro F1: {closed_set['validation_macro_f1']:.4f}")
print(f"Training Time:       {closed_set['training_time_seconds']:.2f}s")
print(f"Inference Latency:   {closed_set['inference_latency_ms_per_sample']:.4f} ms/sample")
print(f"Total Test Samples:  {closed_set['total_test_samples']}")

cls_report = pd.read_csv(base_dir / "classification_report.csv", index_col=0)
print("\nClassification Report:")
print(cls_report)

cm_6x6 = pd.read_csv(base_dir / "confusion_matrix_6x6.csv", index_col=0)
print("\nConfusion Matrix (6x6):")
print(cm_6x6)

# 2. MSP
with open(base_dir / "MSP" / "metrics_openset_msp.json", "r") as f:
    msp = json.load(f)

print("\n--- 2. MSP OPEN-SET EVALUATION ---")
print(f"AUROC:    {msp['threshold_independent_metrics']['auroc']:.4f}")
print(f"AUPR-Out: {msp['threshold_independent_metrics']['aupr_out']:.4f}")
print(f"AUPR-In:  {msp['threshold_independent_metrics']['aupr_in']:.4f}")
print(f"OSCR AUC: {msp['threshold_independent_metrics']['oscr_auc']:.4f}")
print("Timing:")
print(f"  Prep Latency:     {msp['timing']['preprocessing_latency_ms_per_sample']:.4f} ms")
print(f"  Scoring Latency:  {msp['timing']['scoring_latency_ms_per_sample']:.4f} ms")
print(f"  End-to-End:       {msp['timing']['end_to_end_latency_ms_per_sample']:.4f} ms")

print("\ntau_FPR05 Operating Point:")
fpr05_msp = msp['operating_points']['tau_FPR05']
for k in ['threshold_value', 'realized_test_fpr_normal', 'known_coverage', 'unknown_detection_rate', 'open_set_accuracy', 'open_set_macro_f1', 'open_set_weighted_f1']:
    print(f"  {k}: {fpr05_msp[k]}")

print("\ntau_Cov95 Operating Point:")
cov95_msp = msp['operating_points']['tau_Cov95']
for k in ['threshold_value', 'realized_test_fpr_normal', 'known_coverage', 'unknown_detection_rate', 'open_set_accuracy', 'open_set_macro_f1', 'open_set_weighted_f1']:
    print(f"  {k}: {cov95_msp[k]}")

msp_cat = pd.read_csv(base_dir / "MSP" / "per_category_unknown_detection.csv")
print("\nMSP Per-Category Unknown Detection:")
print(msp_cat)

# 3. Energy
with open(base_dir / "Energy" / "metrics_openset_energy.json", "r") as f:
    energy = json.load(f)

print("\n--- 3. ENERGY OPEN-SET EVALUATION ---")
print(f"Method Nature:        {energy.get('method_nature')}")
print(f"Score Representation: {energy.get('score_representation')}")
print(f"AUROC:    {energy['threshold_independent_metrics']['auroc']:.4f}")
print(f"AUPR-Out: {energy['threshold_independent_metrics']['aupr_out']:.4f}")
print(f"AUPR-In:  {energy['threshold_independent_metrics']['aupr_in']:.4f}")
print(f"OSCR AUC: {energy['threshold_independent_metrics']['oscr_auc']:.4f}")
print("Timing:")
print(f"  Prep Latency:     {energy['timing']['preprocessing_latency_ms_per_sample']:.4f} ms")
print(f"  Scoring Latency:  {energy['timing']['scoring_latency_ms_per_sample']:.4f} ms")
print(f"  End-to-End:       {energy['timing']['end_to_end_latency_ms_per_sample']:.4f} ms")

print("\ntau_FPR05 Operating Point:")
fpr05_en = energy['operating_points']['tau_FPR05']
for k in ['threshold_value', 'realized_test_fpr_normal', 'known_coverage', 'unknown_detection_rate', 'open_set_accuracy', 'open_set_macro_f1', 'open_set_weighted_f1']:
    print(f"  {k}: {fpr05_en[k]}")

print("\ntau_Cov95 Operating Point:")
cov95_en = energy['operating_points']['tau_Cov95']
for k in ['threshold_value', 'realized_test_fpr_normal', 'known_coverage', 'unknown_detection_rate', 'open_set_accuracy', 'open_set_macro_f1', 'open_set_weighted_f1']:
    print(f"  {k}: {cov95_en[k]}")

energy_cat = pd.read_csv(base_dir / "Energy" / "per_category_unknown_detection.csv")
print("\nEnergy Per-Category Unknown Detection:")
print(energy_cat)

# 4. OpenMax
with open(base_dir / "OpenMax" / "metrics_openset_openmax.json", "r") as f:
    om = json.load(f)

with open(base_dir / "OpenMax" / "mav_and_weibull_params.json", "r") as f:
    weibull = json.load(f)

print("\n--- 4. OPENMAX OPEN-SET EVALUATION ---")
print(f"Description:       {om['adaptation_details']['description']}")
print(f"Activation Source: {om['adaptation_details']['activation_source']}")
print(f"AUROC:    {om['threshold_independent_metrics']['auroc']:.4f}")
print(f"AUPR-Out: {om['threshold_independent_metrics']['aupr_out']:.4f}")
print(f"AUPR-In:  {om['threshold_independent_metrics']['aupr_in']:.4f}")
print(f"OSCR AUC: {om['threshold_independent_metrics']['oscr_auc']:.4f}")
print("Timing:")
print(f"  Prep Latency:     {om['timing']['preprocessing_latency_ms_per_sample']:.4f} ms")
print(f"  Scoring Latency:  {om['timing']['scoring_latency_ms_per_sample']:.4f} ms")
print(f"  End-to-End:       {om['timing']['end_to_end_latency_ms_per_sample']:.4f} ms")

print("\ntau_FPR05 Operating Point:")
fpr05_om = om['operating_points']['tau_FPR05']
for k in ['threshold_value', 'realized_test_fpr_normal', 'known_coverage', 'unknown_detection_rate', 'open_set_accuracy', 'open_set_macro_f1', 'open_set_weighted_f1']:
    print(f"  {k}: {fpr05_om[k]}")

print("\ntau_Cov95 Operating Point:")
cov95_om = om['operating_points']['tau_Cov95']
for k in ['threshold_value', 'realized_test_fpr_normal', 'known_coverage', 'unknown_detection_rate', 'open_set_accuracy', 'open_set_macro_f1', 'open_set_weighted_f1']:
    print(f"  {k}: {cov95_om[k]}")

print("\nWeibull Parameters per class:")
weibull_df = pd.DataFrame(weibull['weibull_tail_models']).transpose()
print(weibull_df[['n_correct', 'eta', 'shape', 'loc', 'scale', 'floc', 'fallback_applied', 'fallback_reason']])

om_cat = pd.read_csv(base_dir / "OpenMax" / "per_category_unknown_detection.csv")
print("\nOpenMax Per-Category Unknown Detection:")
print(om_cat)
