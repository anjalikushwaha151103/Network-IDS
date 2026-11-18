"""Inspect and extract all verified numerical metrics from the MLP experiment."""

import json
from pathlib import Path
import pandas as pd
import numpy as np

base_dir = Path("results/openset/MLP/features_48")

print("=================================================================")
print("MLP (MULTI-LAYER PERCEPTRON) RESULTS AUDIT & EXTRACTION")
print("=================================================================")

# 1. Closed-set
with open(base_dir / "metrics_closed_set.json") as f:
    closed = json.load(f)

print("\n--- 1. CLOSED-SET METRICS ---")
print(f"Accuracy:            {closed['accuracy']:.4f} ({closed['accuracy']*100:.2f}%)")
print(f"Macro F1:            {closed['macro_f1']:.4f}")
print(f"Weighted F1:         {closed['weighted_f1']:.4f}")
print(f"Validation Acc:      {closed['validation_accuracy']:.4f}")
print(f"Validation Macro F1: {closed['validation_macro_f1']:.4f}")
print(f"Fit Time:            {closed['training_time_seconds']:.2f}s")
print(f"Inference Latency:   {closed['inference_latency_ms_per_sample']:.4f} ms/sample")
print(f"Total Test Samples:  {closed['total_test_samples']}")

cls_report = pd.read_csv(base_dir / "classification_report.csv", index_col=0)
print("\nClassification Report:")
print(cls_report)

cm_6x6 = pd.read_csv(base_dir / "confusion_matrix_6x6.csv", index_col=0)
print("\nConfusion Matrix (6x6):")
print(cm_6x6)

# 2. MSP
with open(base_dir / "MSP" / "metrics_openset_msp.json") as f:
    msp = json.load(f)

print("\n--- 2. MSP OPEN-SET METRICS ---")
print(f"AUROC:    {msp['threshold_independent_metrics']['auroc']:.4f}")
print(f"AUPR-Out: {msp['threshold_independent_metrics']['aupr_out']:.4f}")
print(f"AUPR-In:  {msp['threshold_independent_metrics']['aupr_in']:.4f}")
print(f"OSCR AUC: {msp['threshold_independent_metrics']['oscr_auc']:.4f}")
print("Timing:")
print(f"  Prep Latency:    {msp['timing']['preprocessing_latency_ms_per_sample']:.4f} ms")
print(f"  Scoring Latency: {msp['timing']['scoring_latency_ms_per_sample']:.4f} ms")
print(f"  End-to-End:      {msp['timing']['end_to_end_latency_ms_per_sample']:.4f} ms")

print("\ntau_FPR05 Operating Point:")
f05_msp = msp['operating_points']['tau_FPR05']
for k in ['threshold_value', 'realized_test_fpr_normal', 'known_coverage', 'unknown_detection_rate', 'open_set_accuracy', 'open_set_macro_f1', 'open_set_weighted_f1']:
    print(f"  {k}: {f05_msp[k]}")

print("\ntau_Cov95 Operating Point:")
c95_msp = msp['operating_points']['tau_Cov95']
for k in ['threshold_value', 'realized_test_fpr_normal', 'known_coverage', 'unknown_detection_rate', 'open_set_accuracy', 'open_set_macro_f1', 'open_set_weighted_f1']:
    print(f"  {k}: {c95_msp[k]}")

msp_cat = pd.read_csv(base_dir / "MSP" / "per_category_unknown_detection.csv")
print("\nMSP Per-Category Unknown Detection:")
print(msp_cat.to_string(index=False))

# 3. Energy
with open(base_dir / "Energy" / "metrics_openset_energy.json") as f:
    energy = json.load(f)

print("\n--- 3. ENERGY OPEN-SET METRICS ---")
print(f"Method Nature:        {energy.get('method_nature')}")
print(f"Score Representation: {energy.get('score_representation')}")
print(f"AUROC:    {energy['threshold_independent_metrics']['auroc']:.4f}")
print(f"AUPR-Out: {energy['threshold_independent_metrics']['aupr_out']:.4f}")
print(f"AUPR-In:  {energy['threshold_independent_metrics']['aupr_in']:.4f}")
print(f"OSCR AUC: {energy['threshold_independent_metrics']['oscr_auc']:.4f}")
print("Timing:")
print(f"  Prep Latency:    {energy['timing']['preprocessing_latency_ms_per_sample']:.4f} ms")
print(f"  Scoring Latency: {energy['timing']['scoring_latency_ms_per_sample']:.4f} ms")
print(f"  End-to-End:      {energy['timing']['end_to_end_latency_ms_per_sample']:.4f} ms")

print("\ntau_FPR05 Operating Point:")
f05_en = energy['operating_points']['tau_FPR05']
for k in ['threshold_value', 'realized_test_fpr_normal', 'known_coverage', 'unknown_detection_rate', 'open_set_accuracy', 'open_set_macro_f1', 'open_set_weighted_f1']:
    print(f"  {k}: {f05_en[k]}")

print("\ntau_Cov95 Operating Point:")
c95_en = energy['operating_points']['tau_Cov95']
for k in ['threshold_value', 'realized_test_fpr_normal', 'known_coverage', 'unknown_detection_rate', 'open_set_accuracy', 'open_set_macro_f1', 'open_set_weighted_f1']:
    print(f"  {k}: {c95_en[k]}")

energy_cat = pd.read_csv(base_dir / "Energy" / "per_category_unknown_detection.csv")
print("\nEnergy Per-Category Unknown Detection:")
print(energy_cat.to_string(index=False))

# 4. OpenMax
with open(base_dir / "OpenMax" / "metrics_openset_openmax.json") as f:
    om = json.load(f)

with open(base_dir / "OpenMax" / "mav_and_weibull_params.json") as f:
    weibull = json.load(f)

print("\n--- 4. OPENMAX OPEN-SET METRICS ---")
print(f"Description:       {om['adaptation_details']['description']}")
print(f"Activation Source: {om['adaptation_details']['activation_source']}")
print(f"AUROC:    {om['threshold_independent_metrics']['auroc']:.4f}")
print(f"AUPR-Out: {om['threshold_independent_metrics']['aupr_out']:.4f}")
print(f"AUPR-In:  {om['threshold_independent_metrics']['aupr_in']:.4f}")
print(f"OSCR AUC: {om['threshold_independent_metrics']['oscr_auc']:.4f}")
print("Timing:")
print(f"  Prep Latency:    {om['timing']['preprocessing_latency_ms_per_sample']:.4f} ms")
print(f"  Scoring Latency: {om['timing']['scoring_latency_ms_per_sample']:.4f} ms")
print(f"  End-to-End:      {om['timing']['end_to_end_latency_ms_per_sample']:.4f} ms")

print("\ntau_FPR05 Operating Point:")
f05_om = om['operating_points']['tau_FPR05']
for k in ['threshold_value', 'realized_test_fpr_normal', 'known_coverage', 'unknown_detection_rate', 'open_set_accuracy', 'open_set_macro_f1', 'open_set_weighted_f1']:
    print(f"  {k}: {f05_om[k]}")

print("\ntau_Cov95 Operating Point:")
c95_om = om['operating_points']['tau_Cov95']
for k in ['threshold_value', 'realized_test_fpr_normal', 'known_coverage', 'unknown_detection_rate', 'open_set_accuracy', 'open_set_macro_f1', 'open_set_weighted_f1']:
    print(f"  {k}: {c95_om[k]}")

print("\nWeibull Parameters per class:")
weibull_df = pd.DataFrame(weibull['weibull_tail_models']).transpose()
print(weibull_df[['class_id', 'n_correct', 'eta', 'shape', 'loc', 'scale', 'fallback_applied', 'fallback_reason']])

om_cat = pd.read_csv(base_dir / "OpenMax" / "per_category_unknown_detection.csv")
print("\nOpenMax Per-Category Unknown Detection:")
print(om_cat.to_string(index=False))

# 5. Confusion matrices 7x7
print("\n--- 5. 7x7 CONFUSION MATRICES ---")
for det in ['MSP', 'Energy', 'OpenMax']:
    print(f"\n{det} 7x7 CM tau_FPR05:")
    print(pd.read_csv(base_dir / det / "confusion_matrix_7x7_fpr05.csv", index_col=0))
    print(f"\n{det} 7x7 CM tau_Cov95:")
    print(pd.read_csv(base_dir / det / "confusion_matrix_7x7_cov95.csv", index_col=0))
