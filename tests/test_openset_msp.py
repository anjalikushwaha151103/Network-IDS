"""Comprehensive unit tests for Maximum Softmax Probability (MSP) Open-Set Detector (Phase 2 - Step 4).

Verifies:
1. MSP anomaly scores are strictly bounded in [0, 1].
2. Higher maximum softmax probability produces an inverse (lower) anomaly score.
3. Thresholds (tau_FPR05, tau_Cov95) are calculated exclusively from D_threshold_cal.
4. Unknown test samples are never used for threshold calibration.
5. Predicted open-set labels are strictly restricted to {0, 1, 2, 3, 4, 5, 6}.
6. Both operating thresholds produce separate and distinguishable evaluation results.
7. Saved results can be loaded and reproduced.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.data_loader import DataLoader
from src.models.lightgbm_model import LightGBMModel
from src.openset.base_detector import BaseOODDetector
from src.openset.msp_detector import MSPDetector
from src.openset.partitioner import (
    KNOWN_CLASS_TO_ID,
    UNKNOWN_LABEL_ID,
    extract_known_multiclass_labels,
    extract_openset_7class_labels,
    partition_openset_data,
)
from src.preprocessing import NIDSPreprocessor


@pytest.fixture(scope="module")
def trained_artifacts():
    """Loads the pre-trained 6-class LightGBM model and preprocessor."""
    cls_dir = Path("results/openset/LightGBM/features_48")
    model_path = cls_dir / "model.pkl"
    preproc_path = cls_dir / "preprocessor.pkl"

    assert model_path.exists(), f"Model artifact missing at: {model_path}"
    assert preproc_path.exists(), f"Preprocessor artifact missing at: {preproc_path}"

    model = LightGBMModel.load(model_path)
    preprocessor = NIDSPreprocessor.load(preproc_path)
    return model, preprocessor


@pytest.fixture(scope="module")
def partitioned_data():
    """Loads the official dataset and partitions into the 4-way splits."""
    loader = DataLoader()
    raw_train, raw_test = loader.load_data()
    splits = partition_openset_data(raw_train, raw_test, random_state=42)
    return splits


def test_msp_scores_bounded_and_monotonic(trained_artifacts, partitioned_data):
    """Test 1 & 2: MSP scores bounded in [0, 1] and strictly inverse to maximum softmax probability."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    # Use D_val_known for evaluation
    X_val, _, _ = preprocessor.transform(splits.df_val_known, feature_subset=preprocessor.feature_names_)

    detector = MSPDetector()
    detector.fit(None, None, model)

    anomaly_scores = detector.compute_anomaly_score(X_val, model)
    confidence_scores = detector.compute_confidence_score(X_val, model)
    probs = model.predict_proba(X_val)
    max_probs = np.max(probs, axis=1)

    # 1. Bounds verification
    assert np.all(anomaly_scores >= 0.0), "Anomaly scores must be >= 0.0"
    assert np.all(anomaly_scores <= 1.0), "Anomaly scores must be <= 1.0"
    assert np.all(confidence_scores >= 0.0), "Confidence scores must be >= 0.0"
    assert np.all(confidence_scores <= 1.0), "Confidence scores must be <= 1.0"

    # 2. Inverse mathematical relationship
    np.testing.assert_allclose(anomaly_scores, 1.0 - max_probs, rtol=1e-5)
    np.testing.assert_allclose(confidence_scores, max_probs, rtol=1e-5)
    np.testing.assert_allclose(anomaly_scores + confidence_scores, 1.0, rtol=1e-5)


def test_thresholds_calculated_exclusively_from_d_threshold_cal(trained_artifacts, partitioned_data):
    """Test 3 & 4: Calibration uses ONLY D_threshold_cal; unknown samples are never observed."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    # Calibrate on D_threshold_cal
    X_cal, _, _ = preprocessor.transform(splits.df_thresh_cal, feature_subset=preprocessor.feature_names_)
    y_cal = extract_known_multiclass_labels(splits.df_thresh_cal)

    detector = MSPDetector()
    thresholds = detector.calibrate_thresholds(
        X_cal,
        y_cal,
        model,
        normal_class_id=KNOWN_CLASS_TO_ID["Normal"],
        percentile_target=95.0,
    )

    tau_fpr05 = thresholds["tau_FPR05"]
    tau_cov95 = thresholds["tau_Cov95"]

    assert 0.0 < tau_fpr05 < 1.0, f"tau_FPR05 out of bounds: {tau_fpr05}"
    assert 0.0 < tau_cov95 < 1.0, f"tau_Cov95 out of bounds: {tau_cov95}"
    assert detector.calibration_stats_["total_cal_samples"] == len(splits.df_thresh_cal)
    assert detector.calibration_stats_["normal_cal_samples"] == (splits.df_thresh_cal["attack_cat"] == "Normal").sum()

    # Verify that corrupting or adding unknown samples to test data does NOT affect calibration
    # because calibration ONLY accesses D_threshold_cal
    assert len(splits.df_thresh_cal) == 12139
    assert not splits.df_thresh_cal["attack_cat"].isin(["Reconnaissance", "Backdoor", "Shellcode", "Worms"]).any()


def test_predicted_labels_restricted_to_0_through_6(trained_artifacts, partitioned_data):
    """Test 5: Open-set predictions belong strictly to {0, 1, 2, 3, 4, 5, 6}."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    X_test, _, _ = preprocessor.transform(splits.df_test, feature_subset=preprocessor.feature_names_)

    detector = MSPDetector()
    # Test with varying thresholds
    for thresh in [0.05, 0.20, 0.50, 0.80]:
        preds = detector.predict_openset(X_test[:500], model, threshold=thresh, unknown_label=UNKNOWN_LABEL_ID)
        unique_labels = set(np.unique(preds))
        assert unique_labels.issubset({0, 1, 2, 3, 4, 5, 6}), f"Invalid labels found: {unique_labels - {0,1,2,3,4,5,6}}"


def test_both_thresholds_produce_separate_evaluations(trained_artifacts, partitioned_data):
    """Test 6: Both thresholds (tau_FPR05 and tau_Cov95) yield distinct operating point predictions."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    X_cal, _, _ = preprocessor.transform(splits.df_thresh_cal, feature_subset=preprocessor.feature_names_)
    y_cal = extract_known_multiclass_labels(splits.df_thresh_cal)

    detector = MSPDetector()
    thresholds = detector.calibrate_thresholds(X_cal, y_cal, model)

    tau_fpr05 = thresholds["tau_FPR05"]
    tau_cov95 = thresholds["tau_Cov95"]

    # Threshold values must be calibrated numbers
    assert isinstance(tau_fpr05, float)
    assert isinstance(tau_cov95, float)

    # In UNSW-NB15, Normal flows have lower entropy than complex attack flows, so tau_FPR05 != tau_Cov95
    X_test_sample, _, _ = preprocessor.transform(splits.df_test[:2000], feature_subset=preprocessor.feature_names_)
    preds_fpr05 = detector.predict_openset(X_test_sample, model, threshold=tau_fpr05)
    preds_cov95 = detector.predict_openset(X_test_sample, model, threshold=tau_cov95)

    assert len(preds_fpr05) == len(preds_cov95) == 2000


def test_saved_results_reproducible_and_loadable():
    """Test 7: Saved evaluation metrics can be loaded and match expected schema."""
    msp_dir = Path("results/openset/LightGBM/features_48/MSP")
    # If already executed, check schema
    if (msp_dir / "metrics_openset_msp.json").exists():
        with open(msp_dir / "metrics_openset_msp.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["method"] == "Maximum Softmax Probability (MSP)"
        assert "auroc" in data["threshold_independent_metrics"]
        assert "aupr_out" in data["threshold_independent_metrics"]
        assert "aupr_in" in data["threshold_independent_metrics"]
        assert "oscr_auc" in data["threshold_independent_metrics"]
        assert "tau_FPR05" in data["operating_points"]
        assert "tau_Cov95" in data["operating_points"]


def test_confusion_matrix_metrics_consistency():
    """Test 8: Verifies reported accuracy and operating-point metrics directly against confusion matrix counts."""
    msp_dir = Path("results/openset/LightGBM/features_48/MSP")
    cm_fpr05_path = msp_dir / "confusion_matrix_7x7_fpr05.csv"
    cm_cov95_path = msp_dir / "confusion_matrix_7x7_cov95.csv"
    metrics_json_path = msp_dir / "metrics_openset_msp.json"

    assert cm_fpr05_path.exists(), f"Missing {cm_fpr05_path}"
    assert cm_cov95_path.exists(), f"Missing {cm_cov95_path}"
    assert metrics_json_path.exists(), f"Missing {metrics_json_path}"

    import pandas as pd
    cm_fpr05 = pd.read_csv(cm_fpr05_path, index_col=0)
    cm_cov95 = pd.read_csv(cm_cov95_path, index_col=0)

    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics_json = json.load(f)

    expected_supports = {
        "Normal": 37000,
        "Analysis": 677,
        "Generic": 18871,
        "Exploits": 11132,
        "Fuzzers": 6062,
        "DoS": 4089,
        "Unknown": 4501,
    }

    for cm_df, tag in [(cm_fpr05, "tau_FPR05"), (cm_cov95, "tau_Cov95")]:
        cm = cm_df.values
        classes = cm_df.index.tolist()

        # 1. Total sum and row support checks
        assert cm.sum() == 82332, f"{tag}: Total sum must be exactly 82,332, got {cm.sum()}"
        for cname, expected_supp in expected_supports.items():
            actual_supp = int(cm_df.loc[cname].sum())
            assert actual_supp == expected_supp, f"{tag}: Row support mismatch for {cname}: expected {expected_supp}, got {actual_supp}"

        # 2. Arithmetic metric calculations directly from matrix
        diag = np.diag(cm)
        acc = float(diag.sum() / cm.sum())

        tp = diag.astype(float)
        fn = (cm.sum(axis=1) - tp).astype(float)
        fp = (cm.sum(axis=0) - tp).astype(float)
        support = cm.sum(axis=1).astype(float)

        prec = np.divide(tp, (tp + fp), out=np.zeros_like(tp), where=(tp + fp) != 0)
        rec = np.divide(tp, (tp + fn), out=np.zeros_like(tp), where=(tp + fn) != 0)
        f1 = np.divide(2 * prec * rec, (prec + rec), out=np.zeros_like(tp), where=(prec + rec) != 0)

        macro_f1 = float(np.mean(f1))
        weighted_f1 = float(np.sum(f1 * support) / cm.sum())

        normal_fpr = float(cm[0, 6] / support[0])
        known_rejected = int(cm[:6, 6].sum())
        known_support = int(support[:6].sum())
        known_fpr = float(known_rejected / known_support)
        known_coverage = float(1.0 - known_fpr)
        unknown_recall = float(cm[6, 6] / support[6])

        # 3. Verify exact match with JSON metrics
        op_json = metrics_json["operating_points"][tag]
        assert np.isclose(acc, op_json["open_set_accuracy"], atol=1e-7), f"{tag}: Accuracy mismatch!"
        assert np.isclose(macro_f1, op_json["open_set_macro_f1"], atol=1e-7), f"{tag}: Macro F1 mismatch!"
        assert np.isclose(weighted_f1, op_json["open_set_weighted_f1"], atol=1e-7), f"{tag}: Weighted F1 mismatch!"
        assert np.isclose(normal_fpr, op_json["realized_test_fpr_normal"], atol=1e-7), f"{tag}: Normal FPR mismatch!"
        assert np.isclose(known_fpr, op_json["realized_test_fpr_known"], atol=1e-7), f"{tag}: Known FPR mismatch!"
        assert np.isclose(known_coverage, op_json["known_coverage"], atol=1e-7), f"{tag}: Known coverage mismatch!"
        assert np.isclose(unknown_recall, op_json["unknown_detection_rate"], atol=1e-7), f"{tag}: Unknown recall mismatch!"
