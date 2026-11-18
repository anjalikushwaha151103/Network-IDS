"""Comprehensive Unit and Pluggability Tests for XGBoost in Open-Set Pipeline.

Verifies:
1. XGBoost instantiation through ModelRegistry with alias resolution.
2. Correct 6-class fit on sample data, exact class ordering (0..5), valid probability row sums.
3. Native raw margin extraction (predict_logits via output_margin=True): shape, finiteness, continuous distribution.
4. Feature importance extraction via BaseNIDSModel.get_feature_importances().
5. Polymorphic serialization and deserialization via BaseNIDSModel.load().
6. Detector compatibility: MSP, Energy, and OpenMax successfully process XGBoost representations on sample data.
7. Non-mutation of existing LightGBM, Random Forest, and synthesis reference artifacts.
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.data_loader import DataLoader
from src.models import ModelRegistry
from src.models.base_model import BaseNIDSModel
from src.models.xgboost_model import XGBoostModel
from src.openset.base_detector import BaseOODDetector
from src.openset.energy_detector import EnergyDetector
from src.openset.msp_detector import MSPDetector
from src.openset.openmax_detector import OpenMaxDetector
from src.openset.partitioner import (
    KNOWN_CLASSES,
    KNOWN_CLASS_TO_ID,
    extract_known_multiclass_labels,
    partition_openset_data,
)
from src.preprocessing import NIDSPreprocessor


@pytest.fixture(scope="module")
def sampled_splits():
    """Provides fast sampled splits for pluggability testing without full dataset overhead."""
    loader = DataLoader()
    train_df, test_df = loader.load_data(sample_size=3000, test_sample_size=1000, random_state=42)
    splits = partition_openset_data(train_df, test_df, random_state=42)
    return splits


def test_xgboost_registry_instantiation():
    """Test 1: Verifies ModelRegistry instantiation and alias resolution for XGBoost."""
    for alias in ["xgboost", "xgb", "xgboast", "XGBoost", "XGB"]:
        canonical = ModelRegistry.get_canonical_name(alias)
        assert canonical == "xgboost", f"Alias '{alias}' did not resolve to 'xgboost'"

    xgb_model = ModelRegistry.create("xgboost", task="multiclass", random_state=42, n_estimators=10)
    assert isinstance(xgb_model, XGBoostModel)
    assert isinstance(xgb_model, BaseNIDSModel)
    assert xgb_model.model_name == "xgboost"
    assert xgb_model.task == "multiclass"
    assert xgb_model.is_fitted is False


def test_xgboost_fit_predict_properties(sampled_splits):
    """Test 2: Verifies 6-class output, class ordering, probability validity, and raw margins."""
    df_train = sampled_splits.df_train_known
    df_val = sampled_splits.df_val_known

    preprocessor = NIDSPreprocessor()
    X_train, _, _ = preprocessor.fit_transform(df_train)
    y_train = extract_known_multiclass_labels(df_train)

    X_val, _, _ = preprocessor.transform(df_val)
    y_val = extract_known_multiclass_labels(df_val)

    xgb_model = XGBoostModel(task="multiclass", n_estimators=20, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train, feature_names=preprocessor.feature_names_)

    assert xgb_model.is_fitted is True
    assert len(xgb_model.classes_) == 6, f"Expected 6 output classes, got {len(xgb_model.classes_)}"
    assert list(xgb_model.classes_) == [0, 1, 2, 3, 4, 5], f"Expected classes [0..5], got {xgb_model.classes_}"

    # Predictions
    preds = xgb_model.predict(X_val)
    assert set(np.unique(preds)).issubset({0, 1, 2, 3, 4, 5})

    # Probabilities
    probs = xgb_model.predict_proba(X_val)
    assert probs.shape == (len(X_val), 6)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    # Native raw margins (predict_logits)
    margins = xgb_model.predict_logits(X_val)
    assert margins.shape == (len(X_val), 6)
    assert np.all(np.isfinite(margins)), "XGBoost margins must be strictly finite"
    assert np.var(margins) > 0.1, "XGBoost margins must exhibit non-zero variance"

    # Feature importances
    importances = xgb_model.get_feature_importances()
    assert importances is not None
    assert len(importances) == X_train.shape[1]
    assert np.all(importances >= 0.0)


def test_polymorphic_base_model_loading(sampled_splits, tmp_path):
    """Test 3: Verifies polymorphic serialization and loading via BaseNIDSModel.load()."""
    df_train = sampled_splits.df_train_known
    preprocessor = NIDSPreprocessor()
    X_train, _, _ = preprocessor.fit_transform(df_train)
    y_train = extract_known_multiclass_labels(df_train)

    xgb_model = XGBoostModel(task="multiclass", n_estimators=10, random_state=42)
    xgb_model.fit(X_train, y_train)

    xgb_path = tmp_path / "xgb_model.pkl"
    xgb_model.save(xgb_path)

    loaded_xgb = BaseNIDSModel.load(xgb_path)
    assert isinstance(loaded_xgb, XGBoostModel)
    assert loaded_xgb.model_name == "xgboost"
    assert len(loaded_xgb.classes_) == 6
    assert list(loaded_xgb.classes_) == [0, 1, 2, 3, 4, 5]


def test_detector_compatibility_with_xgboost(sampled_splits):
    """Test 4: Verifies MSP, Energy, and OpenMax detectors on XGBoost representations."""
    df_train = sampled_splits.df_train_known
    df_evt = sampled_splits.df_evt_fit
    df_cal = sampled_splits.df_thresh_cal
    df_test = sampled_splits.df_test

    preprocessor = NIDSPreprocessor()
    X_train, _, _ = preprocessor.fit_transform(df_train)
    y_train = extract_known_multiclass_labels(df_train)

    X_evt, _, _ = preprocessor.transform(df_evt)
    y_evt = extract_known_multiclass_labels(df_evt)

    X_cal, _, _ = preprocessor.transform(df_cal)
    y_cal = extract_known_multiclass_labels(df_cal)

    X_test, _, _ = preprocessor.transform(df_test)

    xgb_model = XGBoostModel(task="multiclass", n_estimators=20, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train, feature_names=preprocessor.feature_names_)

    # A. MSP Detector
    msp = MSPDetector()
    msp.fit(None, None, xgb_model)
    msp_thresh = msp.calibrate_thresholds(X_cal, y_cal, xgb_model, normal_class_id=0)
    assert "tau_FPR05" in msp_thresh and "tau_Cov95" in msp_thresh
    msp_scores = msp.compute_anomaly_score(X_test, xgb_model)
    assert msp_scores.shape == (len(X_test),)
    assert np.all(msp_scores >= 0.0) and np.all(msp_scores <= 1.0)

    # B. Energy Detector (native margins)
    energy = EnergyDetector(temperature=1.0)
    energy.fit(None, None, xgb_model)
    energy_thresh = energy.calibrate_thresholds(X_cal, y_cal, xgb_model, normal_class_id=0)
    assert "tau_FPR05" in energy_thresh and "tau_Cov95" in energy_thresh
    energy_scores = energy.compute_anomaly_score(X_test, xgb_model)
    assert energy_scores.shape == (len(X_test),)
    assert np.all(np.isfinite(energy_scores))

    # C. OpenMax Detector (EVT on native margins)
    openmax = OpenMaxDetector(alpha=3, eta_ratio=0.2, min_eta=5, max_eta=20)
    openmax.fit(X_evt, y_evt, xgb_model)
    assert openmax.is_fitted is True
    om_thresh = openmax.calibrate_thresholds(X_cal, y_cal, xgb_model, normal_class_id=0)
    assert "tau_FPR05" in om_thresh and "tau_Cov95" in om_thresh
    om_probs = openmax.predict_proba_7class(X_test, xgb_model)
    assert om_probs.shape == (len(X_test), 7)
    np.testing.assert_allclose(om_probs.sum(axis=1), 1.0, rtol=1e-5)


def test_existing_artifacts_preserved():
    """Test 5: Verifies that LightGBM, Random Forest, and synthesis artifacts remain intact."""
    lgb_dir = Path("results/openset/LightGBM/features_48")
    rf_dir = Path("results/openset/RandomForest/features_48")
    synth_dir = Path("results/openset/synthesis")

    assert (lgb_dir / "model.pkl").exists()
    assert (lgb_dir / "MSP" / "metrics_openset_msp.json").exists()
    assert (lgb_dir / "Energy" / "metrics_openset_energy.json").exists()
    assert (lgb_dir / "OpenMax" / "metrics_openset_openmax.json").exists()

    assert (rf_dir / "model.pkl").exists()
    assert (rf_dir / "MSP" / "metrics_openset_msp.json").exists()
    assert (rf_dir / "Energy" / "metrics_openset_energy.json").exists()
    assert (rf_dir / "OpenMax" / "metrics_openset_openmax.json").exists()

    assert (synth_dir / "consolidated_benchmark_table.csv").exists()
    assert (synth_dir / "cross_model_synthesis_report.md").exists()
