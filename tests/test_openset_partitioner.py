"""Comprehensive unit tests for the Open-Set 4-way stratified data partitioner.

Verifies:
1. Exact partition sizes and class distributions.
2. Zero index overlap between subsets.
3. Zero unknown-class leakage into any training/validation/calibration split.
4. Preprocessing fitted exclusively on training data.
5. Feature selection fitted exclusively on training data.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.feature_selection import FeatureSelector
from src.openset.partitioner import (
    KNOWN_CLASSES,
    UNKNOWN_CLASSES,
    UNKNOWN_LABEL_ID,
    extract_is_unknown,
    extract_known_multiclass_labels,
    extract_openset_7class_labels,
    partition_openset_data,
)
from src.preprocessing import NIDSPreprocessor


@pytest.fixture(scope="module")
def official_data():
    """Loads full official UNSW-NB15 dataset for exact verification."""
    loader = DataLoader()
    train_df, test_df = loader.load_data()
    return train_df, test_df


@pytest.fixture(scope="module")
def openset_splits(official_data):
    """Generates the official 4-way open-set partitioned splits."""
    train_df, test_df = official_data
    return partition_openset_data(train_df, test_df, random_state=42)


def test_exact_partition_counts(openset_splits):
    """Test 1: Verifies exact sample counts across all 4-way splits and test partitions."""
    s = openset_splits.summary()

    # Exact expected counts
    assert s["train_known_70pct"] == 113288, "D_train_known must have exactly 113,288 rows (70.0%)"
    assert s["val_known_15pct"] == 24276, "D_val_known must have exactly 24,276 rows (15.0%)"
    assert s["evt_fit_7.5pct"] == 12138, "D_evt_fit must have exactly 12,138 rows (7.5%)"
    assert s["thresh_cal_7.5pct"] == 12139, "D_thresh_cal must have exactly 12,139 rows (7.5%)"

    # Known total
    assert s["total_known_train"] == 161841, "Sum of known training subsets must equal 161,841"

    # Purged training unknowns
    assert s["purged_train_unknowns"] == 13500, "Purged training unknowns must equal 13,500"

    # Test set counts
    assert s["test_total"] == 82332, "Official test set must have exactly 82,332 rows"
    assert s["test_known"] == 77831, "Test known subset must have exactly 77,831 rows"
    assert s["test_unknown"] == 4501, "Test unknown subset must have exactly 4,501 rows"


def test_stratified_class_distributions(openset_splits):
    """Test 2: Verifies exact per-class counts in each partitioned subset."""
    df_train = openset_splits.df_train_known
    df_val = openset_splits.df_val_known
    df_evt = openset_splits.df_evt_fit
    df_cal = openset_splits.df_thresh_cal

    expected_per_class = {
        "Normal": {"train": 39200, "val": 8400, "evt": 4200, "cal": 4200},
        "Generic": {"train": 28000, "val": 6000, "evt": 3000, "cal": 3000},
        "Exploits": {"train": 23375, "val": 5009, "evt": 2504, "cal": 2505},
        "Fuzzers": {"train": 12728, "val": 2728, "evt": 1364, "cal": 1364},
        "DoS": {"train": 8585, "val": 1839, "evt": 920, "cal": 920},
        "Analysis": {"train": 1400, "val": 300, "evt": 150, "cal": 150},
    }

    for cat, exp in expected_per_class.items():
        assert (df_train["attack_cat"] == cat).sum() == exp["train"]
        assert (df_val["attack_cat"] == cat).sum() == exp["val"]
        assert (df_evt["attack_cat"] == cat).sum() == exp["evt"]
        assert (df_cal["attack_cat"] == cat).sum() == exp["cal"]


def test_zero_leakage_and_pure_known_classes(openset_splits):
    """Test 3: Confirms zero unknown attack classes exist in any training/val/calibration subset."""
    openset_splits.verify_leakage_free()

    for name, df in [
        ("Train Known", openset_splits.df_train_known),
        ("Val Known", openset_splits.df_val_known),
        ("EVT Fit", openset_splits.df_evt_fit),
        ("Threshold Cal", openset_splits.df_thresh_cal),
    ]:
        present_cats = set(df["attack_cat"].unique())
        assert present_cats.issubset(set(KNOWN_CLASSES)), f"{name} contains non-known classes: {present_cats - set(KNOWN_CLASSES)}"
        for unk in UNKNOWN_CLASSES:
            assert unk not in present_cats, f"CRITICAL: Unknown class '{unk}' leaked into {name}!"


def test_label_extraction_helpers(openset_splits):
    """Test 4: Verifies label extraction functions for known multiclass and 7-class open-set evaluation."""
    # Multiclass labels on known training data
    y_known = extract_known_multiclass_labels(openset_splits.df_train_known)
    assert len(y_known) == 113288
    assert set(np.unique(y_known)) == {0, 1, 2, 3, 4, 5}

    # 7-class open-set labels on official test split
    y_7class = extract_openset_7class_labels(openset_splits.df_test)
    assert len(y_7class) == 82332
    assert set(np.unique(y_7class)) == {0, 1, 2, 3, 4, 5, UNKNOWN_LABEL_ID}
    assert (y_7class == UNKNOWN_LABEL_ID).sum() == 4501

    # Is unknown boolean mask
    is_unk = extract_is_unknown(openset_splits.df_test)
    assert is_unk.sum() == 4501
    assert (~is_unk).sum() == 77831


def test_preprocessing_fitted_exclusively_on_train(openset_splits):
    """Test 5: Verifies that preprocessor is fitted exclusively on D_train_known without data leakage."""
    df_train = openset_splits.df_train_known
    df_val = openset_splits.df_val_known
    df_evt = openset_splits.df_evt_fit
    df_cal = openset_splits.df_thresh_cal
    df_test = openset_splits.df_test

    preprocessor = NIDSPreprocessor()
    preprocessor.fit(df_train)

    assert preprocessor.is_fitted
    assert len(preprocessor.feature_names_) == 48

    # Transform all subsets using the preprocessor fitted on train ONLY
    X_train, _, _ = preprocessor.transform(df_train)
    X_val, _, _ = preprocessor.transform(df_val)
    X_evt, _, _ = preprocessor.transform(df_evt)
    X_cal, _, _ = preprocessor.transform(df_cal)
    X_test, _, _ = preprocessor.transform(df_test)

    assert X_train.shape == (113288, 48)
    assert X_val.shape == (24276, 48)
    assert X_evt.shape == (12138, 48)
    assert X_cal.shape == (12139, 48)
    assert X_test.shape == (82332, 48)

    # Ensure no NaN or infinite values
    for name, X in [("Train", X_train), ("Val", X_val), ("EVT", X_evt), ("Cal", X_cal), ("Test", X_test)]:
        assert not np.isnan(X).any(), f"NaN values detected in {name} feature matrix"
        assert not np.isinf(X).any(), f"Infinite values detected in {name} feature matrix"


def test_feature_selection_fitted_exclusively_on_train(openset_splits):
    """Test 6: Verifies that feature selector is fitted exclusively on D_train_known without data leakage."""
    df_train = openset_splits.df_train_known

    preprocessor = NIDSPreprocessor()
    X_train, _, _ = preprocessor.fit_transform(df_train)
    y_train_multi = extract_known_multiclass_labels(df_train)

    # Fit feature selector on training subset ONLY
    selector = FeatureSelector()
    selector.ranking_method = "tree_importance"
    selector.ranking_file = selector.cache_dir / "feature_ranking_openset_test.csv"
    ranking_df = selector.compute_ranking(
        X_train[:2000],  # Sample 2000 train flows for fast unit test execution
        y_train_multi[:2000],
        preprocessor.feature_names_,
        force_recompute=True,
    )

    assert len(ranking_df) == 48
    top_30 = selector.get_tier_features(30)
    top_20 = selector.get_tier_features(20)
    top_15 = selector.get_tier_features(15)

    assert len(top_30) == 30
    assert len(top_20) == 20
    assert len(top_15) == 15
    assert set(top_20).issubset(set(top_30))
    assert set(top_15).issubset(set(top_20))
