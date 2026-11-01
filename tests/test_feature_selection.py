"""Unit tests for FeatureSelector."""

import numpy as np
import pytest
from src.data_loader import DataLoader
from src.feature_selection import FeatureSelector
from src.preprocessing import NIDSPreprocessor


def test_feature_selector_ranking():
    loader = DataLoader()
    train_df, _ = loader.load_data(sample_size=300)
    prep = NIDSPreprocessor()
    X, y_bin, _ = prep.fit_transform(train_df)

    selector = FeatureSelector()
    ranking_df = selector.compute_ranking(X, y_bin, prep.feature_names_, force_recompute=True)

    assert len(ranking_df) == len(prep.feature_names_)
    assert "feature" in ranking_df.columns
    assert "score" in ranking_df.columns
    assert "rank" in ranking_df.columns

    # Verify scores are sorted descending
    assert ranking_df["score"].is_monotonic_decreasing

    # Test tier extraction
    tier_30 = selector.get_tier_features(30)
    assert len(tier_30) == 30
    assert tier_30 == ranking_df["feature"].head(30).tolist()

    tier_15 = selector.get_tier_features(15)
    assert len(tier_15) == 15
    assert set(tier_15).issubset(set(tier_30))
