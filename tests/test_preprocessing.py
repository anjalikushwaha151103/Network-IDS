"""Unit tests for the NIDS preprocessor."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data_loader import DataLoader
from src.preprocessing import NIDSPreprocessor


@pytest.fixture
def sample_data():
    loader = DataLoader()
    train_df, test_df = loader.load_data(sample_size=200, test_sample_size=100)
    return train_df, test_df


def test_preprocessor_fit_transform(sample_data):
    train_df, test_df = sample_data
    prep = NIDSPreprocessor()
    X_train, y_train_bin, y_train_multi = prep.fit_transform(train_df)

    assert prep.is_fitted
    assert X_train.shape[0] == len(train_df)
    assert X_train.shape[1] == len(prep.feature_names_)
    assert not np.isnan(X_train).any(), "No NaNs should exist in transformed matrix"
    assert not np.isinf(X_train).any(), "No Infs should exist in transformed matrix"

    # Transform test set
    X_test, y_test_bin, y_test_multi = prep.transform(test_df)
    assert X_test.shape[0] == len(test_df)
    assert X_test.shape[1] == X_train.shape[1]
    assert not np.isnan(X_test).any()


def test_unseen_categorical_handling(sample_data):
    train_df, _ = sample_data
    prep = NIDSPreprocessor()
    prep.fit(train_df)

    # Create dummy dataframe with totally novel protocol and state
    dummy = train_df.iloc[:5].copy()
    dummy["proto"] = "quantum_protocol"
    dummy["state"] = "superposition"

    X_dummy, _, _ = prep.transform(dummy)
    assert X_dummy.shape[0] == 5
    assert not np.isnan(X_dummy).any()


def test_feature_subsetting(sample_data):
    train_df, _ = sample_data
    prep = NIDSPreprocessor()
    prep.fit(train_df)

    subset = prep.feature_names_[:15]
    X_sub, _, _ = prep.transform(train_df, feature_subset=subset)
    assert X_sub.shape[1] == 15


def test_serialization(tmp_path, sample_data):
    train_df, test_df = sample_data
    prep = NIDSPreprocessor()
    X1, _, _ = prep.fit_transform(train_df)

    save_path = tmp_path / "preprocessor.pkl"
    prep.save(save_path)

    loaded_prep = NIDSPreprocessor.load(save_path)
    X2, _, _ = loaded_prep.transform(train_df)

    assert np.allclose(X1, X2), "Loaded preprocessor must yield identical transforms"
