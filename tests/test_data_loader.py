"""Unit tests for the data loading module."""

from pathlib import Path
import pytest
from src.data_loader import DataLoader


def test_data_loader_ensure_dataset():
    loader = DataLoader()
    train_path, test_path = loader.ensure_dataset()
    assert train_path.exists(), "Train file should exist"
    assert test_path.exists(), "Test file should exist"


def test_data_loader_sampling():
    loader = DataLoader()
    train_df, test_df = loader.load_data(sample_size=100, test_sample_size=50)
    assert len(train_df) == 100
    assert len(test_df) == 50
    assert "attack_cat" in train_df.columns
    assert "label" in train_df.columns


def test_sample_flows_file():
    loader = DataLoader()
    sample_file = loader.create_sample_flows(50)
    assert sample_file.exists()
    assert sample_file.stat().st_size > 0
