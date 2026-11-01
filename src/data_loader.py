"""Data loading and ingestion module for UNSW-NB15 with local fallback and download capabilities."""

import logging
from pathlib import Path
import shutil
from typing import Optional, Tuple
import pandas as pd
import requests

from src.config import Config, default_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading, local discovery, and downloading of NIDS datasets."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self.raw_dir = self.config.resolve_path(self.config.dataset.get("raw_dir", "data/raw"))
        self.samples_dir = self.config.resolve_path(self.config.dataset.get("samples_dir", "data/samples"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def ensure_dataset(self) -> Tuple[Path, Path]:
        """Ensures training and test files are available in data/raw.

        Searches:
        1. data/raw directly (CSV or Parquet)
        2. Known system paths (e.g., local backup / capstone directories)
        3. Remote download fallback
        """
        train_csv = self.raw_dir / "unsw_nb15_train.csv"
        test_csv = self.raw_dir / "unsw_nb15_test.csv"
        train_parquet = self.raw_dir / "unsw_nb15_train.parquet"
        test_parquet = self.raw_dir / "unsw_nb15_test.parquet"

        # Check if already present in data/raw
        if (train_csv.exists() or train_parquet.exists()) and (test_csv.exists() or test_parquet.exists()):
            train_target = train_parquet if train_parquet.exists() else train_csv
            test_target = test_parquet if test_parquet.exists() else test_csv
            logger.info("Found existing dataset in data/raw: %s, %s", train_target.name, test_target.name)
            return train_target, test_target

        # Check system fallback paths
        fallback_paths = self.config.dataset.get("system_fallback_paths", [])
        found_train = False
        found_test = False

        for fpath_str in fallback_paths:
            fpath = Path(fpath_str)
            if not fpath.exists():
                continue

            # Look for CSVs
            src_train_csv = fpath / "unsw_nb15_train.csv"
            src_test_csv = fpath / "unsw_nb15_test.csv"
            if src_train_csv.exists() and src_test_csv.exists():
                logger.info("Copying UNSW-NB15 CSV files from system cache: %s", fpath)
                shutil.copy2(src_train_csv, train_csv)
                shutil.copy2(src_test_csv, test_csv)
                return train_csv, test_csv

            # Look for Parquet
            src_train_p = fpath / "UNSW_NB15_training-set.parquet"
            src_test_p = fpath / "UNSW_NB15_testing-set.parquet"
            if src_train_p.exists() and src_test_p.exists():
                logger.info("Copying UNSW-NB15 Parquet files from system cache: %s", fpath)
                shutil.copy2(src_train_p, train_parquet)
                shutil.copy2(src_test_p, test_parquet)
                return train_parquet, test_parquet

        # If not found locally, trigger remote download
        logger.warning("UNSW-NB15 dataset not found in local paths. Starting download...")
        self._download_remote_files(train_csv, test_csv)
        return train_csv, test_csv

    def _download_remote_files(self, train_dest: Path, test_dest: Path) -> None:
        """Downloads official UNSW-NB15 splits from public web mirror."""
        urls = self.config.dataset.get("remote_urls", {})
        train_url = urls.get("train_csv")
        test_url = urls.get("test_csv")

        if not train_url or not test_url:
            raise RuntimeError("No remote URLs specified in configuration for dataset download.")

        def download_file(url: str, dest: Path):
            logger.info("Downloading %s to %s ...", url, dest)
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            logger.info("Finished downloading %s (%d bytes)", dest.name, dest.stat().st_size)

        download_file(train_url, train_dest)
        download_file(test_url, test_dest)

    def load_data(
        self,
        sample_size: Optional[int] = None,
        test_sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads train and test splits into DataFrames.

        Args:
            sample_size: Optional row limit for training set (useful for rapid testing).
            test_sample_size: Optional row limit for test set.
            random_state: Seed for reproducible stratified/random sampling.

        Returns:
            (train_df, test_df)
        """
        train_path, test_path = self.ensure_dataset()

        logger.info("Loading training data from %s ...", train_path)
        if train_path.suffix == ".parquet":
            train_df = pd.read_parquet(train_path)
        else:
            train_df = pd.read_csv(train_path, low_memory=False)

        logger.info("Loading test data from %s ...", test_path)
        if test_path.suffix == ".parquet":
            test_df = pd.read_parquet(test_path)
        else:
            test_df = pd.read_csv(test_path, low_memory=False)

        # Standardize string representations of attack categories
        for df in (train_df, test_df):
            if "attack_cat" in df.columns:
                df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
                df["attack_cat"] = df["attack_cat"].replace({
                    "Backdoors": "Backdoor",
                    "Fuzzers ": "Fuzzers",
                    "Reconnaissance ": "Reconnaissance",
                    "nan": "Normal",
                    "": "Normal",
                }).fillna("Normal")

        # Subsample if requested
        if sample_size is not None and sample_size < len(train_df):
            train_df = train_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            logger.info("Subsampled training set to %d rows", len(train_df))

        if test_sample_size is not None and test_sample_size < len(test_df):
            test_df = test_df.sample(n=test_sample_size, random_state=random_state).reset_index(drop=True)
            logger.info("Subsampled test set to %d rows", len(test_df))

        logger.info("Loaded Train: %s, Test: %s", train_df.shape, test_df.shape)
        return train_df, test_df

    def create_sample_flows(self, n_samples: int = 100) -> Path:
        """Extracts a balanced sample of real flows across all categories for quick testing and dashboard upload."""
        sample_file = self.samples_dir / "sample_flows.csv"
        if sample_file.exists():
            return sample_file

        train_path, _ = self.ensure_dataset()
        if train_path.suffix == ".parquet":
            df = pd.read_parquet(train_path)
        else:
            df = pd.read_csv(train_path, low_memory=False)

        if "attack_cat" in df.columns:
            df["attack_cat"] = df["attack_cat"].astype(str).str.strip().fillna("Normal")
            # Sample evenly across attack categories
            cats = df["attack_cat"].unique()
            per_cat = max(2, n_samples // len(cats))
            sampled_dfs = []
            for cat in cats:
                cat_df = df[df["attack_cat"] == cat]
                k = min(len(cat_df), per_cat)
                sampled_dfs.append(cat_df.sample(n=k, random_state=42))
            sample_df = pd.concat(sampled_dfs, ignore_index=True)
        else:
            sample_df = df.sample(n=min(len(df), n_samples), random_state=42)

        sample_df.to_csv(sample_file, index=False)
        logger.info("Generated sample flows file with %d records at %s", len(sample_df), sample_file)
        return sample_file


if __name__ == "__main__":
    loader = DataLoader()
    tr, ts = loader.load_data(sample_size=1000, test_sample_size=500)
    print("Train categories:", tr["attack_cat"].value_counts())
    sample_path = loader.create_sample_flows(100)
    print("Created sample file at:", sample_path)
