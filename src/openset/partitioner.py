"""Four-way stratified data partitioning for Open-Set / Zero-Day Attack Detection.

Implements strict, leakage-free isolation:
- Known Classes (6): Normal, Analysis, Generic, Exploits, Fuzzers, DoS
- Unknown Classes (4): Reconnaissance, Backdoor, Shellcode, Worms (purged from training)

Disjoint 4-way stratified split of known training flows (seed=42):
- D_train_known (70.0%): Fits preprocessor, feature selector, and 6-class classifier
- D_val_known   (15.0%): Used for model validation and early stopping
- D_evt_fit     (7.5%):  Used exclusively for OpenMax MAV computation and Weibull fitting
- D_thresh_cal  (7.5%):  Used exclusively for OOD operating threshold calibration

Official test split (82,332 flows) remains untouched until final benchmark evaluation.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Strict class partition definition
KNOWN_CLASSES: List[str] = [
    "Normal",
    "Analysis",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
]

UNKNOWN_CLASSES: List[str] = [
    "Reconnaissance",
    "Backdoor",
    "Shellcode",
    "Worms",
]

ALL_CLASSES: List[str] = KNOWN_CLASSES + UNKNOWN_CLASSES

KNOWN_CLASS_TO_ID: Dict[str, int] = {cat: idx for idx, cat in enumerate(KNOWN_CLASSES)}
ID_TO_KNOWN_CLASS: Dict[int, str] = {idx: cat for cat, idx in KNOWN_CLASS_TO_ID.items()}
UNKNOWN_CLASS_TO_ID: Dict[str, int] = {cat: idx for idx, cat in enumerate(UNKNOWN_CLASSES)}
UNKNOWN_LABEL_ID: int = len(KNOWN_CLASSES)  # 6 (representing the 7th category: Unknown)

# Canonical 7-class labels for open-set evaluation
OPENSET_7CLASSES: List[str] = KNOWN_CLASSES + ["Unknown"]
OPENSET_7CLASS_TO_ID: Dict[str, int] = {cat: idx for idx, cat in enumerate(OPENSET_7CLASSES)}
ID_TO_OPENSET_7CLASS: Dict[int, str] = {idx: cat for cat, idx in OPENSET_7CLASS_TO_ID.items()}


@dataclass
class OpenSetSplits:
    """Container holding all disjoint subsets for leakage-free open-set experimentation."""

    # 4-way disjoint splits of known training data
    df_train_known: pd.DataFrame
    df_val_known: pd.DataFrame
    df_evt_fit: pd.DataFrame
    df_thresh_cal: pd.DataFrame

    # Untouched official test split and its partitions
    df_test: pd.DataFrame
    df_test_known: pd.DataFrame
    df_test_unknown: pd.DataFrame

    # Purged training unknown records
    purged_train_unknown_count: int

    def summary(self) -> Dict[str, int]:
        """Returns row count summary of all subsets."""
        return {
            "train_known_70pct": len(self.df_train_known),
            "val_known_15pct": len(self.df_val_known),
            "evt_fit_7.5pct": len(self.df_evt_fit),
            "thresh_cal_7.5pct": len(self.df_thresh_cal),
            "total_known_train": (
                len(self.df_train_known)
                + len(self.df_val_known)
                + len(self.df_evt_fit)
                + len(self.df_thresh_cal)
            ),
            "purged_train_unknowns": self.purged_train_unknown_count,
            "test_total": len(self.df_test),
            "test_known": len(self.df_test_known),
            "test_unknown": len(self.df_test_unknown),
        }

    def verify_leakage_free(self) -> bool:
        """Verifies that no unknown attack classes leak into any training/validation/calibration split."""
        for name, df in [
            ("train_known", self.df_train_known),
            ("val_known", self.df_val_known),
            ("evt_fit", self.df_evt_fit),
            ("thresh_cal", self.df_thresh_cal),
        ]:
            if "attack_cat" in df.columns:
                leaked = df["attack_cat"].isin(UNKNOWN_CLASSES).sum()
                if leaked > 0:
                    raise AssertionError(
                        f"CRITICAL LEAKAGE: {leaked} unknown attack flows found in {name}!"
                    )
        return True


def standardize_attack_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes category strings, trimming whitespace and resolving known typographical variants."""
    df = df.copy()
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
        df["attack_cat"] = df["attack_cat"].replace({
            "Backdoors": "Backdoor",
            "Fuzzers ": "Fuzzers",
            "Reconnaissance ": "Reconnaissance",
            "nan": "Normal",
            "": "Normal",
        }).fillna("Normal")
    return df


def partition_openset_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
) -> OpenSetSplits:
    """Partitions raw training and test splits into strict, disjoint open-set evaluation sets.

    Args:
        train_df: Raw official training DataFrame.
        test_df: Raw official test DataFrame.
        random_state: Fixed random seed for stratified splitting (default=42).

    Returns:
        OpenSetSplits containing the disjoint 4-way train splits and untouched test splits.
    """
    # 1. Clean category strings
    train_clean = standardize_attack_categories(train_df)
    test_clean = standardize_attack_categories(test_df)

    # 2. Separate known vs unknown training flows
    is_train_known = train_clean["attack_cat"].isin(KNOWN_CLASSES)
    df_known_train = train_clean[is_train_known].copy()
    purged_train_unknown_count = int((~is_train_known).sum())

    logger.info(
        "Purged %d unknown attack flows from training set. Retained %d known flows.",
        purged_train_unknown_count,
        len(df_known_train),
    )

    # 3. Disjoint Stratified 4-way Splitting:
    # Step A: 70% train vs 30% temp
    df_train_known, df_temp = train_test_split(
        df_known_train,
        test_size=0.30,
        random_state=random_state,
        stratify=df_known_train["attack_cat"],
    )

    # Step B: 30% temp split 50/50 into 15% validation and 15% calibration pool
    df_val_known, df_cal_pool = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=df_temp["attack_cat"],
    )

    # Step C: 15% calibration pool split 50/50 into 7.5% EVT fitting and 7.5% threshold calibration
    df_evt_fit, df_thresh_cal = train_test_split(
        df_cal_pool,
        test_size=0.50,
        random_state=random_state,
        stratify=df_cal_pool["attack_cat"],
    )

    # Verify zero overlap between all four index sets
    idx_train = set(df_train_known.index)
    idx_val = set(df_val_known.index)
    idx_evt = set(df_evt_fit.index)
    idx_cal = set(df_thresh_cal.index)

    assert len(idx_train & idx_val) == 0, "Overlap detected between Train and Val sets"
    assert len(idx_train & idx_evt) == 0, "Overlap detected between Train and EVT Fit sets"
    assert len(idx_train & idx_cal) == 0, "Overlap detected between Train and Threshold Cal sets"
    assert len(idx_val & idx_evt) == 0, "Overlap detected between Val and EVT Fit sets"
    assert len(idx_val & idx_cal) == 0, "Overlap detected between Val and Threshold Cal sets"
    assert len(idx_evt & idx_cal) == 0, "Overlap detected between EVT Fit and Threshold Cal sets"

    # Reset indices for downstream numpy compatibility
    df_train_known = df_train_known.reset_index(drop=True)
    df_val_known = df_val_known.reset_index(drop=True)
    df_evt_fit = df_evt_fit.reset_index(drop=True)
    df_thresh_cal = df_thresh_cal.reset_index(drop=True)

    # 4. Untouched official test split partitioning
    is_test_known = test_clean["attack_cat"].isin(KNOWN_CLASSES)
    df_test_known = test_clean[is_test_known].reset_index(drop=True)
    df_test_unknown = test_clean[~is_test_known].reset_index(drop=True)
    df_test = test_clean.reset_index(drop=True)

    splits = OpenSetSplits(
        df_train_known=df_train_known,
        df_val_known=df_val_known,
        df_evt_fit=df_evt_fit,
        df_thresh_cal=df_thresh_cal,
        df_test=df_test,
        df_test_known=df_test_known,
        df_test_unknown=df_test_unknown,
        purged_train_unknown_count=purged_train_unknown_count,
    )

    # Run assertion verification
    splits.verify_leakage_free()
    return splits


def extract_known_multiclass_labels(df: pd.DataFrame) -> np.ndarray:
    """Extracts integer multiclass labels (0..5) for known categories."""
    cats = df["attack_cat"].astype(str).str.strip()
    return cats.map(KNOWN_CLASS_TO_ID).to_numpy(dtype=int)


def extract_openset_7class_labels(df: pd.DataFrame) -> np.ndarray:
    """Extracts open-set evaluation labels (0..5 for known classes, 6 for any unknown attack class)."""
    cats = df["attack_cat"].astype(str).str.strip()
    return cats.map(lambda c: KNOWN_CLASS_TO_ID.get(c, UNKNOWN_LABEL_ID)).to_numpy(dtype=int)


def extract_is_unknown(df: pd.DataFrame) -> np.ndarray:
    """Extracts boolean indicator where True = Unknown Attack class, False = Known class."""
    cats = df["attack_cat"].astype(str).str.strip()
    return cats.isin(UNKNOWN_CLASSES).to_numpy(dtype=bool)
