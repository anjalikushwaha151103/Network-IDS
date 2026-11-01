"""Preprocessing pipeline for NIDS: cleaning, feature engineering, encoding, and scaling."""

import logging
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

from src.config import Config, default_config

logger = logging.getLogger(__name__)

# Standard attack category mapping for UNSW-NB15
STANDARD_ATTACK_CATEGORIES = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
]


class NIDSPreprocessor:
    """Stateful preprocessor that handles cleaning, feature engineering,

    categorical encoding, and numerical scaling for NIDS traffic flows.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self.prep_cfg = self.config.preprocessing

        self.drop_columns: List[str] = self.prep_cfg.get("drop_columns", [])
        self.categorical_cols: List[str] = self.prep_cfg.get("categorical_cols", ["proto", "service", "state"])
        self.scaling_method: str = self.prep_cfg.get("scaling_method", "standard")
        self.engineer_features: bool = self.prep_cfg.get("engineer_features", True)

        self.target_binary_col: str = self.prep_cfg.get("target_binary_col", "label")
        self.target_multiclass_col: str = self.prep_cfg.get("target_multiclass_col", "attack_cat")

        # Fitted state
        self.is_fitted: bool = False
        self.feature_names_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.label_encoders_: Dict[str, LabelEncoder] = {}
        self.unknown_cat_code_: Dict[str, int] = {}
        self.scaler_: Optional[Union[StandardScaler, RobustScaler]] = None
        self.class_to_id_: Dict[str, int] = {}
        self.id_to_class_: Dict[int, str] = {}
        self.numeric_medians_: Dict[str, float] = {}

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives standard domain-specific flow interaction features."""
        df = df.copy()

        # Total bytes & packets
        if "sbytes" in df.columns and "dbytes" in df.columns:
            df["total_bytes"] = df["sbytes"].astype(float) + df["dbytes"].astype(float)
            df["byte_ratio"] = df["sbytes"].astype(float) / (df["dbytes"].astype(float) + 1.0)
        else:
            df["total_bytes"] = 0.0
            df["byte_ratio"] = 0.0

        if "spkts" in df.columns and "dpkts" in df.columns:
            df["total_pkts"] = df["spkts"].astype(float) + df["dpkts"].astype(float)
            df["pkt_ratio"] = df["spkts"].astype(float) / (df["dpkts"].astype(float) + 1.0)
        else:
            df["total_pkts"] = 0.0
            df["pkt_ratio"] = 0.0

        if "total_bytes" in df.columns and "total_pkts" in df.columns:
            df["avg_pkt_size"] = df["total_bytes"] / (df["total_pkts"] + 1.0)
        else:
            df["avg_pkt_size"] = 0.0

        if "dur" in df.columns:
            df["duration_log"] = np.log1p(np.maximum(0.0, df["dur"].astype(float).fillna(0.0)))
        else:
            df["duration_log"] = 0.0

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles infinite values, NaNs, and drops identifier columns."""
        df = df.copy()

        # Drop non-generalizable identifiers if present
        cols_to_drop = [c for c in self.drop_columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Standardize categorical strings
        for cat_col in self.categorical_cols:
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].astype(str).str.strip().str.lower()
                df[cat_col] = df[cat_col].replace({"-": "unknown", "": "unknown", "nan": "unknown"})

        # Clean infinities in numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

        return df

    def fit(self, df: pd.DataFrame) -> "NIDSPreprocessor":
        """Learns categorical vocabularies, numerical medians, and scaling statistics from training data."""
        logger.info("Fitting NIDSPreprocessor on DataFrame with shape %s", df.shape)
        cleaned = self._clean_data(df)

        if self.engineer_features:
            cleaned = self._engineer_features(cleaned)

        # Separate targets from features
        feature_df = cleaned.drop(
            columns=[c for c in [self.target_binary_col, self.target_multiclass_col] if c in cleaned.columns]
        )

        # Identify categorical vs numerical columns
        actual_cat_cols = [c for c in self.categorical_cols if c in feature_df.columns]
        actual_num_cols = [c for c in feature_df.columns if c not in actual_cat_cols]

        self.numeric_cols_ = sorted(actual_num_cols)
        self.feature_names_ = actual_cat_cols + self.numeric_cols_

        # Fit label encoders for categoricals
        for col in actual_cat_cols:
            le = LabelEncoder()
            unique_vals = list(feature_df[col].dropna().unique())
            if "unknown" not in unique_vals:
                unique_vals.append("unknown")
            le.fit(unique_vals)
            self.label_encoders_[col] = le
            self.unknown_cat_code_[col] = int(le.transform(["unknown"])[0])

        # Compute numerical medians for imputation
        for col in self.numeric_cols_:
            median_val = float(feature_df[col].median()) if not feature_df[col].dropna().empty else 0.0
            self.numeric_medians_[col] = 0.0 if np.isnan(median_val) else median_val

        # Impute numericals before scaling
        imputed_num = feature_df[self.numeric_cols_].copy()
        for col in self.numeric_cols_:
            imputed_num[col] = imputed_num[col].fillna(self.numeric_medians_[col])

        # Fit scaler
        if self.scaling_method == "robust":
            self.scaler_ = RobustScaler()
        else:
            self.scaler_ = StandardScaler()

        self.scaler_.fit(imputed_num)

        # Fit multiclass targets if available
        if self.target_multiclass_col in cleaned.columns:
            raw_cats = cleaned[self.target_multiclass_col].astype(str).str.strip().unique()
            # Combine standard categories with any present categories to preserve canonical order
            all_cats = list(STANDARD_ATTACK_CATEGORIES)
            for c in raw_cats:
                if c not in all_cats:
                    all_cats.append(c)

            self.class_to_id_ = {cat: idx for idx, cat in enumerate(all_cats)}
            self.id_to_class_ = {idx: cat for cat, idx in self.class_to_id_.items()}

        self.is_fitted = True
        logger.info(
            "Preprocessor fitted: %d features (%d categorical, %d numerical)",
            len(self.feature_names_),
            len(actual_cat_cols),
            len(self.numeric_cols_),
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        feature_subset: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Transforms raw flow DataFrame into scaled feature matrix X and target vectors.

        Args:
            df: Raw DataFrame containing network flows.
            feature_subset: Optional list of feature names to filter output columns (for Novelty 3).

        Returns:
            X: Transformed 2D numpy array of shape (n_samples, n_features).
            y_binary: 1D numpy array of binary labels (or None if not present).
            y_multi: 1D numpy array of multiclass labels (or None if not present).
        """
        if not self.is_fitted:
            raise RuntimeError("NIDSPreprocessor must be fitted before calling transform().")

        cleaned = self._clean_data(df)
        if self.engineer_features:
            cleaned = self._engineer_features(cleaned)

        # Encode categoricals
        encoded_parts = []
        for col in [c for c in self.categorical_cols if c in self.label_encoders_]:
            le = self.label_encoders_[col]
            unknown_code = self.unknown_cat_code_[col]
            val_series = cleaned[col].fillna("unknown") if col in cleaned.columns else pd.Series(["unknown"] * len(cleaned))
            # Handle unseen labels safely
            encoded = val_series.map(lambda s: le.transform([s])[0] if s in le.classes_ else unknown_code).to_numpy()
            encoded_parts.append(encoded.reshape(-1, 1))

        # Impute and scale numericals
        num_data = pd.DataFrame(index=cleaned.index)
        for col in self.numeric_cols_:
            if col in cleaned.columns:
                num_data[col] = cleaned[col].fillna(self.numeric_medians_[col])
            else:
                num_data[col] = self.numeric_medians_[col]

        scaled_nums = self.scaler_.transform(num_data[self.numeric_cols_])

        # Combine into full matrix
        if encoded_parts:
            cat_matrix = np.hstack(encoded_parts)
            full_X = np.hstack([cat_matrix, scaled_nums])
        else:
            full_X = scaled_nums

        feature_cols = [c for c in self.categorical_cols if c in self.label_encoders_] + self.numeric_cols_

        # Apply feature subset filtering if requested (e.g. 30, 20, 15 features)
        if feature_subset is not None:
            indices = [feature_cols.index(f) for f in feature_subset if f in feature_cols]
            X = full_X[:, indices]
        else:
            X = full_X

        # Extract binary labels
        y_binary = None
        if self.target_binary_col in cleaned.columns:
            y_binary = cleaned[self.target_binary_col].astype(int).to_numpy()

        # Extract multiclass labels
        y_multi = None
        if self.target_multiclass_col in cleaned.columns:
            raw_cats = cleaned[self.target_multiclass_col].astype(str).str.strip()
            y_multi = raw_cats.map(lambda c: self.class_to_id_.get(c, 0)).to_numpy()

        return X, y_binary, y_multi

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_subset: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Fits preprocessor and transforms DataFrame in a single call."""
        return self.fit(df).transform(df, feature_subset=feature_subset)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes fitted preprocessor to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved preprocessor to %s", filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "NIDSPreprocessor":
        """Loads serialized preprocessor from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {filepath}")
        with open(filepath, "rb") as f:
            preprocessor = pickle.load(f)
        logger.info("Loaded preprocessor from %s", filepath)
        return preprocessor
