"""Feature selection module supporting ranking via Mutual Information and Tree Importance,

generating feature tiers (e.g. 40, 30, 20, 15) for Novelty 3.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from src.config import Config, default_config

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Computes, caches, and provides feature subsets across defined tiers."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self.fs_cfg = self.config.feature_selection
        self.ranking_method: str = self.fs_cfg.get("ranking_method", "mutual_info")
        self.tiers: List[int] = self.fs_cfg.get("tiers", [40, 30, 20, 15])
        self.active_tier: int = self.fs_cfg.get("active_tier", 30)

        self.cache_dir = self.config.resolve_path("data/processed")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ranking_file = self.cache_dir / f"feature_ranking_{self.ranking_method}.csv"

        self.ranking_df: Optional[pd.DataFrame] = None

    def compute_ranking(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Computes feature importance rankings and persists them to cache.

        Args:
            X: Scaled feature matrix of shape (n_samples, n_features).
            y: Target label vector (binary or multiclass).
            feature_names: List of feature names corresponding to columns of X.
            force_recompute: If True, ignores cached ranking.

        Returns:
            DataFrame with columns ['feature', 'score', 'rank'].
        """
        if not force_recompute and self.ranking_file.exists():
            logger.info("Loading cached feature ranking from %s", self.ranking_file)
            self.ranking_df = pd.read_csv(self.ranking_file)
            return self.ranking_df

        logger.info(
            "Computing feature ranking using '%s' on %d samples and %d features...",
            self.ranking_method,
            len(X),
            len(feature_names),
        )

        if self.ranking_method == "tree_importance":
            rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            scores = rf.feature_importances_
        else:  # mutual_info
            # Subsample if dataset is very large to ensure responsive execution
            if len(X) > 20000:
                indices = np.random.RandomState(42).choice(len(X), size=20000, replace=False)
                X_sub, y_sub = X[indices], y[indices]
            else:
                X_sub, y_sub = X, y
            scores = mutual_info_classif(X_sub, y_sub, random_state=42)

        ranking_df = pd.DataFrame({
            "feature": feature_names,
            "score": scores,
        }).sort_values(by="score", ascending=False).reset_index(drop=True)

        ranking_df["rank"] = np.arange(1, len(ranking_df) + 1)
        ranking_df.to_csv(self.ranking_file, index=False)
        self.ranking_df = ranking_df
        logger.info("Saved feature ranking to %s", self.ranking_file)
        return ranking_df

    def get_tier_features(self, n_features: Optional[int] = None, all_features: Optional[List[str]] = None) -> List[str]:
        """Returns the top N features based on computed ranking.

        If ranking is not yet computed, falls back to the first n_features of all_features.
        """
        target_n = n_features if n_features is not None else self.active_tier

        if self.ranking_df is None and self.ranking_file.exists():
            self.ranking_df = pd.read_csv(self.ranking_file)

        if self.ranking_df is not None:
            top_features = self.ranking_df["feature"].head(target_n).tolist()
            return top_features

        if all_features is not None:
            logger.warning("No ranking available. Falling back to first %d features from pool.", target_n)
            return all_features[:target_n]

        raise ValueError("Cannot determine tier features: no ranking computed and no feature pool provided.")
