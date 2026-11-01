"""Energy-Based Out-of-Distribution Detector (Liu et al., NeurIPS 2020).

Implements the numerically stable free energy formulation with temperature scaling:
    E(x; T) = -z_max - T * log(sum(exp((z_k - z_max) / T)))
where z_max = max_k z_k(x).

Higher free energy indicates lower likelihood under known training distributions,
signaling an Unknown / Zero-Day attack.
Known-confidence is defined as negative free energy: C_known(x) = -E(x; T).
"""

import numpy as np

from src.models.base_model import BaseNIDSModel
from src.openset.base_detector import BaseOODDetector


class EnergyDetector(BaseOODDetector):
    """Energy-based OOD detector using unnormalized model logits and LogSumExp scaling."""

    def __init__(self, temperature: float = 1.0):
        super().__init__(f"Energy-Based Detector (T={temperature})")
        self.temperature = float(temperature)

    def fit(self, X_evt: np.ndarray, y_evt: np.ndarray, model: BaseNIDSModel) -> "EnergyDetector":
        """Energy detection requires no fitted tail distributions or centroids."""
        self.is_fitted = True
        return self

    def compute_anomaly_score(self, X: np.ndarray, model: BaseNIDSModel) -> np.ndarray:
        """Computes numerically stable free energy anomaly scores.

        Formula:
            E(x; T) = -z_max - T * log(sum_{k=1}^K exp((z_k(x) - z_max) / T))
        """
        logits = model.predict_logits(X)
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits matrix, got shape {logits.shape}")
        if not np.all(np.isfinite(logits)):
            raise ValueError("Non-finite values encountered in raw logits returned by model.predict_logits()")

        T = self.temperature
        z_max = np.max(logits, axis=1, keepdims=True)  # (N, 1)

        # Numerically stable exponentiation: max argument is 0.0, preventing overflow
        scaled_diff = (logits - z_max) / T
        sum_exp = np.sum(np.exp(scaled_diff), axis=1, keepdims=True)  # (N, 1), guaranteed >= 1.0

        # Energy calculation
        energy = -z_max - T * np.log(sum_exp)
        return energy.squeeze(axis=1)

    def compute_confidence_score(self, X: np.ndarray, model: BaseNIDSModel) -> np.ndarray:
        """Computes known-confidence score as negative free energy: C_known(x) = -E(x; T)."""
        return -self.compute_anomaly_score(X, model)
