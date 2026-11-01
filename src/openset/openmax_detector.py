"""OpenMax (EVT-Based Adaptation) for Tree-Based Classifiers (Bendale & Boult, 2016 adaptation).

Adapts the OpenMax methodology for tree-based models (such as LightGBM) using
raw class margin vectors (unnormalized margins from predict_logits).

Methodology & Calibration Scope:
1. EVT Fitting (Strictly D_evt_fit):
   - Computes Class Mean Activation Vectors (MAVs) on correctly classified known samples.
   - Computes Euclidean distances d(v, mu_c) for all correctly classified samples.
   - Identifies upper-tail distances with eta_c = min(max_eta, max(min_eta, floor(eta_ratio * n_c))).
   - Fits class-specific Weibull distributions with location parameter floc fixed to min(tail).
   - Handles low sample support, zero variance, and numerical fitting failures deterministically.
2. Inference & Top-Alpha Recalibration:
   - Ranks classes in descending activation order: rho = argsort(z)[::-1].
   - Recalibrates top alpha classes (alpha=3) using rank-weighted Weibull CDF attenuation:
     w_{rho[r]} = 1.0 - ((alpha - r) / alpha) * omega_{rho[r]}.
   - Synthesizes an explicit Unknown activation logit:
     z_unknown = sum_{r=0}^{alpha-1} z_{rho[r]} * (1 - w_{rho[r]}).
   - Computes 7-class probability vector using numerically stable softmax.
3. Anomaly Score:
   - S(x) = P(Unknown) = hat{P}_6(x).
   - Higher score indicates greater likelihood of being an unknown attack.
4. Known Confidence:
   - C_known(x) = 1.0 - S(x) = sum_{c=0}^5 hat{P}_c(x).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import weibull_min

from src.models.base_model import BaseNIDSModel
from src.openset.base_detector import BaseOODDetector

logger = logging.getLogger(__name__)


class OpenMaxDetector(BaseOODDetector):
    """OpenMax EVT-based Out-of-Distribution Detector.

    Adapted for tree-based classifiers using unnormalized margin vectors.
    """

    def __init__(
        self,
        alpha: int = 3,
        eta_ratio: float = 0.2,
        min_eta: int = 5,
        max_eta: int = 20,
        distance_metric: str = "euclidean",
    ):
        """Initializes OpenMax detector hyperparameters.

        Args:
            alpha: Number of top-ranked activation classes to recalibrate (default=3).
            eta_ratio: Fraction of correctly classified samples used for tail sizing (default=0.2).
            min_eta: Minimum number of tail distances for Weibull fitting (default=5).
            max_eta: Maximum number of tail distances for Weibull fitting (default=20).
            distance_metric: Distance metric used to measure separation from MAV (default="euclidean").
        """
        super().__init__(f"OpenMax (alpha={alpha})")
        self.alpha = int(alpha)
        self.eta_ratio = float(eta_ratio)
        self.min_eta = int(min_eta)
        self.max_eta = int(max_eta)
        self.distance_metric = distance_metric.lower()

        if self.distance_metric != "euclidean":
            raise ValueError(f"Unsupported distance metric '{distance_metric}'. Only 'euclidean' is currently supported.")

        # Fitted parameters (populated during fit on D_evt_fit)
        self.mavs_: Dict[int, np.ndarray] = {}  # class_id -> MAV in R^K
        self.weibull_params_: Dict[int, Dict[str, Any]] = {}  # class_id -> {shape, loc, scale, eta, ...}
        self.n_classes_: int = 6

        # Fast vectorized matrix representations
        self._mavs_matrix: Optional[np.ndarray] = None      # (K, K)
        self._shapes_array: Optional[np.ndarray] = None     # (K,)
        self._locs_array: Optional[np.ndarray] = None       # (K,)
        self._scales_array: Optional[np.ndarray] = None     # (K,)

    def fit(self, X_evt: np.ndarray, y_evt: np.ndarray, model: BaseNIDSModel) -> "OpenMaxDetector":
        """Fits Mean Activation Vectors (MAVs) and Weibull upper-tail models on D_evt_fit.

        Args:
            X_evt: Preprocessed feature matrix of D_evt_fit (12,138 samples).
            y_evt: Ground truth multiclass known labels (0..5).
            model: Trained 6-class base classifier.

        Returns:
            Fitted OpenMaxDetector instance.
        """
        if X_evt is None or y_evt is None or model is None:
            raise ValueError("X_evt, y_evt, and model must not be None when fitting OpenMaxDetector.")

        logits_evt = model.predict_logits(X_evt)
        preds_evt = model.predict(X_evt)

        if logits_evt.ndim != 2:
            raise ValueError(f"Expected 2D logits matrix, got shape {logits_evt.shape}")
        if not np.all(np.isfinite(logits_evt)):
            raise ValueError("Non-finite values encountered in raw logits returned by model.predict_logits()")

        self.n_classes_ = logits_evt.shape[1]
        correct_mask = (preds_evt == y_evt)

        self.mavs_ = {}
        self.weibull_params_ = {}

        for c in range(self.n_classes_):
            # 1. Identify correctly classified samples for class c
            class_correct = (y_evt == c) & correct_mask
            n_c = int(class_correct.sum())
            fallback_applied = False
            fallback_reason = "none"

            if n_c == 0:
                # Deterministic Fallback 1: Fallback to all samples of class c in D_evt_fit
                logger.warning(
                    "Class %d has 0 correctly classified samples in D_evt_fit. Falling back to all class samples.", c
                )
                class_samples = (y_evt == c)
                n_c = int(class_samples.sum())
                fallback_applied = True
                fallback_reason = "zero_correct_samples_fallback_to_all_class_samples"
                if n_c == 0:
                    raise ValueError(f"Class {c} has 0 samples in D_evt_fit!")
                c_logits = logits_evt[class_samples]
            else:
                c_logits = logits_evt[class_correct]

            # 2. Compute Mean Activation Vector (MAV)
            mav_c = np.mean(c_logits, axis=0)
            self.mavs_[c] = mav_c

            # 3. Compute Euclidean distances to MAV
            dists = np.linalg.norm(c_logits - mav_c, axis=1)

            # 4. Determine tail size eta_c
            eta_c = min(self.max_eta, max(self.min_eta, int(np.floor(self.eta_ratio * n_c))))
            if n_c < self.min_eta:
                eta_c = n_c

            # 5. Extract largest eta_c distances
            sorted_dists = np.sort(dists)
            tail_c = sorted_dists[-eta_c:]

            # 6. Fit Weibull distribution with floc fixed to min(tail_c)
            floc_val = float(np.min(tail_c))
            tail_span = float(np.max(tail_c) - floc_val)

            if tail_span == 0.0 or len(tail_c) < 2:
                # Deterministic Fallback 2: Zero variance in tail
                shape = 1.0
                loc = floc_val
                scale = max(float(np.std(tail_c)), 1e-4)
                fallback_applied = True
                fallback_reason = "zero_variance_tail_fallback"
                logger.warning("Class %d tail distances have zero variance. Using deterministic fallback.", c)
            else:
                try:
                    shape_fit, loc_fit, scale_fit = weibull_min.fit(tail_c, floc=floc_val)
                    if (
                        np.isfinite(shape_fit)
                        and np.isfinite(loc_fit)
                        and np.isfinite(scale_fit)
                        and scale_fit > 0.0
                        and shape_fit > 0.0
                    ):
                        shape = float(shape_fit)
                        loc = float(loc_fit)
                        scale = float(scale_fit)
                    else:
                        shape = 1.0
                        loc = floc_val
                        scale = max(float(np.std(tail_c)), 1e-4)
                        fallback_applied = True
                        fallback_reason = "non_finite_weibull_fit_parameters"
                        logger.warning("Class %d Weibull fit yielded non-finite/non-positive parameters.", c)
                except Exception as exc:
                    shape = 1.0
                    loc = floc_val
                    scale = max(float(np.std(tail_c)), 1e-4)
                    fallback_applied = True
                    fallback_reason = f"weibull_fit_exception_{type(exc).__name__}"
                    logger.warning("Class %d Weibull fit failed (%s). Using deterministic fallback.", c, exc)

            self.weibull_params_[c] = {
                "class_id": c,
                "n_correct": n_c,
                "eta": eta_c,
                "shape": shape,
                "loc": loc,
                "scale": scale,
                "floc": floc_val,
                "fallback_applied": fallback_applied,
                "fallback_reason": fallback_reason,
                "tail_min": floc_val,
                "tail_max": float(np.max(tail_c)),
                "tail_mean": float(np.mean(tail_c)),
            }

            logger.info(
                "Class %d EVT Tail: n=%d, eta=%d, shape=%.4f, loc=%.4f, scale=%.4f (fallback=%s)",
                c,
                n_c,
                eta_c,
                shape,
                loc,
                scale,
                fallback_reason,
            )

        # Build fast vectorized cache matrices
        self._mavs_matrix = np.array([self.mavs_[c] for c in range(self.n_classes_)], dtype=np.float64)  # (K, K)
        self._shapes_array = np.array([self.weibull_params_[c]["shape"] for c in range(self.n_classes_)], dtype=np.float64)
        self._locs_array = np.array([self.weibull_params_[c]["loc"] for c in range(self.n_classes_)], dtype=np.float64)
        self._scales_array = np.array([self.weibull_params_[c]["scale"] for c in range(self.n_classes_)], dtype=np.float64)

        self.is_fitted = True
        return self

    def predict_proba_7class(self, X: np.ndarray, model: BaseNIDSModel) -> np.ndarray:
        """Computes 7-class OpenMax probabilities with top-alpha recalibration and unknown logit synthesis.

        Vectorized over all N samples for efficiency.

        Returns:
            2D array of shape (N, 7) containing valid probability distributions
            [P_0, P_1, P_2, P_3, P_4, P_5, P_unknown] where sum_j P_j = 1.0.
        """
        if not self.is_fitted:
            raise RuntimeError("OpenMaxDetector must be fitted on D_evt_fit before calling predict_proba_7class().")

        logits = model.predict_logits(X)
        if logits.ndim != 2 or logits.shape[1] != self.n_classes_:
            raise ValueError(f"Expected logits of shape (N, {self.n_classes_}), got {logits.shape}")

        N = logits.shape[0]
        K = self.n_classes_
        alpha = min(self.alpha, K)

        # 1. Compute Euclidean distance from each sample's activation vector to all K MAVs
        # dists shape: (N, K)
        dists = np.linalg.norm(logits[:, np.newaxis, :] - self._mavs_matrix[np.newaxis, :, :], axis=2)

        # 2. Evaluate analytic Weibull CDF: omega_{i, c} in [0, 1]
        # For d <= loc, CDF = 0.0; for d > loc, CDF = 1.0 - exp(-((d - loc)/scale)^shape)
        diff = np.maximum(0.0, dists - self._locs_array[np.newaxis, :])
        safe_scales = np.maximum(self._scales_array[np.newaxis, :], 1e-12)
        exponent = (diff / safe_scales) ** self._shapes_array[np.newaxis, :]
        # Clip exponent to avoid overflow in np.exp
        exponent = np.clip(exponent, 0.0, 700.0)
        omega = np.where(dists <= self._locs_array[np.newaxis, :], 0.0, 1.0 - np.exp(-exponent))

        # 3. Rank classes by activation margin in descending order
        # ranked shape: (N, K)
        ranked = np.argsort(logits, axis=1)[:, ::-1]

        # 4. Top-alpha recalibration
        weights = np.ones((N, K), dtype=np.float64)
        unknown_activations = np.zeros(N, dtype=np.float64)
        sample_indices = np.arange(N)

        for r in range(alpha):
            c_r = ranked[:, r]  # Class ID with rank (r+1) for each sample
            # Rank weighting factor: (alpha - r) / alpha for 0-indexed r
            rank_factor = (alpha - r) / alpha
            omega_r = omega[sample_indices, c_r]
            w_r = 1.0 - rank_factor * omega_r
            weights[sample_indices, c_r] = w_r

            # Accumulate discounted activation into the unknown logit
            z_r = logits[sample_indices, c_r]
            unknown_activations += z_r * (1.0 - w_r)

        # Recalibrated known class logits
        z_recalibrated = logits * weights

        # Append synthesized unknown logit at index 6 (7-class representation)
        z_7 = np.column_stack([z_recalibrated, unknown_activations])

        # 5. Numerically stable softmax
        z_max = np.max(z_7, axis=1, keepdims=True)
        exp_z = np.exp(z_7 - z_max)
        sum_exp = np.sum(exp_z, axis=1, keepdims=True)
        probs_7 = exp_z / np.maximum(sum_exp, 1e-12)

        return probs_7

    def compute_anomaly_score(self, X: np.ndarray, model: BaseNIDSModel) -> np.ndarray:
        """Computes OOD anomaly score as the OpenMax synthesized Unknown probability P_6(x).

        Convention: Higher score indicates greater likelihood of being Unknown.

        Returns:
            1D array of shape (N,) with float anomaly scores in [0.0, 1.0].
        """
        probs_7 = self.predict_proba_7class(X, model)
        return probs_7[:, 6]

    def compute_confidence_score(self, X: np.ndarray, model: BaseNIDSModel) -> np.ndarray:
        """Computes Known-confidence score: C_known(x) = 1.0 - P_6(x) = sum_{c=0}^5 P_c(x).

        Convention: Higher score indicates greater confidence that the sample is Known.

        Returns:
            1D array of shape (N,) with float confidence scores in [0.0, 1.0].
        """
        return 1.0 - self.compute_anomaly_score(X, model)

    def predict_openset(
        self,
        X: np.ndarray,
        model: BaseNIDSModel,
        threshold: float,
        unknown_label: int = 6,
        use_recalibrated_argmax: bool = True,
    ) -> np.ndarray:
        """Predicts 7-class open-set labels using calibrated rejection threshold.

        Decision Rule:
            If S(x) = P(Unknown) > threshold:
                predict unknown_label (6)
            Else:
                predict recalibrated argmax class in {0..5} (or base classifier prediction)

        Args:
            X: Input feature matrix.
            model: Fitted 6-class classifier.
            threshold: Calibrated rejection threshold.
            unknown_label: Label assigned to rejected/unknown samples (default=6).
            use_recalibrated_argmax: If True, uses argmax of OpenMax recalibrated known probabilities;
                                     if False, uses base classifier predict() for unrejected samples.

        Returns:
            1D array of open-set class predictions in {0, 1, 2, 3, 4, 5, 6}.
        """
        probs_7 = self.predict_proba_7class(X, model)
        unknown_scores = probs_7[:, 6]

        if use_recalibrated_argmax:
            known_preds = np.argmax(probs_7[:, :self.n_classes_], axis=1)
        else:
            known_preds = model.predict(X)

        openset_preds = np.copy(known_preds)
        is_rejected = (unknown_scores > threshold)
        openset_preds[is_rejected] = unknown_label
        return openset_preds

    def get_fitted_params(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary of fitted MAVs and Weibull tail parameters."""
        if not self.is_fitted:
            raise RuntimeError("Detector is not fitted yet.")

        return {
            "method": "OpenMax (EVT-Based Adaptation)",
            "adaptation_note": "LightGBM raw margins used as Class Activation Vectors (adaptation of canonical CNN OpenMax)",
            "hyperparameters": {
                "alpha": self.alpha,
                "eta_ratio": self.eta_ratio,
                "min_eta": self.min_eta,
                "max_eta": self.max_eta,
                "distance_metric": self.distance_metric,
            },
            "mean_activation_vectors": {
                int(c): [float(v) for v in self.mavs_[c]] for c in range(self.n_classes_)
            },
            "weibull_tail_models": {
                int(c): self.weibull_params_[c] for c in range(self.n_classes_)
            },
        }
