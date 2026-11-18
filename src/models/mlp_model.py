"""Multi-Layer Perceptron (MLP) wrapper for XAI-NIDS."""

from typing import List, Optional
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.models.base_model import BaseNIDSModel


class MLPModel(BaseNIDSModel):
    """Multi-Layer Perceptron neural network with pre-softmax logit extraction."""

    def __init__(self, task: str = "multiclass", random_state: int = 42, **hyperparameters):
        super().__init__("mlp", task=task, random_state=random_state, **hyperparameters)
        default_params = {
            "hidden_layer_sizes": (128, 64),
            "max_iter": 150,
            "learning_rate_init": 0.001,
            "random_state": random_state,
            "early_stopping": True,
            "n_iter_no_change": 10,
        }
        default_params.update(hyperparameters)
        if "hidden_layer_sizes" in default_params and isinstance(default_params["hidden_layer_sizes"], (list, tuple)):
            default_params["hidden_layer_sizes"] = tuple(default_params["hidden_layer_sizes"])
        self._underlying_model = MLPClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "MLPModel":
        self._underlying_model.fit(X, y)
        self.classes_ = self._underlying_model.classes_
        self.is_fitted = True
        if feature_names is not None:
            self.feature_names_ = feature_names
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._underlying_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._underlying_model.predict_proba(X)

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Computes exact forward pass activations up to the output layer prior to softmax/sigmoid."""
        activation = X
        act_fn = getattr(self._underlying_model, "activation", "relu")
        # Propagate through hidden layers
        for i in range(len(self._underlying_model.coefs_) - 1):
            w = self._underlying_model.coefs_[i]
            b = self._underlying_model.intercepts_[i]
            z = np.dot(activation, w) + b
            if act_fn == "relu":
                activation = np.maximum(0, z)
            elif act_fn == "tanh":
                activation = np.tanh(z)
            elif act_fn == "logistic":
                activation = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            else:
                activation = z

        # Output layer logits
        w_out = self._underlying_model.coefs_[-1]
        b_out = self._underlying_model.intercepts_[-1]
        logits = np.dot(activation, w_out) + b_out

        if len(self.classes_) == 2 and logits.shape[1] == 1:
            logits = np.hstack([-logits, logits])
        return logits

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Returns input layer weight magnitude proxy for feature importance."""
        if hasattr(self._underlying_model, "coefs_") and len(self._underlying_model.coefs_) > 0:
            return np.mean(np.abs(self._underlying_model.coefs_[0]), axis=1)
        return super().get_feature_importances()
