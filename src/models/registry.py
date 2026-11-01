"""Model Registry and Dynamic Factory for XAI-NIDS.

Enables seamless swapping and instantiation of models via configuration or CLI flags.
"""

import logging
from typing import Any, Dict, List, Optional, Type
from src.models.base_model import BaseNIDSModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry maintaining available NIDS models with alias resolution."""

    _registry: Dict[str, Type[BaseNIDSModel]] = {}
    _aliases: Dict[str, str] = {
        "lr": "logistic_regression",
        "logistic": "logistic_regression",
        "logistic_regression": "logistic_regression",
        "rf": "random_forest",
        "random_forest": "random_forest",
        "randomforest": "random_forest",
        "lgb": "lightgbm",
        "lgbm": "lightgbm",
        "lightgbm": "lightgbm",
        "xgb": "xgboost",
        "xgboost": "xgboost",
        "xgboast": "xgboost",  # Common user typo alias
        "mlp": "mlp",
        "neural_net": "mlp",
    }

    @classmethod
    def register(cls, name: str, model_cls: Type[BaseNIDSModel]) -> None:
        """Registers a new model class under a canonical name."""
        canonical_name = name.lower().strip()
        cls._registry[canonical_name] = model_cls
        logger.debug("Registered model '%s' -> %s", canonical_name, model_cls.__name__)

    @classmethod
    def get_canonical_name(cls, name: str) -> str:
        """Resolves potential aliases or typos to the canonical model identifier."""
        clean_name = name.lower().strip().replace(" ", "_").replace("-", "_")
        return cls._aliases.get(clean_name, clean_name)

    @classmethod
    def create(cls, name: str, task: str = "multiclass", **kwargs) -> BaseNIDSModel:
        """Instantiates a model by name with the given task and hyperparameters."""
        canonical = cls.get_canonical_name(name)
        if canonical not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Model '{name}' (resolved as '{canonical}') is not registered. Available models: {available}"
            )

        model_cls = cls._registry[canonical]
        logger.info("Instantiating model '%s' (class: %s, task: %s)", canonical, model_cls.__name__, task)
        return model_cls(task=task, **kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        """Returns the list of all registered canonical model names."""
        return sorted(list(cls._registry.keys()))
