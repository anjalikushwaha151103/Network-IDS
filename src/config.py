"""Configuration manager for XAI-NIDS."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class Config:
    """Loads and provides access to hierarchical system configuration."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to configs/default_config.yaml relative to project root
            base_dir = Path(__file__).resolve().parent.parent
            config_path = base_dir / "configs" / "default_config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._raw_config: Dict[str, Any] = yaml.safe_load(f)

        self.project_root = Path(__file__).resolve().parent.parent

    @property
    def dataset(self) -> Dict[str, Any]:
        return self._raw_config.get("dataset", {})

    @property
    def preprocessing(self) -> Dict[str, Any]:
        return self._raw_config.get("preprocessing", {})

    @property
    def feature_selection(self) -> Dict[str, Any]:
        return self._raw_config.get("feature_selection", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self._raw_config.get("model", {})

    @property
    def openset(self) -> Dict[str, Any]:
        return self._raw_config.get("openset", {})

    @property
    def xai(self) -> Dict[str, Any]:
        return self._raw_config.get("xai", {})

    @property
    def server(self) -> Dict[str, Any]:
        return self._raw_config.get("server", {})

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        sec = self._raw_config.get(section, {})
        if key is None:
            return sec
        return sec.get(key, default)

    def resolve_path(self, relative_path: str) -> Path:
        """Resolves a relative path against the project root."""
        return self.project_root / relative_path


# Global default configuration instance
default_config = Config()
