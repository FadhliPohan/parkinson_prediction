from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .paths import CONFIGS_DIR


class ConfigError(RuntimeError):
    pass


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Format YAML harus object/dict: {path}")
    return data


def load_datasets_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return _load_yaml(path or (CONFIGS_DIR / "datasets.yaml"))


def load_models_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return _load_yaml(path or (CONFIGS_DIR / "models.yaml"))


def load_methods_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return _load_yaml(path or (CONFIGS_DIR / "training_methods.yaml"))


def load_augmentations_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return _load_yaml(path or (CONFIGS_DIR / "augmentations.yaml"))


def load_default_training_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return _load_yaml(path or (CONFIGS_DIR / "default_training.yaml"))


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
