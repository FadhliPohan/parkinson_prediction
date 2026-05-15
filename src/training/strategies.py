from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.utils.config import deep_merge, load_default_training_config, load_methods_config


@dataclass
class TrainingMethod:
    method_id: str
    description: str
    arg_overrides: Dict[str, Any]


class TrainingMethodRegistry:
    def __init__(self):
        cfg = load_methods_config()
        self._default_method = cfg.get("defaults", {}).get("default_method")
        raw_methods = cfg.get("methods", {})
        if not isinstance(raw_methods, dict) or not raw_methods:
            raise ValueError("Konfigurasi training methods kosong")

        self._methods: Dict[str, TrainingMethod] = {}
        for method_id, item in raw_methods.items():
            self._methods[method_id] = TrainingMethod(
                method_id=method_id,
                description=str(item.get("description", "")),
                arg_overrides=dict(item.get("arg_overrides", {})),
            )

        if self._default_method not in self._methods:
            self._default_method = sorted(self._methods.keys())[0]

    @property
    def default_method(self) -> str:
        return str(self._default_method)

    def list_method_ids(self) -> List[str]:
        return sorted(self._methods.keys())

    def get(self, method_id: str) -> TrainingMethod:
        if method_id not in self._methods:
            available = ", ".join(self.list_method_ids())
            raise KeyError(f"Method '{method_id}' tidak ditemukan. Tersedia: {available}")
        return self._methods[method_id]


def build_training_params(method: TrainingMethod, user_overrides: Dict[str, Any]) -> Dict[str, Any]:
    base_cfg = load_default_training_config().get("default_training", {})
    if not isinstance(base_cfg, dict):
        raise ValueError("default_training pada default_training.yaml harus object")

    merged = deep_merge(base_cfg, method.arg_overrides)
    merged = deep_merge(merged, user_overrides)
    return merged
