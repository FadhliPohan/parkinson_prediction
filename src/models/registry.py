from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.utils.config import load_models_config
from src.utils.paths import resolve_project_path


@dataclass
class ModelConfig:
    model_id: str
    display_name: str
    script_path: str
    framework: str
    preprocess_key: str
    enabled: bool = True

    @property
    def script_abs_path(self):
        return resolve_project_path(self.script_path)


class ModelRegistry:
    def __init__(self):
        cfg = load_models_config()
        raw_models = cfg.get("models", {})
        if not isinstance(raw_models, dict) or not raw_models:
            raise ValueError("Konfigurasi model kosong")

        self._models: Dict[str, ModelConfig] = {}
        for model_id, item in raw_models.items():
            model_cfg = ModelConfig(
                model_id=model_id,
                display_name=str(item.get("display_name", model_id)),
                script_path=str(item.get("script_path", "")),
                framework=str(item.get("framework", "tensorflow")),
                preprocess_key=str(item.get("preprocess_key", "")),
                enabled=bool(item.get("enabled", True)),
            )
            self._models[model_id] = model_cfg

    def list_model_ids(self, enabled_only: bool = True) -> List[str]:
        model_ids = []
        for model_id, cfg in self._models.items():
            if enabled_only and not cfg.enabled:
                continue
            model_ids.append(model_id)
        return sorted(model_ids)

    def get(self, model_id: str) -> ModelConfig:
        if model_id not in self._models:
            available = ", ".join(self.list_model_ids(enabled_only=False))
            raise KeyError(f"Model '{model_id}' tidak ditemukan. Tersedia: {available}")
        return self._models[model_id]

    def list_models(self, enabled_only: bool = True) -> List[ModelConfig]:
        return [self._models[mid] for mid in self.list_model_ids(enabled_only=enabled_only)]
