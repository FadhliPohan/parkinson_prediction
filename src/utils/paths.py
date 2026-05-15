from __future__ import annotations

from pathlib import Path
from typing import Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASET_ROOT = PROJECT_ROOT / "dataset"
REPORT_ROOT = PROJECT_ROOT / "report"
TRAINED_MODELS_ROOT = PROJECT_ROOT / "trained_models"


def resolve_project_path(raw_path: Union[str, Path]) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def ensure_dir(path: Union[str, Path]) -> Path:
    target = resolve_project_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
