from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.datasets.registry import DatasetConfig
from src.models.registry import ModelConfig
from src.training.pytorch_trainer import run_pytorch_training
from src.training.tensorflow_trainer import run_tensorflow_training
from src.training.yolo_trainer import run_yolo_training


FRAMEWORK_RUNNERS = {
    "tensorflow": run_tensorflow_training,
    "yolo": run_yolo_training,
    "pytorch": run_pytorch_training,
}


def run_model_training(
    model_cfg: ModelConfig,
    dataset_cfg: DatasetConfig,
    method_id: str,
    training_params: Dict[str, Any],
    report_root: Path,
    models_root: Path,
    augmentation_id: str = "",
    augmentation_label: str = "",
    experiment_id: str = "",
) -> int:
    framework = str(model_cfg.framework).strip().lower()
    runner = FRAMEWORK_RUNNERS.get(framework)
    if runner is None:
        raise ValueError(f"Framework runner belum didukung: {framework} untuk model {model_cfg.model_id}")

    return int(
        runner(
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            method_id=method_id,
            training_params=training_params,
            report_root=report_root,
            models_root=models_root,
            augmentation_id=augmentation_id,
            augmentation_label=augmentation_label,
            experiment_id=experiment_id,
        )
    )


def run_training_jobs(
    models: Iterable[ModelConfig],
    dataset_cfg: DatasetConfig,
    method_id: str,
    training_params: Dict[str, Any],
    report_root: Path,
    models_root: Path,
    augmentation_id: str = "",
    augmentation_label: str = "",
    experiment_id_map: Dict[str, str] | None = None,
    stop_on_error: bool = True,
) -> Dict[str, int]:
    status_map: Dict[str, int] = {}
    experiment_id_map = experiment_id_map or {}
    for model_cfg in models:
        rc = run_model_training(
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            method_id=method_id,
            training_params=training_params,
            report_root=report_root,
            models_root=models_root,
            augmentation_id=augmentation_id,
            augmentation_label=augmentation_label,
            experiment_id=experiment_id_map.get(model_cfg.model_id, ""),
        )
        status_map[model_cfg.model_id] = rc
        if rc != 0 and stop_on_error:
            break
    return status_map


def pick_failed_models(status_map: Dict[str, int]) -> List[str]:
    return [model_id for model_id, code in status_map.items() if code != 0]
