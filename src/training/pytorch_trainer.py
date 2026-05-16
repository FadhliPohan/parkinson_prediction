from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.datasets.registry import DatasetConfig
from src.models.registry import ModelConfig
from src.training.yolo_trainer import run_yolo_training


def run_pytorch_training(
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
    # Saat ini workflow PyTorch di project ini menggunakan runner YOLO classification.
    return run_yolo_training(
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
