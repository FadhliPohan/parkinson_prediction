from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.datasets.registry import DatasetConfig

COMMON_BOOL_FLAGS = {
    "disable_cpu_fallback",
    "mixed_precision",
    "no_pretrained",
    "disable_augmentation",
}


def normalized_flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def build_cli_args(params: Dict[str, Any], framework: str) -> List[str]:
    args: List[str] = []
    skip_keys = set()

    if framework not in {"yolo", "pytorch"}:
        skip_keys.add("yolo_size")

    for key, value in params.items():
        if key in skip_keys:
            continue

        flag = normalized_flag_name(key)
        if key in COMMON_BOOL_FLAGS:
            if bool(value):
                args.append(flag)
            continue

        if value is None:
            continue

        args.extend([flag, str(value)])

    return args


def build_context_args(
    dataset_cfg: DatasetConfig,
    method_id: str,
    report_root: Path,
    models_root: Path,
    augmentation_id: str,
    augmentation_label: str,
    experiment_id: str,
) -> List[str]:
    args = [
        "--dataset-dir",
        str(dataset_cfg.split_path),
        "--dataset-name",
        dataset_cfg.dataset_id,
        "--training-method",
        method_id,
        "--report-root",
        str(report_root),
        "--models-root",
        str(models_root),
    ]
    if augmentation_id:
        args.extend(["--augmentation-id", str(augmentation_id)])
    if augmentation_label:
        args.extend(["--augmentation-label", str(augmentation_label)])
    if experiment_id:
        args.extend(["--experiment-id", str(experiment_id)])
    return args
