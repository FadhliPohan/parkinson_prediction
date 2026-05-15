from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.datasets.registry import DatasetConfig
from src.models.registry import ModelConfig
from src.utils.paths import PROJECT_ROOT
from src.utils.runtime import build_runtime_env, get_runtime_python


COMMON_BOOL_FLAGS = {
    "disable_cpu_fallback",
    "mixed_precision",
    "no_pretrained",
}


def _normalized_flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _build_cli_args(params: Dict[str, Any], framework: str) -> List[str]:
    args: List[str] = []
    skip_keys = set()

    if framework != "yolo":
        skip_keys.add("yolo_size")

    for key, value in params.items():
        if key in skip_keys:
            continue

        flag = _normalized_flag_name(key)
        if key in COMMON_BOOL_FLAGS:
            if bool(value):
                args.append(flag)
            continue

        if value is None:
            continue

        args.extend([flag, str(value)])

    return args


def run_model_training(
    model_cfg: ModelConfig,
    dataset_cfg: DatasetConfig,
    method_id: str,
    training_params: Dict[str, Any],
    report_root: Path,
    models_root: Path,
) -> int:
    script_path = model_cfg.script_abs_path
    if not script_path.exists():
        raise FileNotFoundError(f"Script model tidak ditemukan: {script_path}")

    cli_args = _build_cli_args(training_params, framework=model_cfg.framework)
    cli_args.extend(
        [
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
    )

    command = [str(get_runtime_python()), str(script_path)] + cli_args
    print("\nMenjalankan:", " ".join(command))
    result = subprocess.run(command, env=build_runtime_env(), cwd=str(PROJECT_ROOT))
    return int(result.returncode)


def run_training_jobs(
    models: Iterable[ModelConfig],
    dataset_cfg: DatasetConfig,
    method_id: str,
    training_params: Dict[str, Any],
    report_root: Path,
    models_root: Path,
    stop_on_error: bool = True,
) -> Dict[str, int]:
    status_map: Dict[str, int] = {}
    for model_cfg in models:
        rc = run_model_training(
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            method_id=method_id,
            training_params=training_params,
            report_root=report_root,
            models_root=models_root,
        )
        status_map[model_cfg.model_id] = rc
        if rc != 0 and stop_on_error:
            break
    return status_map


def pick_failed_models(status_map: Dict[str, int]) -> List[str]:
    return [model_id for model_id, code in status_map.items() if code != 0]
