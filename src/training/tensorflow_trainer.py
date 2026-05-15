from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict

from src.datasets.registry import DatasetConfig
from src.models.registry import ModelConfig
from src.training.cli_args import build_cli_args, build_context_args
from src.utils.paths import PROJECT_ROOT
from src.utils.runtime import build_runtime_env, get_runtime_python


def run_tensorflow_training(
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

    cli_args = build_cli_args(training_params, framework="tensorflow")
    cli_args.extend(
        build_context_args(
            dataset_cfg=dataset_cfg,
            method_id=method_id,
            report_root=report_root,
            models_root=models_root,
        )
    )

    command = [str(get_runtime_python()), str(script_path)] + cli_args
    print("\nMenjalankan:", " ".join(command))
    result = subprocess.run(command, env=build_runtime_env(), cwd=str(PROJECT_ROOT))
    return int(result.returncode)
