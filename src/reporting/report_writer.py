from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from .schemas import RUN_MANIFEST_FILE, SCHEMA_VERSION


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _sanitize_path_token(value: str, fallback: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value).strip())
    token = token.strip("-")
    return token or fallback


def build_artifact_dirs(
    report_root: Path,
    models_root: Path,
    dataset_name: str,
    augmentation_id: str,
    method_id: str,
    model_name: str,
    run_id: str,
) -> Tuple[Path, Path, Path, Path]:
    dataset_token = _sanitize_path_token(dataset_name, "default_dataset")
    augmentation_token = _sanitize_path_token(augmentation_id, "default_augmentation")
    method_token = _sanitize_path_token(method_id, "default_method")
    model_token = _sanitize_path_token(model_name, "default_model")

    report_dataset_root = report_root / dataset_token / augmentation_token / method_token / model_token
    models_dataset_root = models_root / dataset_token / augmentation_token / method_token / model_token
    report_dataset_root.mkdir(parents=True, exist_ok=True)
    models_dataset_root.mkdir(parents=True, exist_ok=True)

    run_report_dir = report_dataset_root / run_id
    run_model_dir = models_dataset_root / run_id
    run_report_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)
    return report_dataset_root, models_dataset_root, run_report_dir, run_model_dir


def write_latest_run_marker(model_root: Path, run_id: str) -> None:
    with open(model_root / "latest_run.txt", "w", encoding="utf-8") as fh:
        fh.write(run_id)


def write_run_manifest(run_report_dir: Path, manifest: Dict[str, object]) -> Path:
    payload = dict(manifest)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload_path = run_report_dir / RUN_MANIFEST_FILE
    with open(payload_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload_path
