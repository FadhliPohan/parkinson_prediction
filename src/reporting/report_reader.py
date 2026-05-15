from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import EVALUATION_METRICS_JSON_FILE, RUN_MANIFEST_FILE


def safe_load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def list_datasets(report_root: Path) -> List[str]:
    if not report_root.exists() or not report_root.is_dir():
        return []
    return sorted([p.name for p in report_root.iterdir() if p.is_dir()])


def list_models(report_root: Path, dataset_name: str) -> List[str]:
    dataset_root = report_root / dataset_name
    if not dataset_root.exists() or not dataset_root.is_dir():
        return []
    return sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])


def list_runs(report_root: Path, dataset_name: str, model_name: str) -> List[str]:
    model_root = report_root / dataset_name / model_name
    if not model_root.exists() or not model_root.is_dir():
        return []
    return sorted([p.name for p in model_root.iterdir() if p.is_dir()], reverse=True)


def read_latest_run_id(model_root: Path) -> Optional[str]:
    latest_file = model_root / "latest_run.txt"
    if latest_file.exists():
        run_id = latest_file.read_text(encoding="utf-8").strip()
        if run_id and (model_root / run_id).exists():
            return run_id

    runs = sorted([p.name for p in model_root.iterdir() if p.is_dir()], reverse=True) if model_root.exists() else []
    return runs[0] if runs else None


def build_latest_summary_table(report_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset_name in list_datasets(report_root):
        for model_name in list_models(report_root, dataset_name):
            model_root = report_root / dataset_name / model_name
            run_id = read_latest_run_id(model_root)
            if not run_id:
                continue

            run_dir = model_root / run_id
            manifest = safe_load_json(run_dir / RUN_MANIFEST_FILE) or {}
            metrics = safe_load_json(run_dir / EVALUATION_METRICS_JSON_FILE) or {}
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "run_id": run_id,
                    "training_method": manifest.get("training_method"),
                    "accuracy": metrics.get("accuracy"),
                    "f1_score": metrics.get("f1_score"),
                    "roc_auc": metrics.get("roc_auc"),
                    "test_samples": metrics.get("test_samples"),
                }
            )
    return rows
