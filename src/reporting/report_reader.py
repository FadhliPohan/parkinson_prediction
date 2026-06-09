from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _fallback_augmentation_id(manifest: Dict[str, Any]) -> str:
    training = manifest.get("training", {}) if isinstance(manifest, dict) else {}
    if not isinstance(training, dict):
        return "augment_on_the_fly"

    augmentation_id = str(training.get("augmentation_id", "")).strip()
    if augmentation_id:
        return augmentation_id

    params = training.get("parameters", {})
    if isinstance(params, dict):
        if bool(params.get("disable_augmentation", False)):
            return "no_augment"
    return "augment_on_the_fly"


def _fallback_augmentation_label(augmentation_id: str) -> str:
    if augmentation_id == "no_augment":
        return "tanpa_augmentasi"
    return "augmentasi_on_the_fly"


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _build_record_from_manifest(run_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    metrics = safe_load_json(run_dir / EVALUATION_METRICS_JSON_FILE) or {}

    dataset = manifest.get("dataset", {}) if isinstance(manifest, dict) else {}
    model = manifest.get("model", {}) if isinstance(manifest, dict) else {}
    training = manifest.get("training", {}) if isinstance(manifest, dict) else {}
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    params = training.get("parameters", {}) if isinstance(training, dict) else {}
    if not isinstance(params, dict):
        params = {}

    dataset_name = str(dataset.get("dataset_name", "unknown_dataset"))
    model_name = str(model.get("model_name", run_dir.parent.name))
    method_id = str(training.get("method", "unknown_method"))
    split_preset = str(dataset.get("split_preset") or "unknown")
    split_ratio = dataset.get("split_ratio") if isinstance(dataset, dict) else None
    optimizer_name = str(params.get("optimizer") or "unknown")

    augmentation_id = _fallback_augmentation_id(manifest)
    augmentation_label = str(training.get("augmentation_label", "")).strip() or _fallback_augmentation_label(augmentation_id)

    run_id = str(manifest.get("run_id", run_dir.name))

    return {
        "experiment_id": str(manifest.get("experiment_id") or training.get("experiment_id") or ""),
        "dataset": dataset_name,
        "split_preset": split_preset,
        "split_ratio": split_ratio,
        "optimizer": optimizer_name,
        "augmentation": augmentation_id,
        "augmentation_label": augmentation_label,
        "method": method_id,
        "model": model_name,
        "run_id": run_id,
        "run_started_at": manifest.get("run_started_at"),
        "run_finished_at": manifest.get("run_finished_at"),
        "run_dir": str(run_dir),
        "manifest": manifest,
        "metrics": metrics,
        "epochs": _to_int_or_none(params.get("epochs")),
        "batch_size": _to_int_or_none(params.get("batch_size")),
        "fine_tune_epochs": _to_int_or_none(params.get("fine_tune_epochs")),
        "training_parameters": params,
        "accuracy": _to_float_or_none(metrics.get("accuracy")),
        "f1_score": _to_float_or_none(metrics.get("f1_score")),
        "roc_auc": _to_float_or_none(metrics.get("roc_auc")),
        "train_accuracy": _to_float_or_none(metrics.get("train_accuracy")),
        "val_accuracy": _to_float_or_none(metrics.get("val_accuracy")),
        "train_loss": _to_float_or_none(metrics.get("train_loss")),
        "val_loss": _to_float_or_none(metrics.get("val_loss")),
        "training_time_seconds": _to_float_or_none(metrics.get("training_time_seconds"))
        or _to_float_or_none(training.get("duration_seconds")),
        "test_samples": metrics.get("test_samples"),
        "best_model_path": artifacts.get("best_model_path"),
        "final_model_path": artifacts.get("final_model_path"),
        "report_dir": artifacts.get("report_dir", str(run_dir)),
        "model_dir": artifacts.get("model_dir"),
    }


def _sort_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: (
            str(row.get("run_started_at") or ""),
            str(row.get("run_id") or ""),
        ),
        reverse=True,
    )


def build_experiment_index(report_root: Path) -> List[Dict[str, Any]]:
    if not report_root.exists() or not report_root.is_dir():
        return []

    records: List[Dict[str, Any]] = []
    for manifest_path in report_root.rglob(RUN_MANIFEST_FILE):
        run_dir = manifest_path.parent
        manifest = safe_load_json(manifest_path)
        if not isinstance(manifest, dict):
            continue
        records.append(_build_record_from_manifest(run_dir, manifest))

    return _sort_records(records)


def list_datasets(report_root: Path) -> List[str]:
    records = build_experiment_index(report_root)
    return sorted(set(str(item.get("dataset")) for item in records if item.get("dataset")))


def list_augmentations(report_root: Path, dataset_name: str) -> List[str]:
    records = build_experiment_index(report_root)
    return sorted(
        set(
            str(item.get("augmentation"))
            for item in records
            if item.get("dataset") == dataset_name and item.get("augmentation")
        )
    )


def list_methods(report_root: Path, dataset_name: str, augmentation_id: str) -> List[str]:
    records = build_experiment_index(report_root)
    return sorted(
        set(
            str(item.get("method"))
            for item in records
            if item.get("dataset") == dataset_name
            and item.get("augmentation") == augmentation_id
            and item.get("method")
        )
    )


def list_models(
    report_root: Path,
    dataset_name: str,
    augmentation_id: Optional[str] = None,
    method_id: Optional[str] = None,
) -> List[str]:
    records = build_experiment_index(report_root)
    return sorted(
        set(
            str(item.get("model"))
            for item in records
            if item.get("dataset") == dataset_name
            and (augmentation_id is None or item.get("augmentation") == augmentation_id)
            and (method_id is None or item.get("method") == method_id)
            and item.get("model")
        )
    )


def list_runs(
    report_root: Path,
    dataset_name: str,
    model_name: str,
    augmentation_id: Optional[str] = None,
    method_id: Optional[str] = None,
) -> List[str]:
    records = build_experiment_index(report_root)
    filtered = [
        item
        for item in records
        if item.get("dataset") == dataset_name
        and item.get("model") == model_name
        and (augmentation_id is None or item.get("augmentation") == augmentation_id)
        and (method_id is None or item.get("method") == method_id)
    ]
    return [str(item.get("run_id")) for item in _sort_records(filtered) if item.get("run_id")]


def read_latest_run_id(model_root: Path) -> Optional[str]:
    latest_file = model_root / "latest_run.txt"
    if latest_file.exists():
        run_id = latest_file.read_text(encoding="utf-8").strip()
        if run_id and (model_root / run_id).exists():
            return run_id

    runs = sorted([p.name for p in model_root.iterdir() if p.is_dir()], reverse=True) if model_root.exists() else []
    return runs[0] if runs else None


def find_run_record(
    report_root: Path,
    dataset_name: str,
    augmentation_id: str,
    method_id: str,
    model_name: str,
    run_id: str,
) -> Optional[Dict[str, Any]]:
    records = build_experiment_index(report_root)
    for item in records:
        if (
            item.get("dataset") == dataset_name
            and item.get("augmentation") == augmentation_id
            and item.get("method") == method_id
            and item.get("model") == model_name
            and item.get("run_id") == run_id
        ):
            return item
    return None


def build_latest_summary_table(report_root: Path) -> List[Dict[str, object]]:
    records = build_experiment_index(report_root)
    latest_by_key: Dict[str, Dict[str, Any]] = {}

    for record in records:
        key = "{}|{}|{}|{}|{}".format(
            record.get("dataset"),
            record.get("split_preset"),
            record.get("augmentation"),
            record.get("method"),
            record.get("model"),
        )
        current = latest_by_key.get(key)
        if current is None:
            latest_by_key[key] = record
            continue

        current_started = str(current.get("run_started_at") or "")
        incoming_started = str(record.get("run_started_at") or "")
        if incoming_started > current_started:
            latest_by_key[key] = record

    rows: List[Dict[str, object]] = []
    for record in latest_by_key.values():
        rows.append(
            {
                "dataset": record.get("dataset"),
                "split_preset": record.get("split_preset"),
                "optimizer": record.get("optimizer"),
                "augmentation": record.get("augmentation"),
                "method": record.get("method"),
                "model": record.get("model"),
                "run_id": record.get("run_id"),
                "epochs": record.get("epochs"),
                "batch_size": record.get("batch_size"),
                "fine_tune_epochs": record.get("fine_tune_epochs"),
                "train_accuracy": record.get("train_accuracy"),
                "val_accuracy": record.get("val_accuracy"),
                "train_loss": record.get("train_loss"),
                "val_loss": record.get("val_loss"),
                "test_accuracy": record.get("accuracy"),
                "f1_score": record.get("f1_score"),
                "training_time_seconds": record.get("training_time_seconds"),
            }
        )

    return _sort_records(rows)
