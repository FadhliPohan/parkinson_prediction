import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


REPORT_VERSION = 2


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "run"


def build_run_name(model_name: str, augmentation_tag: str, timestamp: Optional[datetime] = None) -> str:
    effective_time = timestamp or datetime.now()
    return "{}_{}_{}".format(
        sanitize_name(model_name),
        sanitize_name(augmentation_tag),
        effective_time.strftime("%Y%m%d_%H%M%S"),
    )


def ensure_run_directories(project_root: Path, model_name: str, run_name: str) -> Dict[str, Path]:
    report_root = (project_root / "report" / model_name).resolve()
    model_root = (project_root / "trained_models" / model_name).resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    report_dir = report_root / run_name
    model_dir = model_root / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "report_root": report_root,
        "model_root": model_root,
        "report_dir": report_dir,
        "model_dir": model_dir,
    }


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    with open(str(path), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        with open(str(path), "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def write_run_metadata(report_dir: Path, model_dir: Path, metadata: Dict[str, object]) -> None:
    payload = dict(metadata)
    payload["report_version"] = REPORT_VERSION
    dump_json(report_dir / "run_metadata.json", payload)
    dump_json(model_dir / "run_metadata.json", payload)


def write_experiment_summary(
    report_dir: Path,
    metadata: Dict[str, object],
    metrics: Dict[str, object],
) -> None:
    payload = {
        "report_version": REPORT_VERSION,
        "run_name": metadata.get("run_name"),
        "model_name": metadata.get("model_name"),
        "model_family": metadata.get("model_family"),
        "augmentation_profile": metadata.get("augmentation_profile"),
        "augmentation_display_name": metadata.get("augmentation_display_name"),
        "feature_extractor": metadata.get("feature_extractor"),
        "classifier_name": metadata.get("classifier_name"),
        "metrics": metrics,
    }
    dump_json(report_dir / "experiment_summary.json", payload)


def update_latest_run_pointer(root_dir: Path, run_name: str) -> None:
    with open(str(root_dir / "latest_run.txt"), "w", encoding="utf-8") as fp:
        fp.write(str(run_name))
