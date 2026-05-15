from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from src.utils.paths import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "model"
if str(MODEL_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(MODEL_DIR))

try:
    from transformer_backbones import TRANSFORMER_CUSTOM_OBJECTS
except Exception:
    TRANSFORMER_CUSTOM_OBJECTS = {}


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def resolve_model_file(run_dir: Path) -> Path:
    candidates = [
        run_dir / "best_model.keras",
        run_dir / "best_model.pt",
        run_dir / "final_model.keras",
        run_dir / "final_model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    keras = sorted(run_dir.glob("*.keras"))
    if keras:
        return keras[0]

    pt = sorted(run_dir.glob("*.pt"))
    if pt:
        return pt[0]

    raise FileNotFoundError(f"File model tidak ditemukan di {run_dir}")


def load_model_bundle(run_dir: Path) -> Dict[str, object]:
    model_file = resolve_model_file(run_dir)
    class_names_path = run_dir / "class_names.json"
    class_names = _load_json(class_names_path)

    if model_file.suffix.lower() == ".pt":
        if YOLO is None:
            raise ImportError("ultralytics belum tersedia untuk memuat model .pt")
        model = YOLO(str(model_file))
        framework = "yolo"
        if not isinstance(class_names, list) or not class_names:
            yolo_names = getattr(model, "names", None)
            if isinstance(yolo_names, dict):
                class_names = [v for _, v in sorted(yolo_names.items(), key=lambda item: str(item[0]))]
            elif isinstance(yolo_names, list):
                class_names = yolo_names
            else:
                class_names = []
    else:
        model = tf.keras.models.load_model(
            str(model_file),
            custom_objects=TRANSFORMER_CUSTOM_OBJECTS,
            compile=False,
        )
        framework = "tensorflow"

    if not isinstance(class_names, list) or not class_names:
        class_names = ["class_0", "class_1"]

    return {
        "model": model,
        "framework": framework,
        "class_names": class_names,
        "model_file": str(model_file),
    }
