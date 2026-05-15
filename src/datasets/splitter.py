from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .validator import ClassEntry, list_image_files, validate_dataset

try:
    BILINEAR_RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    BILINEAR_RESAMPLE = Image.BILINEAR


def _ensure_valid_ratios(split_cfg: Dict[str, float]) -> Tuple[float, float, float]:
    train = float(split_cfg.get("train", 0.0))
    testing = float(split_cfg.get("testing", 0.0))
    validation = float(split_cfg.get("validation", 0.0))

    total = train + testing + validation
    if total <= 0:
        raise ValueError("Total rasio split harus > 0")

    # Normalisasi otomatis agar robust jika tidak persis 1.0.
    return train / total, testing / total, validation / total


def _parse_resize(value: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"Format resize tidak valid: {value}")
    width = int(value[0])
    height = int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Resize harus > 0: {value}")
    return (width, height)


def _save_image(source_path: Path, target_path: Path, resize_to: Optional[Tuple[int, int]]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if resize_to is None:
        shutil.copy2(str(source_path), str(target_path))
        return

    with Image.open(source_path) as image:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")
        resized = image.resize(resize_to, resample=BILINEAR_RESAMPLE)
        save_kwargs = {}
        if target_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 95
            save_kwargs["subsampling"] = 0
        resized.save(str(target_path), **save_kwargs)


def _compute_split_indices(total_count: int, train_ratio: float, test_ratio: float) -> Tuple[int, int]:
    train_count = int(total_count * train_ratio)
    test_count = int(total_count * test_ratio)
    return train_count, test_count


def split_dataset(
    original_dir: Path,
    split_dir: Path,
    class_mode: str,
    extensions: Sequence[str],
    split_cfg: Dict[str, float],
    resize_cfg: Dict[str, Optional[Sequence[int]]],
    seed: int,
) -> Dict[str, object]:
    class_entries = validate_dataset(
        dataset_root=original_dir,
        class_mode=class_mode,
        extensions=extensions,
    )

    train_ratio, test_ratio, _validation_ratio = _ensure_valid_ratios(split_cfg)
    train_resize = _parse_resize(resize_cfg.get("train"))
    test_resize = _parse_resize(resize_cfg.get("testing"))
    validation_resize = _parse_resize(resize_cfg.get("validation"))

    if split_dir.exists():
        shutil.rmtree(str(split_dir))
    split_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    split_stats: Dict[str, Dict[str, int]] = {
        "train": {},
        "testing": {},
        "validation": {},
    }
    class_manifest: List[Dict[str, object]] = []

    for entry in class_entries:
        files = list_image_files(entry.source_dir, extensions)
        shuffled = files[:]
        rng.shuffle(shuffled)

        train_count, test_count = _compute_split_indices(
            total_count=len(shuffled),
            train_ratio=train_ratio,
            test_ratio=test_ratio,
        )
        train_files = shuffled[:train_count]
        test_files = shuffled[train_count : train_count + test_count]
        validation_files = shuffled[train_count + test_count :]

        class_manifest.append(
            {
                "class_name": entry.class_name,
                "source_dir": str(entry.source_dir),
                "relative_path": entry.relative_path,
                "total_images": len(shuffled),
                "split_counts": {
                    "train": len(train_files),
                    "testing": len(test_files),
                    "validation": len(validation_files),
                },
            }
        )

        split_stats["train"][entry.class_name] = len(train_files)
        split_stats["testing"][entry.class_name] = len(test_files)
        split_stats["validation"][entry.class_name] = len(validation_files)

        for idx, source_path in enumerate(train_files):
            filename = f"{idx:06d}_{source_path.name}"
            target = split_dir / "train" / entry.class_name / filename
            _save_image(source_path, target, train_resize)

        for idx, source_path in enumerate(test_files):
            filename = f"{idx:06d}_{source_path.name}"
            target = split_dir / "testing" / entry.class_name / filename
            _save_image(source_path, target, test_resize)

        for idx, source_path in enumerate(validation_files):
            filename = f"{idx:06d}_{source_path.name}"
            target = split_dir / "validation" / entry.class_name / filename
            _save_image(source_path, target, validation_resize)

    metadata_dir = split_dir / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "1.0.0",
        "seed": int(seed),
        "original_dir": str(original_dir),
        "split_dir": str(split_dir),
        "class_mode": class_mode,
        "split_ratio": {
            "train": train_ratio,
            "testing": test_ratio,
            "validation": 1.0 - train_ratio - test_ratio,
        },
        "classes": class_manifest,
        "split_stats": split_stats,
    }

    manifest_path = metadata_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest
