from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .validator import ClassEntry, list_image_files, validate_dataset

try:
    BILINEAR_RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    BILINEAR_RESAMPLE = Image.BILINEAR

ROTATION_MIN_DEGREES = -20.0
ROTATION_MAX_DEGREES = 20.0


@dataclass(frozen=True)
class SplitSample:
    source_path: Path
    is_augmented: bool = False
    rotation_angle: float = 0.0


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


def _serialize_resize(value: Optional[Tuple[int, int]]) -> Optional[List[int]]:
    if value is None:
        return None
    return [int(value[0]), int(value[1])]


def _normalize_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode == "L":
        return image.convert("RGB")
    if image.mode in {"RGBA", "P", "CMYK"}:
        return image.convert("RGB")
    return image.convert("RGB")


def _rotate_image(image: Image.Image, angle: float) -> Image.Image:
    try:
        return image.rotate(
            angle=angle,
            resample=BILINEAR_RESAMPLE,
            expand=False,
            fillcolor=(0, 0, 0),
        )
    except TypeError:
        # Fallback untuk Pillow lama tanpa argumen fillcolor.
        return image.rotate(
            angle=angle,
            resample=BILINEAR_RESAMPLE,
            expand=False,
        )


def _build_sample_filename(index: int, sample: SplitSample) -> str:
    if not sample.is_augmented:
        return f"{index:06d}_{sample.source_path.name}"
    angle_token = "{:+05.1f}".format(sample.rotation_angle)
    angle_token = angle_token.replace("+", "p").replace("-", "m").replace(".", "d")
    return f"{index:06d}_aug_rot_{angle_token}_{sample.source_path.stem}{sample.source_path.suffix.lower()}"


def _save_sample(sample: SplitSample, target_path: Path, resize_to: Optional[Tuple[int, int]]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if resize_to is None and not sample.is_augmented:
        shutil.copy2(str(sample.source_path), str(target_path))
        return

    with Image.open(sample.source_path) as image:
        image = _normalize_to_rgb(image)
        if sample.is_augmented:
            image = _rotate_image(image=image, angle=sample.rotation_angle)
        if resize_to is not None:
            image = image.resize(resize_to, resample=BILINEAR_RESAMPLE)
        save_kwargs = {}
        if target_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 95
            save_kwargs["subsampling"] = 0
        image.save(str(target_path), **save_kwargs)


def _build_balanced_samples(
    class_entries: Sequence[ClassEntry],
    extensions: Sequence[str],
    rng: random.Random,
) -> Tuple[Dict[str, List[SplitSample]], Dict[str, object]]:
    source_files_per_class: Dict[str, List[Path]] = {}
    before_counts: Dict[str, int] = {}
    for entry in class_entries:
        files = list_image_files(entry.source_dir, extensions)
        source_files_per_class[entry.class_name] = files
        before_counts[entry.class_name] = len(files)

    if not before_counts:
        raise ValueError("Tidak ada kelas valid untuk proses split.")

    unique_counts = sorted(set(before_counts.values()))
    is_balanced = len(unique_counts) == 1
    target_per_class = max(before_counts.values())

    samples_per_class: Dict[str, List[SplitSample]] = {}
    generated_per_class: Dict[str, int] = {}

    for entry in class_entries:
        class_name = entry.class_name
        files = source_files_per_class[class_name]
        base_samples = [SplitSample(source_path=path) for path in files]

        missing_count = target_per_class - len(base_samples)
        generated_per_class[class_name] = max(0, missing_count)

        synthetic_samples: List[SplitSample] = []
        if missing_count > 0:
            for _ in range(missing_count):
                source_path = rng.choice(files)
                angle = rng.uniform(ROTATION_MIN_DEGREES, ROTATION_MAX_DEGREES)
                synthetic_samples.append(
                    SplitSample(
                        source_path=source_path,
                        is_augmented=True,
                        rotation_angle=float(angle),
                    )
                )

        samples_per_class[class_name] = base_samples + synthetic_samples

    after_counts = {class_name: len(samples) for class_name, samples in samples_per_class.items()}

    balancing_manifest = {
        "applied": not is_balanced,
        "strategy": "minority_rotation_to_majority_before_split",
        "rotation_range_degrees": [ROTATION_MIN_DEGREES, ROTATION_MAX_DEGREES],
        "class_count": len(before_counts),
        "target_per_class": target_per_class,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "generated_per_class": generated_per_class,
        "total_generated": int(sum(generated_per_class.values())),
    }
    return samples_per_class, balancing_manifest


def _compute_split_counts(
    total_count: int,
    train_ratio: float,
    test_ratio: float,
    validation_ratio: float,
) -> Tuple[int, int, int]:
    # Kebijakan split proyek saat ini:
    # - testing dan validation harus sama besar.
    # - sisa pembulatan selalu masuk ke train.
    if abs(float(test_ratio) - float(validation_ratio)) > 1e-9:
        raise ValueError(
            "Rasio testing dan validation harus sama agar jumlah data split simetris "
            "(contoh: 80:10:10)."
        )

    holdout_count = int(total_count * test_ratio)
    test_count = holdout_count
    validation_count = holdout_count
    train_count = int(total_count) - test_count - validation_count
    if train_count < 0:
        raise ValueError(
            "Konfigurasi split tidak valid: train negatif. "
            "Periksa rasio split pada konfigurasi dataset."
        )
    return train_count, test_count, validation_count


def _is_uniform_count_map(count_map: Dict[str, object]) -> bool:
    if not count_map:
        return False
    values = [int(value) for value in count_map.values()]
    return len(set(values)) == 1


def validate_balanced_split_manifest(manifest: Dict[str, object]) -> Dict[str, object]:
    balancing = manifest.get("class_balancing", {})
    split_stats = manifest.get("split_stats", {})

    if not isinstance(balancing, dict):
        balancing = {}
    if not isinstance(split_stats, dict):
        split_stats = {}

    after_counts = balancing.get("after_counts", {})
    if not isinstance(after_counts, dict):
        after_counts = {}

    train_map = split_stats.get("train", {})
    test_map = split_stats.get("testing", {})
    validation_map = split_stats.get("validation", {})
    if not isinstance(train_map, dict):
        train_map = {}
    if not isinstance(test_map, dict):
        test_map = {}
    if not isinstance(validation_map, dict):
        validation_map = {}

    testing_equals_validation = bool(test_map and validation_map and test_map == validation_map)

    validation_result = {
        "after_counts_balanced": _is_uniform_count_map(after_counts),
        "train_balanced": _is_uniform_count_map(train_map),
        "testing_balanced": _is_uniform_count_map(test_map),
        "validation_balanced": _is_uniform_count_map(validation_map),
        "testing_equals_validation": testing_equals_validation,
    }
    validation_result["is_balanced"] = bool(
        validation_result["after_counts_balanced"]
        and validation_result["train_balanced"]
        and validation_result["testing_balanced"]
        and validation_result["validation_balanced"]
        and validation_result["testing_equals_validation"]
    )
    return validation_result


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

    train_ratio, test_ratio, validation_ratio = _ensure_valid_ratios(split_cfg)
    train_resize = _parse_resize(resize_cfg.get("train"))
    test_resize = _parse_resize(resize_cfg.get("testing"))
    validation_resize = _parse_resize(resize_cfg.get("validation"))

    # Jaga konsistensi format split: jika testing/validation tidak diset,
    # otomatis ikuti format resize train.
    if train_resize is not None:
        if test_resize is None:
            test_resize = train_resize
        if validation_resize is None:
            validation_resize = train_resize

    if split_dir.exists():
        shutil.rmtree(str(split_dir))
    split_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    split_stats: Dict[str, Dict[str, int]] = {
        "train": {},
        "testing": {},
        "validation": {},
    }
    split_augmented_stats: Dict[str, Dict[str, int]] = {
        "train": {},
        "testing": {},
        "validation": {},
    }
    class_manifest: List[Dict[str, object]] = []
    class_samples, balancing_manifest = _build_balanced_samples(
        class_entries=class_entries,
        extensions=extensions,
        rng=rng,
    )

    for entry in class_entries:
        samples = class_samples[entry.class_name]
        shuffled = samples[:]
        rng.shuffle(shuffled)

        train_count, test_count, validation_count = _compute_split_counts(
            total_count=len(shuffled),
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            validation_ratio=validation_ratio,
        )
        train_samples = shuffled[:train_count]
        test_samples = shuffled[train_count : train_count + test_count]
        validation_samples = shuffled[train_count + test_count : train_count + test_count + validation_count]

        augmented_train_count = sum(1 for sample in train_samples if sample.is_augmented)
        augmented_test_count = sum(1 for sample in test_samples if sample.is_augmented)
        augmented_validation_count = sum(1 for sample in validation_samples if sample.is_augmented)

        class_manifest.append(
            {
                "class_name": entry.class_name,
                "source_dir": str(entry.source_dir),
                "relative_path": entry.relative_path,
                "source_images": entry.image_count,
                "generated_images": int(balancing_manifest["generated_per_class"].get(entry.class_name, 0)),
                "total_images": len(shuffled),
                "split_counts": {
                    "train": len(train_samples),
                    "testing": len(test_samples),
                    "validation": len(validation_samples),
                },
                "split_generated_counts": {
                    "train": augmented_train_count,
                    "testing": augmented_test_count,
                    "validation": augmented_validation_count,
                },
            }
        )

        split_stats["train"][entry.class_name] = len(train_samples)
        split_stats["testing"][entry.class_name] = len(test_samples)
        split_stats["validation"][entry.class_name] = len(validation_samples)
        split_augmented_stats["train"][entry.class_name] = augmented_train_count
        split_augmented_stats["testing"][entry.class_name] = augmented_test_count
        split_augmented_stats["validation"][entry.class_name] = augmented_validation_count

        for idx, sample in enumerate(train_samples):
            filename = _build_sample_filename(index=idx, sample=sample)
            target = split_dir / "train" / entry.class_name / filename
            _save_sample(sample=sample, target_path=target, resize_to=train_resize)

        for idx, sample in enumerate(test_samples):
            filename = _build_sample_filename(index=idx, sample=sample)
            target = split_dir / "testing" / entry.class_name / filename
            _save_sample(sample=sample, target_path=target, resize_to=test_resize)

        for idx, sample in enumerate(validation_samples):
            filename = _build_sample_filename(index=idx, sample=sample)
            target = split_dir / "validation" / entry.class_name / filename
            _save_sample(sample=sample, target_path=target, resize_to=validation_resize)

    metadata_dir = split_dir / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "1.2.0",
        "seed": int(seed),
        "original_dir": str(original_dir),
        "split_dir": str(split_dir),
        "class_mode": class_mode,
        "resize": {
            "train": _serialize_resize(train_resize),
            "testing": _serialize_resize(test_resize),
            "validation": _serialize_resize(validation_resize),
        },
        "split_ratio": {
            "train": train_ratio,
            "testing": test_ratio,
            "validation": validation_ratio,
        },
        "classes": class_manifest,
        "split_stats": split_stats,
        "split_generated_stats": split_augmented_stats,
        "class_balancing": balancing_manifest,
    }
    balance_validation = validate_balanced_split_manifest(manifest)
    manifest["balance_validation"] = balance_validation
    if not balance_validation.get("is_balanced", False):
        raise RuntimeError(
            "Hasil split tidak seimbang. "
            "Periksa konfigurasi split/kelas pada dataset source sebelum training."
        )

    manifest_path = metadata_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest
