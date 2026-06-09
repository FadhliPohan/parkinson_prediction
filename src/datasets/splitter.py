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

# Preset rasio split dinamis (train, testing, validation).
# Catatan: testing dan validation WAJIB sama besar agar split simetris
# (lihat _compute_split_counts). Kedua preset di bawah memenuhi syarat itu.
SPLIT_PRESETS: Dict[str, Dict[str, float]] = {
    "80-10-10": {"train": 0.80, "testing": 0.10, "validation": 0.10},
    "70-15-15": {"train": 0.70, "testing": 0.15, "validation": 0.15},
}

# Token alias yang dianggap "pakai rasio default dari config dataset" (bukan preset eksplisit).
_CONFIG_PRESET_ALIASES = {"", "config", "default", "none", "config-default"}


def normalize_preset(preset: Optional[str]) -> Optional[str]:
    """Normalisasi token preset.

    Mengembalikan None untuk nilai yang berarti "pakai rasio default config"
    (mis. None/"config"/"default"), atau key preset terstandar (mis. "80-10-10").
    """
    if preset is None:
        return None
    key = str(preset).strip().lower().replace("_", "-")
    if key in _CONFIG_PRESET_ALIASES:
        return None
    return key


def preset_label_for_ratios(split_cfg: Dict[str, float]) -> str:
    """Label preset (mis. "80-10-10") yang diturunkan dari rasio split aktif."""
    train = int(round(float(split_cfg.get("train", 0.0)) * 100))
    testing = int(round(float(split_cfg.get("testing", 0.0)) * 100))
    validation = int(round(float(split_cfg.get("validation", 0.0)) * 100))
    return f"{train}-{testing}-{validation}"


def split_dir_for_preset(base_split_path: Path, preset: Optional[str]) -> Path:
    """Tentukan folder split untuk sebuah preset.

    - preset None / "config" / "default" -> folder split dasar (kompatibel lama).
    - preset eksplisit -> folder bersuffix, mis. ``<base>__80-10-10``.

    Dengan begitu beberapa preset (mis. 80-10-10 dan 70-15-15) bisa hidup
    berdampingan di disk tanpa saling menimpa.
    """
    norm = normalize_preset(preset)
    if norm is None:
        return base_split_path
    return base_split_path.parent / f"{base_split_path.name}__{norm}"


def resolve_preset_list(raw_value: Optional[str]) -> List[Optional[str]]:
    """Ubah argumen preset (mis. "both"/"config"/"80-10-10,70-15-15") jadi daftar preset.

    Setiap elemen adalah key preset ternormalisasi atau None (config-default).
    Duplikat dibuang dengan mempertahankan urutan.
    """
    if raw_value is None or not str(raw_value).strip():
        return [None]

    normalized = str(raw_value).strip().lower()
    if normalized in {"both", "all", "*"}:
        return list(SPLIT_PRESETS.keys())

    presets: List[Optional[str]] = []
    seen = set()
    for token in [item.strip() for item in str(raw_value).split(",") if item.strip()]:
        norm = normalize_preset(token)
        marker = norm or "__config__"
        if marker in seen:
            continue
        seen.add(marker)
        presets.append(norm)
    return presets or [None]


def validate_split_ratios(
    train: float,
    testing: float,
    validation: float,
    tolerance: float = 1e-6,
) -> Dict[str, float]:
    """Validasi rasio split dan kembalikan dict ternormalisasi.

    Aturan:
        - Setiap rasio harus > 0.
        - Total train+testing+validation harus = 1.0 (dalam toleransi).
        - testing harus sama dengan validation (kebijakan split simetris).

    Raises:
        ValueError: bila rasio tidak valid.
    """
    train_f = float(train)
    testing_f = float(testing)
    validation_f = float(validation)

    if train_f <= 0 or testing_f <= 0 or validation_f <= 0:
        raise ValueError("Setiap rasio split harus > 0.")

    total = train_f + testing_f + validation_f
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            f"Total rasio split harus = 1.0 (100%), saat ini {total:.4f} "
            f"(train={train_f}, testing={testing_f}, validation={validation_f})."
        )

    if abs(testing_f - validation_f) > tolerance:
        raise ValueError(
            "Rasio testing dan validation harus sama (split simetris), "
            f"saat ini testing={testing_f}, validation={validation_f}."
        )

    return {"train": train_f, "testing": testing_f, "validation": validation_f}


def resolve_split_cfg(
    preset: Optional[str] = None,
    train: Optional[float] = None,
    testing: Optional[float] = None,
    validation: Optional[float] = None,
    fallback: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Tentukan konfigurasi split dari preset / rasio manual / fallback.

    Prioritas: rasio manual lengkap > preset > fallback. Hasil divalidasi
    dengan :func:`validate_split_ratios`.
    """
    if train is not None and testing is not None and validation is not None:
        return validate_split_ratios(train, testing, validation)

    if preset:
        key = str(preset).strip()
        if key not in SPLIT_PRESETS:
            available = ", ".join(sorted(SPLIT_PRESETS.keys()))
            raise ValueError(f"Preset split '{preset}' tidak dikenal. Tersedia: {available}")
        cfg = SPLIT_PRESETS[key]
        return validate_split_ratios(cfg["train"], cfg["testing"], cfg["validation"])

    if fallback is not None:
        return validate_split_ratios(
            fallback.get("train", 0.0),
            fallback.get("testing", 0.0),
            fallback.get("validation", 0.0),
        )

    raise ValueError("Tidak ada konfigurasi split yang bisa ditentukan.")


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


def _split_real_per_class(
    class_entries: Sequence[ClassEntry],
    extensions: Sequence[str],
    train_ratio: float,
    test_ratio: float,
    validation_ratio: float,
    rng: random.Random,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], Dict[str, List[Path]], Dict[str, int]]:
    """Split gambar ASLI tiap kelas ke train/test/validation (stratified, tanpa augmentasi).

    BLOCKER-02: split dilakukan SEBELUM augmentasi/balancing apa pun. Dengan begitu
    test & validation hanya berisi gambar asli, dan tidak ada gambar (atau rotasinya)
    yang muncul di lebih dari satu split. Balancing dilakukan terpisah pada train saja
    via :func:`_balance_train_per_class`.
    """
    train_real: Dict[str, List[Path]] = {}
    test_real: Dict[str, List[Path]] = {}
    validation_real: Dict[str, List[Path]] = {}
    before_counts: Dict[str, int] = {}

    for entry in class_entries:
        files = list_image_files(entry.source_dir, extensions)
        before_counts[entry.class_name] = len(files)

        shuffled = files[:]
        rng.shuffle(shuffled)
        train_count, test_count, validation_count = _compute_split_counts(
            total_count=len(shuffled),
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            validation_ratio=validation_ratio,
        )
        train_real[entry.class_name] = shuffled[:train_count]
        test_real[entry.class_name] = shuffled[train_count : train_count + test_count]
        validation_real[entry.class_name] = shuffled[
            train_count + test_count : train_count + test_count + validation_count
        ]

    if not before_counts:
        raise ValueError("Tidak ada kelas valid untuk proses split.")

    return train_real, test_real, validation_real, before_counts


def _balance_train_per_class(
    train_real: Dict[str, List[Path]],
    rng: random.Random,
) -> Tuple[Dict[str, List[SplitSample]], int, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Seimbangkan HANYA train: tambah rotasi gambar train kelas minoritas ke jumlah mayoritas.

    BLOCKER-02: sumber rotasi (`rng.choice(files)`) diambil EKSKLUSIF dari gambar train
    kelas yang sama, sehingga gambar sintetis tidak pernah berbagi sumber dengan test/validation.
    Mengembalikan: train_samples per kelas, target_per_class (train), generated_per_class,
    after_train_counts, before_train_counts.
    """
    before_train_counts = {class_name: len(files) for class_name, files in train_real.items()}
    target_per_class = max(before_train_counts.values()) if before_train_counts else 0

    train_samples: Dict[str, List[SplitSample]] = {}
    generated_per_class: Dict[str, int] = {}

    for class_name, files in train_real.items():
        base_samples = [SplitSample(source_path=path) for path in files]
        missing_count = max(0, target_per_class - len(files))
        generated_per_class[class_name] = missing_count

        synthetic_samples: List[SplitSample] = []
        if missing_count > 0 and files:
            for _ in range(missing_count):
                source_path = rng.choice(files)  # hanya dari gambar TRAIN kelas ini
                angle = rng.uniform(ROTATION_MIN_DEGREES, ROTATION_MAX_DEGREES)
                synthetic_samples.append(
                    SplitSample(
                        source_path=source_path,
                        is_augmented=True,
                        rotation_angle=float(angle),
                    )
                )

        train_samples[class_name] = base_samples + synthetic_samples

    after_train_counts = {class_name: len(samples) for class_name, samples in train_samples.items()}
    return train_samples, target_per_class, generated_per_class, after_train_counts, before_train_counts


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


def _sum_count_map(count_map: object) -> int:
    if not isinstance(count_map, dict):
        return 0
    total = 0
    for value in count_map.values():
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


def validate_balanced_split_manifest(manifest: Dict[str, object]) -> Dict[str, object]:
    """Validasi split dengan kebijakan anti-leakage (BLOCKER-02).

    Invariant yang divalidasi:
      - ``train_balanced``: jumlah train per kelas seragam (hasil balancing train-only).
      - ``testing_equals_validation``: jumlah test == validation per kelas (rasio simetris).
      - ``no_synthetic_in_testing`` / ``no_synthetic_in_validation``: test & validation
        TIDAK boleh mengandung gambar sintetis (guard kebocoran data).

    ``is_balanced`` (dipertahankan untuk kompatibilitas konsumen) = True hanya jika
    train seimbang, test==val, DAN tidak ada sintetis di test/validation.
    """
    split_stats = manifest.get("split_stats", {})
    generated_stats = manifest.get("split_generated_stats", {})
    if not isinstance(split_stats, dict):
        split_stats = {}
    if not isinstance(generated_stats, dict):
        generated_stats = {}

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
    no_synthetic_in_testing = _sum_count_map(generated_stats.get("testing", {})) == 0
    no_synthetic_in_validation = _sum_count_map(generated_stats.get("validation", {})) == 0
    no_leakage_in_eval = no_synthetic_in_testing and no_synthetic_in_validation

    validation_result = {
        "train_balanced": _is_uniform_count_map(train_map),
        "testing_equals_validation": testing_equals_validation,
        "no_synthetic_in_testing": no_synthetic_in_testing,
        "no_synthetic_in_validation": no_synthetic_in_validation,
        "no_leakage_in_eval": no_leakage_in_eval,
    }
    validation_result["is_balanced"] = bool(
        validation_result["train_balanced"]
        and validation_result["testing_equals_validation"]
        and validation_result["no_leakage_in_eval"]
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
    split_preset: Optional[str] = None,
) -> Dict[str, object]:
    class_entries = validate_dataset(
        dataset_root=original_dir,
        class_mode=class_mode,
        extensions=extensions,
    )

    train_ratio, test_ratio, validation_ratio = _ensure_valid_ratios(split_cfg)
    # Label preset: pakai yang eksplisit bila ada, kalau tidak turunkan dari rasio.
    normalized_preset = normalize_preset(split_preset)
    preset_label = normalized_preset or preset_label_for_ratios(
        {"train": train_ratio, "testing": test_ratio, "validation": validation_ratio}
    )
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

    # BLOCKER-02: split gambar ASLI dulu (stratified), lalu balancing HANYA pada train.
    # Test & validation selalu 100% gambar asli → tidak ada kebocoran sintetis, dan
    # rotasi train tidak pernah berbagi sumber dengan test/validation.
    train_real, test_real, validation_real, before_counts = _split_real_per_class(
        class_entries=class_entries,
        extensions=extensions,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        validation_ratio=validation_ratio,
        rng=rng,
    )
    (
        train_samples_per_class,
        target_train,
        generated_per_class,
        after_train_counts,
        before_train_counts,
    ) = _balance_train_per_class(train_real=train_real, rng=rng)

    source_is_balanced = len(set(before_counts.values())) <= 1
    balancing_manifest = {
        "applied": not source_is_balanced,
        "scope": "train_only",
        "strategy": "minority_rotation_to_majority_train_only",
        "rotation_range_degrees": [ROTATION_MIN_DEGREES, ROTATION_MAX_DEGREES],
        "class_count": len(before_counts),
        "target_per_class": target_train,
        "before_counts": before_counts,
        "before_train_counts": before_train_counts,
        "after_train_counts": after_train_counts,
        "generated_per_class": generated_per_class,
        "total_generated": int(sum(generated_per_class.values())),
    }

    for entry in class_entries:
        class_name = entry.class_name
        train_samples = train_samples_per_class[class_name]
        test_samples = [SplitSample(source_path=path) for path in test_real[class_name]]
        validation_samples = [SplitSample(source_path=path) for path in validation_real[class_name]]

        # Acak urutan train agar sampel sintetis tidak menumpuk di akhir.
        rng.shuffle(train_samples)

        augmented_train_count = sum(1 for sample in train_samples if sample.is_augmented)
        augmented_test_count = 0  # test selalu gambar asli (BLOCKER-02)
        augmented_validation_count = 0  # validation selalu gambar asli (BLOCKER-02)

        class_manifest.append(
            {
                "class_name": class_name,
                "source_dir": str(entry.source_dir),
                "relative_path": entry.relative_path,
                "source_images": entry.image_count,
                "generated_images": int(generated_per_class.get(class_name, 0)),
                "total_images": len(train_samples) + len(test_samples) + len(validation_samples),
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

        split_stats["train"][class_name] = len(train_samples)
        split_stats["testing"][class_name] = len(test_samples)
        split_stats["validation"][class_name] = len(validation_samples)
        split_augmented_stats["train"][class_name] = augmented_train_count
        split_augmented_stats["testing"][class_name] = augmented_test_count
        split_augmented_stats["validation"][class_name] = augmented_validation_count

        for idx, sample in enumerate(train_samples):
            filename = _build_sample_filename(index=idx, sample=sample)
            target = split_dir / "train" / class_name / filename
            _save_sample(sample=sample, target_path=target, resize_to=train_resize)

        for idx, sample in enumerate(test_samples):
            filename = _build_sample_filename(index=idx, sample=sample)
            target = split_dir / "testing" / class_name / filename
            _save_sample(sample=sample, target_path=target, resize_to=test_resize)

        for idx, sample in enumerate(validation_samples):
            filename = _build_sample_filename(index=idx, sample=sample)
            target = split_dir / "validation" / class_name / filename
            _save_sample(sample=sample, target_path=target, resize_to=validation_resize)

    metadata_dir = split_dir / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "1.4.0",
        "seed": int(seed),
        "split_preset": preset_label,
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
            "Hasil split tidak valid (train tidak seimbang, test≠validation, "
            "atau ada gambar sintetis di test/validation). "
            "Periksa konfigurasi split/kelas pada dataset source sebelum training."
        )

    manifest_path = metadata_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest
