from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from src.utils.config import load_datasets_config
from src.utils.paths import PROJECT_ROOT, resolve_project_path


@dataclass
class DatasetConfig:
    dataset_id: str
    description: str
    original_dir: str
    split_dir: str
    class_mode: str
    seed: int
    valid_extensions: List[str]
    split: Dict[str, float]
    resize: Dict[str, object]
    augmentation_options: List[str]

    @property
    def original_path(self):
        return resolve_project_path(self.original_dir)

    @property
    def split_path(self):
        return resolve_project_path(self.split_dir)


DEFAULT_CLASS_MODE = "direct"
DEFAULT_SEED = 42
DEFAULT_VALID_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
DEFAULT_SPLIT_CFG = {"train": 0.70, "testing": 0.15, "validation": 0.15}
DEFAULT_RESIZE_CFG = {"train": [224, 224], "testing": None, "validation": None}
DEFAULT_AUGMENTATIONS = ["no_augment", "augment_on_the_fly"]


def _to_relative_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _coerce_string_list(value: object, fallback: Sequence[str], allow_empty: bool = False) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return list(fallback)
    cleaned: List[str] = []
    for item in value:
        token = str(item).strip()
        if token:
            cleaned.append(token)
    if cleaned or allow_empty:
        return cleaned
    return list(fallback)


def _coerce_split(value: object, fallback: Dict[str, float]) -> Dict[str, float]:
    if not isinstance(value, dict):
        return dict(fallback)
    out = dict(fallback)
    for key in ["train", "testing", "validation"]:
        raw_value = value.get(key)
        if raw_value is None:
            continue
        try:
            out[key] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return out


def _coerce_resize(value: object, fallback: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(value, dict):
        return dict(fallback)
    out = dict(fallback)
    for key in ["train", "testing", "validation"]:
        if key in value:
            out[key] = value.get(key)
    return out


def _has_image_files(directory: Path, extensions: Sequence[str]) -> bool:
    extset = set(str(ext).strip().lower() for ext in extensions if str(ext).strip())
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in extset:
            return True
    return False


def _infer_class_mode(dataset_root: Path, extensions: Sequence[str], fallback_mode: str) -> str:
    if not dataset_root.exists() or not dataset_root.is_dir():
        return fallback_mode

    direct_dirs = [path for path in dataset_root.iterdir() if path.is_dir() and _has_image_files(path, extensions)]
    if len(direct_dirs) >= 2:
        return "direct"

    recursive_leaf_count = 0
    for path in dataset_root.rglob("*"):
        if not path.is_dir():
            continue
        if _has_image_files(path, extensions):
            recursive_leaf_count += 1
            if recursive_leaf_count >= 2:
                return "recursive_leaf"
    return fallback_mode


def _slugify_dataset_id(raw_value: str, strip_numeric_prefix: bool) -> str:
    token = str(raw_value).strip().lower()
    if strip_numeric_prefix:
        token = re.sub(r"^\d+[\s._-]*", "", token)
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    return token or "dataset"


def _next_dataset_id(base_id: str, used_ids: Set[str]) -> str:
    if base_id not in used_ids:
        return base_id
    index = 2
    while f"{base_id}_{index}" in used_ids:
        index += 1
    return f"{base_id}_{index}"


class DatasetRegistry:
    def __init__(self):
        cfg = load_datasets_config()
        defaults = cfg.get("defaults", {}) if isinstance(cfg.get("defaults", {}), dict) else {}
        raw_datasets = cfg.get("datasets", {})
        if not isinstance(raw_datasets, dict):
            raw_datasets = {}

        default_original_root = str(defaults.get("original_root", "dataset/original"))
        default_split_root = str(defaults.get("split_root", "dataset/split"))
        default_class_mode = str(defaults.get("class_mode", DEFAULT_CLASS_MODE)).strip() or DEFAULT_CLASS_MODE
        default_seed = int(defaults.get("seed", DEFAULT_SEED))
        default_extensions = _coerce_string_list(defaults.get("valid_extensions"), DEFAULT_VALID_EXTENSIONS)
        default_split_cfg = _coerce_split(defaults.get("split"), DEFAULT_SPLIT_CFG)
        default_resize_cfg = _coerce_resize(defaults.get("resize"), DEFAULT_RESIZE_CFG)
        default_augmentations = _coerce_string_list(
            defaults.get("augmentation_options"),
            DEFAULT_AUGMENTATIONS,
            allow_empty=True,
        )
        auto_discovery = bool(defaults.get("auto_discovery", True))
        strip_numeric_prefix = bool(defaults.get("strip_numeric_prefix", True))

        discovered: Dict[str, DatasetConfig] = {}
        discovered_slug_index: Dict[str, List[str]] = {}

        if auto_discovery:
            original_root_path = resolve_project_path(default_original_root)
            split_root_path = resolve_project_path(default_split_root)
            used_discovered_ids: Set[str] = set()

            if original_root_path.exists() and original_root_path.is_dir():
                for folder in sorted([path for path in original_root_path.iterdir() if path.is_dir()]):
                    slug = _slugify_dataset_id(folder.name, strip_numeric_prefix=strip_numeric_prefix)
                    dataset_id = _next_dataset_id(slug, used_discovered_ids)
                    used_discovered_ids.add(dataset_id)

                    discovered_cfg = DatasetConfig(
                        dataset_id=dataset_id,
                        description=f"Auto-discovered dari folder '{folder.name}'.",
                        original_dir=_to_relative_project_path(folder),
                        split_dir=_to_relative_project_path(split_root_path / dataset_id),
                        class_mode=_infer_class_mode(folder, default_extensions, default_class_mode),
                        seed=default_seed,
                        valid_extensions=list(default_extensions),
                        split=dict(default_split_cfg),
                        resize=dict(default_resize_cfg),
                        augmentation_options=list(default_augmentations),
                    )
                    discovered[dataset_id] = discovered_cfg
                    discovered_slug_index.setdefault(slug, []).append(dataset_id)

        self._datasets = {}
        configured_source_paths: Set[str] = set()
        used_final_ids: Set[str] = set()

        def _pick_discovered_for_config(dataset_id: str, original_dir_hint: str) -> Optional[DatasetConfig]:
            candidates = [
                _slugify_dataset_id(dataset_id, strip_numeric_prefix=strip_numeric_prefix),
                _slugify_dataset_id(Path(original_dir_hint).name, strip_numeric_prefix=strip_numeric_prefix),
            ]
            for slug in candidates:
                for discovered_id in discovered_slug_index.get(slug, []):
                    cfg_item = discovered.get(discovered_id)
                    if cfg_item is not None:
                        return cfg_item
            return None

        for dataset_id, raw_item in raw_datasets.items():
            item = raw_item if isinstance(raw_item, dict) else {}
            dataset_id = str(dataset_id).strip()
            if not dataset_id:
                continue

            original_dir_hint = str(item.get("original_dir", "")).strip()
            resolved_original = resolve_project_path(original_dir_hint) if original_dir_hint else None
            mapped_discovered = None

            if resolved_original is None or not resolved_original.exists():
                mapped_discovered = _pick_discovered_for_config(dataset_id, original_dir_hint)
                if mapped_discovered is not None:
                    original_dir = mapped_discovered.original_dir
                elif original_dir_hint:
                    original_dir = original_dir_hint
                else:
                    original_dir = _to_relative_project_path(resolve_project_path(default_original_root) / dataset_id)
            else:
                original_dir = original_dir_hint

            split_dir = str(item.get("split_dir", f"{default_split_root}/{dataset_id}")).strip()
            if not split_dir:
                split_dir = f"{default_split_root}/{dataset_id}"

            class_mode = str(item.get("class_mode") or (mapped_discovered.class_mode if mapped_discovered else default_class_mode))
            dataset_cfg = DatasetConfig(
                dataset_id=dataset_id,
                description=str(item.get("description", mapped_discovered.description if mapped_discovered else "")),
                original_dir=original_dir,
                split_dir=split_dir,
                class_mode=class_mode,
                seed=int(item.get("seed", default_seed)),
                valid_extensions=_coerce_string_list(item.get("valid_extensions"), default_extensions),
                split=_coerce_split(item.get("split"), default_split_cfg),
                resize=_coerce_resize(item.get("resize"), default_resize_cfg),
                augmentation_options=_coerce_string_list(
                    item.get("augmentation_options"),
                    default_augmentations,
                    allow_empty=True,
                ),
            )
            self._datasets[dataset_id] = dataset_cfg
            used_final_ids.add(dataset_id)
            configured_source_paths.add(str(dataset_cfg.original_path.resolve()))

        for discovered_id, discovered_cfg in discovered.items():
            if str(discovered_cfg.original_path.resolve()) in configured_source_paths:
                continue

            final_dataset_id = _next_dataset_id(discovered_id, used_final_ids)
            if final_dataset_id == discovered_id:
                self._datasets[final_dataset_id] = discovered_cfg
            else:
                split_root_path = resolve_project_path(default_split_root)
                self._datasets[final_dataset_id] = DatasetConfig(
                    dataset_id=final_dataset_id,
                    description=discovered_cfg.description,
                    original_dir=discovered_cfg.original_dir,
                    split_dir=_to_relative_project_path(split_root_path / final_dataset_id),
                    class_mode=discovered_cfg.class_mode,
                    seed=discovered_cfg.seed,
                    valid_extensions=list(discovered_cfg.valid_extensions),
                    split=dict(discovered_cfg.split),
                    resize=dict(discovered_cfg.resize),
                    augmentation_options=list(discovered_cfg.augmentation_options),
                )
            used_final_ids.add(final_dataset_id)

        if not self._datasets:
            raise ValueError(
                "Tidak ada dataset ditemukan. Tambahkan folder pada '{}' atau isi configs/datasets.yaml.".format(
                    default_original_root
                )
            )

        requested_default = str(defaults.get("default_dataset", "")).strip()
        self._default_dataset = requested_default if requested_default in self._datasets else next(iter(self._datasets.keys()))

    @property
    def default_dataset(self) -> str:
        return str(self._default_dataset)

    def list_dataset_ids(self) -> List[str]:
        # Urutan tetap deterministik: dataset config dulu, lalu hasil auto-discovery.
        return list(self._datasets.keys())

    def get(self, dataset_id: str) -> DatasetConfig:
        if dataset_id not in self._datasets:
            available = ", ".join(self.list_dataset_ids())
            raise KeyError(f"Dataset '{dataset_id}' tidak ditemukan. Tersedia: {available}")
        return self._datasets[dataset_id]
