from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass
class ClassEntry:
    class_name: str
    source_dir: Path
    relative_path: str
    image_count: int


DEFAULT_EXTENSIONS: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


def _normalized_extensions(extensions: Sequence[str]) -> Tuple[str, ...]:
    normalized = []
    for ext in extensions:
        cleaned = str(ext).strip().lower()
        if not cleaned:
            continue
        if not cleaned.startswith("."):
            cleaned = "." + cleaned
        normalized.append(cleaned)
    return tuple(sorted(set(normalized)))


def list_image_files(folder: Path, extensions: Sequence[str]) -> List[Path]:
    extset = set(_normalized_extensions(extensions))
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in extset
        ]
    )


def _dir_has_images(folder: Path, extensions: Sequence[str]) -> bool:
    return len(list_image_files(folder, extensions)) > 0


def _build_class_name(relative_path: Path, used_names: Dict[str, str]) -> str:
    base = relative_path.as_posix().strip("/")
    base = base.replace("/", "__") or "class_root"

    if base not in used_names:
        used_names[base] = relative_path.as_posix()
        return base

    # Hindari collision nama kelas jika ada path berbeda dengan slug sama.
    suffix = sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:8]
    class_name = f"{base}__{suffix}"
    used_names[class_name] = relative_path.as_posix()
    return class_name


def discover_class_directories(
    dataset_root: Path,
    class_mode: str,
    extensions: Sequence[str],
) -> List[ClassEntry]:
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_root}")

    mode = str(class_mode).strip().lower()
    if mode not in {"direct", "recursive_leaf"}:
        raise ValueError(f"class_mode tidak didukung: {class_mode}")

    entries: List[ClassEntry] = []
    used_names: Dict[str, str] = {}

    if mode == "direct":
        for class_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
            files = list_image_files(class_dir, extensions)
            if not files:
                continue
            class_name = _build_class_name(class_dir.relative_to(dataset_root), used_names)
            entries.append(
                ClassEntry(
                    class_name=class_name,
                    source_dir=class_dir,
                    relative_path=class_dir.relative_to(dataset_root).as_posix(),
                    image_count=len(files),
                )
            )
        return entries

    # recursive_leaf
    candidate_dirs = []
    for directory in sorted([p for p in dataset_root.rglob("*") if p.is_dir()]):
        if _dir_has_images(directory, extensions):
            candidate_dirs.append(directory)

    candidate_set = set(candidate_dirs)
    leaf_dirs = []
    for directory in candidate_dirs:
        has_child_candidate = any(
            child != directory and child in candidate_set
            for child in directory.rglob("*")
            if child.is_dir()
        )
        if not has_child_candidate:
            leaf_dirs.append(directory)

    for class_dir in sorted(leaf_dirs):
        files = list_image_files(class_dir, extensions)
        if not files:
            continue
        relative_path = class_dir.relative_to(dataset_root)
        class_name = _build_class_name(relative_path, used_names)
        entries.append(
            ClassEntry(
                class_name=class_name,
                source_dir=class_dir,
                relative_path=relative_path.as_posix(),
                image_count=len(files),
            )
        )

    return entries


def validate_dataset(
    dataset_root: Path,
    class_mode: str,
    extensions: Sequence[str],
) -> List[ClassEntry]:
    entries = discover_class_directories(
        dataset_root=dataset_root,
        class_mode=class_mode,
        extensions=extensions,
    )
    if len(entries) < 2:
        raise ValueError(
            "Dataset minimal harus punya 2 kelas valid. "
            f"Saat ini ditemukan {len(entries)} kelas dari: {dataset_root}"
        )
    return entries
