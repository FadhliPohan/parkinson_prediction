from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.registry import DatasetRegistry
from src.datasets.splitter import split_dataset, validate_balanced_split_manifest
from src.datasets.validator import list_image_files


ON_EXISTING_SPLIT_CHOICES = ("ask", "resplit", "skip")


def _split_manifest_path(split_dir: Path) -> Path:
    return split_dir / "_metadata" / "split_manifest.json"


def _load_split_manifest(split_dir: Path):
    path = _split_manifest_path(split_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return payload
        return None
    except Exception:
        return None


def _split_dir_has_required_structure(split_dir: Path, extensions):
    for split_name in ["train", "testing", "validation"]:
        split_path = split_dir / split_name
        if not split_path.exists() or not split_path.is_dir():
            return False
        class_dirs = [path for path in split_path.iterdir() if path.is_dir()]
        if not class_dirs:
            return False
        has_images = any(list_image_files(class_dir, extensions) for class_dir in class_dirs)
        if not has_images:
            return False
    return True


def _ask_resplit(dataset_id: str, split_dir: Path) -> bool:
    answer = input(
        "\nFolder split untuk dataset '{}' sudah ada di:\n{}\nPerlu split ulang? [y/N]: ".format(
            dataset_id, split_dir
        )
    ).strip().lower()
    return answer in {"y", "yes"}


def _assert_split_balance(manifest, source_label: str) -> None:
    validation = manifest.get("balance_validation") if isinstance(manifest, dict) else None
    if not isinstance(validation, dict):
        validation = validate_balanced_split_manifest(manifest)
    if not validation.get("is_balanced", False):
        raise RuntimeError(
            "Split {} tidak seimbang antar kelas. Jalankan split ulang dengan konfigurasi dataset yang benar.".format(
                source_label
            )
        )
    print("[VALIDASI] Split {} seimbang antar kelas.".format(source_label))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split dataset dinamis berbasis registry/config.")
    parser.add_argument("--dataset", type=str, default=None, help="ID dataset dari configs/datasets.yaml")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Override folder dataset original")
    parser.add_argument("--split-dir", type=str, default=None, help="Override folder output split")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--class-mode",
        type=str,
        default=None,
        choices=["direct", "recursive_leaf"],
        help="Override mode deteksi kelas",
    )
    parser.add_argument(
        "--on-existing-split",
        type=str,
        default="ask",
        choices=list(ON_EXISTING_SPLIT_CHOICES),
        help="Perilaku jika folder split sudah ada: ask, resplit, atau skip.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = DatasetRegistry()
    dataset_id = args.dataset or registry.default_dataset
    dataset_cfg = registry.get(dataset_id)

    original_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else dataset_cfg.original_path
    split_dir = Path(args.split_dir).resolve() if args.split_dir else dataset_cfg.split_path
    seed = int(args.seed) if args.seed is not None else int(dataset_cfg.seed)
    class_mode = args.class_mode or dataset_cfg.class_mode

    print("\n=== Split Dataset ===")
    print("Dataset ID   :", dataset_id)
    print("Original dir :", original_dir)
    print("Split dir    :", split_dir)
    print("Class mode   :", class_mode)
    print("Seed         :", seed)

    if split_dir.exists():
        if args.on_existing_split == "resplit":
            should_resplit = True
        elif args.on_existing_split == "skip":
            should_resplit = False
        elif not sys.stdin.isatty():
            print(
                "[INFO] Split folder sudah ada, sesi non-interaktif tidak bisa bertanya. "
                "Gunakan split existing. Pakai `--on-existing-split resplit` untuk memaksa split ulang."
            )
            should_resplit = False
        else:
            should_resplit = _ask_resplit(dataset_id=dataset_id, split_dir=split_dir)
    else:
        should_resplit = True

    if should_resplit:
        manifest = split_dataset(
            original_dir=original_dir,
            split_dir=split_dir,
            class_mode=class_mode,
            extensions=dataset_cfg.valid_extensions,
            split_cfg=dataset_cfg.split,
            resize_cfg=dataset_cfg.resize,
            seed=seed,
        )
        source_label = "baru"
    else:
        if not _split_dir_has_required_structure(split_dir, dataset_cfg.valid_extensions):
            raise RuntimeError(
                "Split existing tidak valid/kurang lengkap. Jalankan ulang dengan `--on-existing-split resplit`."
            )
        manifest = _load_split_manifest(split_dir)
        if manifest is None:
            raise RuntimeError(
                "split_manifest.json tidak ditemukan pada split existing. "
                "Jalankan ulang dengan `--on-existing-split resplit`."
            )
        source_label = "existing"

    split_stats = manifest.get("split_stats", {})
    print("\n=== Ringkasan Split ===")
    for split_name in ["train", "testing", "validation"]:
        per_class = split_stats.get(split_name, {})
        total = sum(int(v) for v in per_class.values())
        print(f"- {split_name:10s}: {total} gambar | {len(per_class)} kelas")
    balancing_info = manifest.get("class_balancing", {})
    if balancing_info:
        print("\n=== Ringkasan Balancing Kelas ===")
        if balancing_info.get("applied"):
            print("Status      : Aktif (dataset awal tidak seimbang)")
            print("Strategi    :", balancing_info.get("strategy", "-"))
            print("Target/kelas:", balancing_info.get("target_per_class", "-"))
            print("Total tambah:", balancing_info.get("total_generated", 0))
            generated_per_class = balancing_info.get("generated_per_class", {})
            for class_name in sorted(generated_per_class.keys()):
                generated = int(generated_per_class[class_name])
                if generated > 0:
                    print(f"- {class_name}: +{generated} gambar rotasi")
        else:
            print("Status      : Tidak perlu (dataset sudah seimbang)")
    _assert_split_balance(manifest, source_label=source_label)
    print("Split selesai.")


if __name__ == "__main__":
    main()
