from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.registry import DatasetRegistry
from src.datasets.splitter import split_dataset


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

    manifest = split_dataset(
        original_dir=original_dir,
        split_dir=split_dir,
        class_mode=class_mode,
        extensions=dataset_cfg.valid_extensions,
        split_cfg=dataset_cfg.split,
        resize_cfg=dataset_cfg.resize,
        seed=seed,
    )

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
    print("Split selesai.")


if __name__ == "__main__":
    main()
