from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from src.datasets.registry import DatasetRegistry
from src.datasets.validator import discover_class_directories


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cek distribusi dataset dari registry/config.")
    parser.add_argument("--dataset", type=str, default=None, help="ID dataset dari configs/datasets.yaml")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Override folder dataset original")
    parser.add_argument("--class-mode", type=str, default=None, choices=["direct", "recursive_leaf"])
    parser.add_argument("--save-plot", type=str, default=None, help="Path output gambar distribusi")
    return parser


def _print_distribution(distribution: Dict[str, int], source: Path) -> None:
    total_images = sum(distribution.values())
    print("\n=== Distribusi Dataset ===")
    print("Sumber   :", source)
    print("Kelas    :", len(distribution))
    print("Total img:", total_images)
    print("-" * 80)
    print("{:<45s} {:>12s} {:>12s}".format("Kelas", "Jumlah", "Persen"))
    print("-" * 80)
    for class_name, count in sorted(distribution.items()):
        pct = (float(count) / float(total_images) * 100.0) if total_images else 0.0
        print("{:<45s} {:>12d} {:>11.2f}%".format(class_name, count, pct))
    print("-" * 80)


def _plot_distribution(distribution: Dict[str, int], output_path: Path) -> None:
    class_names = list(distribution.keys())
    counts = list(distribution.values())

    plt.figure(figsize=(14, 6))
    bars = plt.bar(class_names, counts)
    plt.title("Distribusi Jumlah Data per Kelas")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Gambar")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        height = int(bar.get_height())
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print("Grafik distribusi disimpan ke:", output_path)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = DatasetRegistry()
    dataset_id = args.dataset or registry.default_dataset
    dataset_cfg = registry.get(dataset_id)

    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else dataset_cfg.original_path
    class_mode = args.class_mode or dataset_cfg.class_mode
    extensions = dataset_cfg.valid_extensions

    class_entries = discover_class_directories(
        dataset_root=dataset_dir,
        class_mode=class_mode,
        extensions=extensions,
    )
    if len(class_entries) < 2:
        raise ValueError(
            f"Dataset '{dataset_id}' harus punya minimal 2 kelas valid. "
            f"Saat ini: {len(class_entries)}"
        )

    distribution = {entry.class_name: entry.image_count for entry in class_entries}
    _print_distribution(distribution, dataset_dir)

    if args.save_plot:
        output_path = Path(args.save_plot).resolve()
    else:
        output_path = Path("dataset_distribution_{}.png".format(dataset_id)).resolve()
    _plot_distribution(distribution, output_path)


if __name__ == "__main__":
    main()
