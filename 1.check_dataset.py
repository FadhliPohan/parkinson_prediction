from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt


DATASET_DIR = Path("dataset/original")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def count_images_per_class(dataset_dir: Path) -> Dict[str, int]:
    class_counts: Dict[str, int] = defaultdict(int)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir.resolve()}")

    for class_dir in dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue

        image_count = sum(
            1
            for file in class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )
        class_counts[class_dir.name] = image_count

    if not class_counts:
        raise ValueError(f"Tidak ada folder kelas di dalam: {dataset_dir.resolve()}")

    return dict(class_counts)


def print_distribution(class_counts: Dict[str, int]) -> None:
    total_images = sum(class_counts.values())
    print("\n=== Distribusi Dataset ===")
    print(f"Total gambar: {total_images}")
    print("-" * 40)

    for class_name, count in sorted(class_counts.items()):
        percentage = (count / total_images * 100) if total_images else 0
        print(f"{class_name:15s}: {count:5d} ({percentage:6.2f}%)")

    print("-" * 40)


def plot_distribution(class_counts: Dict[str, int], save_path: Optional[Path] = None) -> None:
    class_names = list(class_counts.keys())
    counts = list(class_counts.values())
    total_images = sum(counts)
    percentages = [(c / total_images * 100) if total_images else 0 for c in counts]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(class_names, counts)
    plt.title("Distribusi Jumlah Data per Kelas")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Gambar")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Grafik disimpan ke: {save_path.resolve()}")

    backend_name = plt.get_backend().lower()
    if "agg" in backend_name:
        plt.close()
    else:
        plt.show()


def main() -> None:
    class_counts = count_images_per_class(DATASET_DIR)
    print_distribution(class_counts)
    plot_distribution(class_counts, save_path=Path("dataset_distribution.png"))


if __name__ == "__main__":
    main()
