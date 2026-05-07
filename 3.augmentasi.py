import argparse
from pathlib import Path
from typing import Dict, List

from augmentation_selected.registry import list_available_profiles, load_profile


SPLIT_DIR = Path("dataset/split")
TRAIN_DIR = SPLIT_DIR / "train"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TARGET_SIZE = (224, 224)


def list_image_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def count_train_images_per_class(train_dir: Path) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    if not train_dir.exists() or not train_dir.is_dir():
        return distribution
    for class_dir in sorted([path for path in train_dir.iterdir() if path.is_dir()]):
        distribution[class_dir.name] = len(list_image_files(class_dir))
    return distribution


def validate_train_split() -> Dict[str, int]:
    if not SPLIT_DIR.exists():
        raise FileNotFoundError("Folder split tidak ditemukan: {}".format(SPLIT_DIR.resolve()))
    if not TRAIN_DIR.exists():
        raise FileNotFoundError("Folder train tidak ditemukan: {}".format(TRAIN_DIR.resolve()))

    class_distribution = count_train_images_per_class(TRAIN_DIR)
    if not class_distribution:
        raise ValueError("Folder train kosong atau tidak memiliki subfolder kelas: {}".format(TRAIN_DIR.resolve()))
    return class_distribution


def print_profile(profile_name: str) -> None:
    profile = load_profile(profile_name)
    print("\nProfile:", profile["display_name"])
    print("Nama internal :", profile["name"])
    print("Tag run       :", profile["run_tag"])
    print("Aktif augment :", "ya" if profile["enabled"] else "tidak")
    print("File sumber   :", Path(profile["source_path"]).resolve())
    print("Kebijakan:")
    for description in profile["policy_lines"]:
        print("- {}".format(description))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lihat ringkasan profile augmentasi untuk eksperimen training Parkinson.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="all",
        choices=["without_augment", "mycostum_augment", "all"],
        help="Profile augmentasi yang ingin ditampilkan.",
    )
    args = parser.parse_args()

    class_distribution = validate_train_split()
    total_train = sum(class_distribution.values())

    print("\n=== Ringkasan Split Train untuk Eksperimen Augmentasi ===")
    print("Sumber train split :", TRAIN_DIR.resolve())
    print("Jumlah data train  :", total_train)
    print("Ukuran input acuan :", "{}x{}".format(TARGET_SIZE[0], TARGET_SIZE[1]))
    print("-" * 60)
    print("Distribusi train per kelas:")
    for class_name in sorted(class_distribution.keys()):
        print("- {}: {} gambar".format(class_name, class_distribution[class_name]))
    print("-" * 60)

    if args.profile == "all":
        print("Profile augmentasi yang tersedia:")
        for profile in list_available_profiles():
            print_profile(str(profile["name"]))
            print("-" * 60)
    else:
        print_profile(args.profile)
        print("-" * 60)

    print("Catatan:")
    print("- Mode tanpa augmentasi cocok sebagai baseline komparasi.")
    print("- Mode augmentasi kustom membuat versi train yang diperluas secara statis per run.")
    print("- Validation dan testing selalu dievaluasi tanpa augmentasi acak.")


if __name__ == "__main__":
    main()
