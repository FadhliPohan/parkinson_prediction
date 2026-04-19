import random
import os
import stat
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


SOURCE_DIR = Path("dataset/praprosesing")
OUTPUT_DIR = Path("dataset/split")
SPLIT_NAMES = ["train", "testing", "validation"]
TRAIN_RATIO = 0.70
TEST_RATIO = 0.15
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CLEAN_OUTPUT_FIRST = False


def list_image_files(folder: Path) -> List[Path]:
    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def prepare_split_dirs(class_names: List[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in SPLIT_NAMES:
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for class_name in class_names:
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            if CLEAN_OUTPUT_FIRST:
                for file in class_dir.iterdir():
                    if file.is_file():
                        try:
                            file.unlink()
                        except PermissionError:
                            try:
                                os.chmod(str(file), stat.S_IWRITE)
                                file.unlink()
                            except PermissionError:
                                print("Peringatan: file terkunci, tidak bisa dihapus -> {}".format(file))


def calculate_split_counts(total_count: int) -> Tuple[int, int, int]:
    train_count = int(total_count * TRAIN_RATIO)
    test_count = int(total_count * TEST_RATIO)
    validation_count = total_count - train_count - test_count
    return train_count, test_count, validation_count


def copy_to_split(files: List[Path], class_name: str, split_name: str) -> int:
    destination = OUTPUT_DIR / split_name / class_name
    copied = 0

    for image_path in files:
        target_path = destination / image_path.name
        shutil.copy2(image_path, target_path)
        copied += 1

    return copied


def print_split_distribution(class_stats: Dict[str, Dict[str, int]]) -> None:
    print("\n=== Distribusi Split Dataset ===")
    print("Sumber data:", SOURCE_DIR.resolve())
    print("Output data:", OUTPUT_DIR.resolve())
    print(
        "Rasio split: train {}% | testing {}% | validation {}%".format(
            int(TRAIN_RATIO * 100),
            int(TEST_RATIO * 100),
            int(VALIDATION_RATIO * 100),
        )
    )
    print("-" * 72)
    print("{:<12s} {:>8s} {:>8s} {:>11s} {:>10s}".format("Kelas", "Train", "Testing", "Validation", "Total"))
    print("-" * 72)

    total_train = 0
    total_test = 0
    total_validation = 0
    total_all = 0

    for class_name in sorted(class_stats.keys()):
        train_count = class_stats[class_name]["train"]
        test_count = class_stats[class_name]["testing"]
        validation_count = class_stats[class_name]["validation"]
        total_count = class_stats[class_name]["total"]

        total_train += train_count
        total_test += test_count
        total_validation += validation_count
        total_all += total_count

        print(
            "{:<12s} {:>8d} {:>8d} {:>11d} {:>10d}".format(
                class_name, train_count, test_count, validation_count, total_count
            )
        )

    print("-" * 72)
    print(
        "{:<12s} {:>8d} {:>8d} {:>11d} {:>10d}".format(
            "Total", total_train, total_test, total_validation, total_all
        )
    )
    print("-" * 72)

    if total_all:
        print(
            "Persentase total: train {:.2f}% | testing {:.2f}% | validation {:.2f}%".format(
                (float(total_train) / float(total_all)) * 100.0,
                (float(total_test) / float(total_all)) * 100.0,
                (float(total_validation) / float(total_all)) * 100.0,
            )
        )


def run_split() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError("Folder sumber tidak ditemukan: {}".format(SOURCE_DIR.resolve()))

    class_dirs = sorted([path for path in SOURCE_DIR.iterdir() if path.is_dir()])
    if not class_dirs:
        raise ValueError("Tidak ada folder kelas pada {}".format(SOURCE_DIR.resolve()))

    class_names = [class_dir.name for class_dir in class_dirs]
    prepare_split_dirs(class_names)

    rng = random.Random(RANDOM_SEED)
    class_stats: Dict[str, Dict[str, int]] = {}

    for class_dir in class_dirs:
        image_files = list_image_files(class_dir)
        if not image_files:
            class_stats[class_dir.name] = {"train": 0, "testing": 0, "validation": 0, "total": 0}
            continue

        shuffled = image_files[:]
        rng.shuffle(shuffled)

        train_count, test_count, validation_count = calculate_split_counts(len(shuffled))

        train_files = shuffled[:train_count]
        test_files = shuffled[train_count : train_count + test_count]
        validation_files = shuffled[train_count + test_count :]

        copied_train = copy_to_split(train_files, class_dir.name, "train")
        copied_test = copy_to_split(test_files, class_dir.name, "testing")
        copied_validation = copy_to_split(validation_files, class_dir.name, "validation")

        class_stats[class_dir.name] = {
            "train": copied_train,
            "testing": copied_test,
            "validation": copied_validation,
            "total": copied_train + copied_test + copied_validation,
        }

    print_split_distribution(class_stats)
    print("Split selesai.")


if __name__ == "__main__":
    run_split()
