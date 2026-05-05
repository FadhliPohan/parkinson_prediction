import os
import random
import shutil
import stat
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


SOURCE_DIR = Path("dataset/original")
OUTPUT_DIR = Path("dataset/split")
SPLIT_NAMES = ["train", "testing", "validation"]
TRAIN_RATIO = 0.70
TEST_RATIO = 0.15
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TRAIN_IMAGE_SIZE = (224, 224)

try:
    BILINEAR_RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    BILINEAR_RESAMPLE = Image.BILINEAR


def list_image_files(folder: Path) -> List[Path]:
    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def count_image_files_recursive(root_dir: Path) -> int:
    if not root_dir.exists():
        return 0
    return sum(
        1
        for file in root_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def delete_image_files_recursive(root_dir: Path) -> int:
    if not root_dir.exists():
        return 0

    deleted_count = 0
    for file in sorted(root_dir.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            file.unlink()
            deleted_count += 1
        except PermissionError:
            try:
                os.chmod(str(file), stat.S_IWRITE)
                file.unlink()
                deleted_count += 1
            except PermissionError:
                print("Peringatan: file terkunci, tidak bisa dihapus -> {}".format(file))
    return deleted_count


def ask_confirmation(prompt_text: str, default_no: bool = True) -> bool:
    default_label = "y/N" if default_no else "Y/n"
    user_input = input("{} [{}]: ".format(prompt_text, default_label)).strip().lower()
    if not user_input:
        return not default_no
    if user_input in ("y", "yes"):
        return True
    if user_input in ("n", "no"):
        return False
    print("Input tidak valid. Proses dibatalkan.")
    return False


def confirm_resplit_if_existing() -> None:
    existing_split = count_image_files_recursive(OUTPUT_DIR)
    if existing_split == 0:
        return

    print("\n=== Konfirmasi Split Ulang ===")
    print("Data split sudah ada: {} file".format(existing_split))
    print("Jika dilanjutkan, folder split lama akan dibersihkan lalu dibuat ulang.")

    is_confirmed = ask_confirmation("Yakin ingin lanjut split ulang?")
    if not is_confirmed:
        print("Split dibatalkan oleh pengguna.")
        raise SystemExit(0)

    deleted_split = delete_image_files_recursive(OUTPUT_DIR)
    print("Data split lama dibersihkan: {} file dihapus dari {}".format(deleted_split, OUTPUT_DIR.resolve()))


def prepare_split_dirs(class_names: List[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in SPLIT_NAMES:
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for class_name in class_names:
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)


def calculate_split_counts(total_count: int) -> Tuple[int, int, int]:
    train_count = int(total_count * TRAIN_RATIO)
    test_count = int(total_count * TEST_RATIO)
    validation_count = total_count - train_count - test_count
    return train_count, test_count, validation_count


def convert_image_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba_image = image.convert("RGBA")
        background = Image.new("RGB", rgba_image.size, (255, 255, 255))
        background.paste(rgba_image, mask=rgba_image.getchannel("A"))
        return background
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def save_split_image(source_path: Path, target_path: Path, target_size: Tuple[int, int] | None) -> None:
    if target_size is None:
        shutil.copy2(source_path, target_path)
        return

    with Image.open(source_path) as image:
        prepared = convert_image_to_rgb(image)
        resized = prepared.resize(target_size, resample=BILINEAR_RESAMPLE)
        save_kwargs = {}
        if target_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 95
            save_kwargs["subsampling"] = 0
        resized.save(target_path, **save_kwargs)


def copy_to_split(files: List[Path], class_name: str, split_name: str) -> int:
    destination = OUTPUT_DIR / split_name / class_name
    copied = 0
    resize_target = TRAIN_IMAGE_SIZE if split_name == "train" else None

    for image_path in files:
        target_path = destination / image_path.name
        save_split_image(image_path, target_path, resize_target)
        copied += 1

    return copied


def print_split_distribution(class_stats: Dict[str, Dict[str, int]]) -> None:
    print("\n=== Distribusi Split Dataset ===")
    print("Sumber data:", SOURCE_DIR.resolve())
    print("Output data:", OUTPUT_DIR.resolve())
    print("Ukuran gambar split train:", "{}x{}".format(TRAIN_IMAGE_SIZE[0], TRAIN_IMAGE_SIZE[1]))
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

    confirm_resplit_if_existing()

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
