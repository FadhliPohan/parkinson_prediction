import os
import stat
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError


SOURCE_DIR = Path("dataset/original")
OUTPUT_DIR = Path("dataset/praprosesing")
TARGET_SIZE: Tuple[int, int] = (227, 227)
ROTATION_ANGLES = [90, 180, 270]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CLEAN_OUTPUT_FIRST = False


def get_resample_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def list_image_files(folder: Path) -> List[Path]:
    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def reset_output_dirs(class_names: List[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        class_output_dir = OUTPUT_DIR / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        if CLEAN_OUTPUT_FIRST:
            for file in class_output_dir.iterdir():
                if file.is_file():
                    try:
                        file.unlink()
                    except PermissionError:
                        try:
                            os.chmod(str(file), stat.S_IWRITE)
                            file.unlink()
                        except PermissionError:
                            print("Peringatan: file terkunci, tidak bisa dihapus -> {}".format(file))


def save_augmented_variants(image: Image.Image, output_dir: Path, image_stem: str) -> int:
    saved_count = 0
    output_files = [(image, "rot0")]
    for angle in ROTATION_ANGLES:
        output_files.append((image.rotate(angle, expand=False), "rot{}".format(angle)))

    for img_variant, suffix in output_files:
        output_path = output_dir / f"{image_stem}_{suffix}.png"
        img_variant.save(output_path)
        saved_count += 1

    return saved_count


def print_augmentation_distribution(
    class_result: Dict[str, Dict[str, int]],
    variant_counts: Dict[str, int],
) -> None:
    print("\n=== Distribusi Augmentasi ===")
    total_output = sum(result["output_images"] for result in class_result.values())
    print("Total hasil augmentasi:", total_output)
    print("-" * 55)

    print("Per kelas:")
    for class_name in sorted(class_result.keys()):
        output_count = class_result[class_name]["output_images"]
        percentage = (float(output_count) / float(total_output) * 100.0) if total_output else 0.0
        print("{:<12s}: {:5d} ({:6.2f}%)".format(class_name, output_count, percentage))

    print("-" * 55)
    print("Per tipe augmentasi:")
    for variant in sorted(variant_counts.keys()):
        count = variant_counts[variant]
        percentage = (float(count) / float(total_output) * 100.0) if total_output else 0.0
        print("{:<12s}: {:5d} ({:6.2f}%)".format(variant, count, percentage))
    print("-" * 55)


def process_dataset() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            "Folder sumber tidak ditemukan: {}".format(SOURCE_DIR.resolve())
        )

    class_dirs = sorted([p for p in SOURCE_DIR.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError("Tidak ada folder kelas di dalam {}".format(SOURCE_DIR.resolve()))

    class_names = [class_dir.name for class_dir in class_dirs]
    reset_output_dirs(class_names)

    resample_filter = get_resample_filter()
    class_result: Dict[str, Dict[str, int]] = {}
    variant_counts: Dict[str, int] = {"rot0": 0, "rot90": 0, "rot180": 0, "rot270": 0}

    for class_dir in class_dirs:
        image_files = list_image_files(class_dir)
        output_class_dir = OUTPUT_DIR / class_dir.name

        processed = 0
        skipped = 0
        saved = 0

        for image_path in image_files:
            try:
                with Image.open(image_path) as img:
                    rgb_img = img.convert("RGB")
                    resized_img = rgb_img.resize(TARGET_SIZE, resample=resample_filter)
                    saved += save_augmented_variants(
                        image=resized_img,
                        output_dir=output_class_dir,
                        image_stem=image_path.stem,
                    )
                    variant_counts["rot0"] += 1
                    variant_counts["rot90"] += 1
                    variant_counts["rot180"] += 1
                    variant_counts["rot270"] += 1
                    processed += 1
            except (UnidentifiedImageError, OSError):
                skipped += 1

        class_result[class_dir.name] = {
            "input_images": len(image_files),
            "processed_images": processed,
            "skipped_images": skipped,
            "output_images": saved,
        }

    print("\n=== Ringkasan Augmentasi ===")
    print("Sumber data   :", SOURCE_DIR.resolve())
    print("Output data   :", OUTPUT_DIR.resolve())
    print("Ukuran output :", "{}x{}".format(TARGET_SIZE[0], TARGET_SIZE[1]))
    print("-" * 55)

    total_input = 0
    total_output = 0

    for class_name in sorted(class_result.keys()):
        result = class_result[class_name]
        total_input += result["input_images"]
        total_output += result["output_images"]
        print(
            "{:<12s} | input: {:4d} | diproses: {:4d} | skip: {:3d} | output: {:5d}".format(
                class_name,
                result["input_images"],
                result["processed_images"],
                result["skipped_images"],
                result["output_images"],
            )
        )

    print("-" * 55)
    print("Total input  :", total_input)
    print("Total output :", total_output)
    print_augmentation_distribution(class_result, variant_counts)
    print("Done.")


if __name__ == "__main__":
    process_dataset()
