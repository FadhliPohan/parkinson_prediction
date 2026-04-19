import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent


def run_python_script(script_path: Path, extra_args: List[str] = None) -> int:
    if extra_args is None:
        extra_args = []

    command = [sys.executable, str(script_path)] + extra_args
    print("\nMenjalankan:", " ".join(command))
    result = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("Perintah gagal dengan exit code:", result.returncode)
    return result.returncode


def run_streamlit_app(app_path: Path) -> int:
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print("\nMenjalankan:", " ".join(command))
    result = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("Streamlit gagal dijalankan. Exit code:", result.returncode)
    return result.returncode


def ask_int(prompt_text: str, default_value: int) -> int:
    user_input = input("{} [{}]: ".format(prompt_text, default_value)).strip()
    if not user_input:
        return default_value
    try:
        return int(user_input)
    except ValueError:
        print("Input tidak valid. Gunakan default:", default_value)
        return default_value


def ask_yes_no(prompt_text: str, default_yes: bool) -> bool:
    default_label = "Y/n" if default_yes else "y/N"
    user_input = input("{} [{}]: ".format(prompt_text, default_label)).strip().lower()
    if not user_input:
        return default_yes
    if user_input in ("y", "yes"):
        return True
    if user_input in ("n", "no"):
        return False
    print("Input tidak valid. Gunakan default.")
    return default_yes


def build_training_args() -> List[str]:
    print("\nKonfigurasi training (tekan Enter untuk pakai default).")
    epochs = ask_int("Epoch stage-1", 8)
    fine_tune_epochs = ask_int("Epoch fine-tuning", 2)
    batch_size = ask_int("Batch size", 16)
    max_per_class = ask_int("Max data per kelas per split (0=tanpa batas)", 0)
    train_batch_limit = ask_int("Limit batch train per epoch (0=tanpa batas)", 0)
    validation_batch_limit = ask_int("Limit batch validation (0=tanpa batas)", 0)
    test_batch_limit = ask_int("Limit batch testing (0=tanpa batas)", 0)
    num_parallel_calls = ask_int("Jumlah worker decode/resize", 2)
    prefetch_buffer = ask_int("Prefetch buffer", 1)
    shuffle_buffer_size = ask_int("Shuffle buffer size", 2048)
    gpu_memory_limit_mb = ask_int("Batas memori GPU MB (0=memory growth)", 0)
    mixed_precision = ask_yes_no("Aktifkan mixed precision (hemat memori GPU)?", True)

    args = [
        "--epochs",
        str(epochs),
        "--fine-tune-epochs",
        str(fine_tune_epochs),
        "--batch-size",
        str(batch_size),
        "--num-parallel-calls",
        str(max(1, num_parallel_calls)),
        "--prefetch-buffer",
        str(max(1, prefetch_buffer)),
        "--shuffle-buffer-size",
        str(max(1, shuffle_buffer_size)),
    ]
    if max_per_class > 0:
        args.extend(["--max-per-class", str(max_per_class)])
    if train_batch_limit > 0:
        args.extend(["--train-batch-limit", str(train_batch_limit)])
    if validation_batch_limit > 0:
        args.extend(["--validation-batch-limit", str(validation_batch_limit)])
    if test_batch_limit > 0:
        args.extend(["--test-batch-limit", str(test_batch_limit)])
    if gpu_memory_limit_mb > 0:
        args.extend(["--gpu-memory-limit-mb", str(gpu_memory_limit_mb)])
    if mixed_precision:
        args.append("--mixed-precision")
    return args


def print_menu() -> None:
    print("\n" + "=" * 60)
    print("Pipeline Klasifikasi Parkinson")
    print("=" * 60)
    print("1. Check distribusi dataset")
    print("2. Augmentasi data (resize + rotasi)")
    print("3. Split data (train/testing/validation)")
    print("4. Training MobileNetV2")
    print("5. Training ResNet50")
    print("6. Training semua model (MobileNetV2 + ResNet50)")
    print("7. Jalankan pipeline penuh (1 -> 6)")
    print("8. Jalankan Dashboard Streamlit")
    print("0. Keluar")


def main() -> None:
    check_script = PROJECT_ROOT / "1.check_dataset.py"
    augment_script = PROJECT_ROOT / "2.augmentasi.py"
    split_script = PROJECT_ROOT / "3.split_data_testing.py"
    mobilenet_script = PROJECT_ROOT / "model" / "mobilenetv2.py"
    resnet_script = PROJECT_ROOT / "model" / "resnet50.py"
    streamlit_app = PROJECT_ROOT / "web" / "app.py"

    while True:
        print_menu()
        choice = input("Pilih aksi: ").strip()

        if choice == "1":
            run_python_script(check_script)
        elif choice == "2":
            run_python_script(augment_script)
        elif choice == "3":
            run_python_script(split_script)
        elif choice == "4":
            args = build_training_args()
            run_python_script(mobilenet_script, args)
        elif choice == "5":
            args = build_training_args()
            run_python_script(resnet_script, args)
        elif choice == "6":
            args = build_training_args()
            first_rc = run_python_script(mobilenet_script, args)
            if first_rc == 0:
                run_python_script(resnet_script, args)
            else:
                print("Training MobileNetV2 gagal/dibatalkan, ResNet50 tidak dijalankan.")
        elif choice == "7":
            rc = run_python_script(check_script)
            if rc != 0:
                continue
            rc = run_python_script(augment_script)
            if rc != 0:
                continue
            rc = run_python_script(split_script)
            if rc != 0:
                continue

            args = build_training_args()
            rc = run_python_script(mobilenet_script, args)
            if rc == 0:
                run_python_script(resnet_script, args)
            else:
                print("Training MobileNetV2 gagal/dibatalkan, ResNet50 tidak dijalankan.")
        elif choice == "8":
            if not streamlit_app.exists():
                print("File dashboard tidak ditemukan:", streamlit_app)
                continue
            run_streamlit_app(streamlit_app)
        elif choice == "0":
            print("Selesai.")
            break
        else:
            print("Pilihan tidak dikenali. Coba lagi.")


if __name__ == "__main__":
    main()
