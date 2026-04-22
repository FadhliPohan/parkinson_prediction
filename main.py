import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def build_runtime_env() -> dict:
    env = os.environ.copy()
    lib_dirs: List[str] = []

    nvidia_root_candidates = sorted(
        (VENV_DIR / "lib").glob("python*/site-packages/nvidia")
    )
    for nvidia_root in nvidia_root_candidates:
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                lib_dirs.append(str(lib_dir))

    if lib_dirs:
        current_ld_path = env.get("LD_LIBRARY_PATH", "")
        prefix = ":".join(lib_dirs)
        env["LD_LIBRARY_PATH"] = "{}:{}".format(prefix, current_ld_path) if current_ld_path else prefix

    return env


def get_venv_python_path() -> Path:
    if sys.platform.startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def get_runtime_python() -> Path:
    venv_python = get_venv_python_path()
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def run_command(command: List[str], failure_prefix: str) -> int:
    print("\nMenjalankan:", " ".join(command))
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=build_runtime_env())
    if result.returncode != 0:
        print("{} {}".format(failure_prefix, result.returncode))
    return result.returncode


def install_environment() -> int:
    venv_python = get_venv_python_path()
    if not venv_python.exists():
        print("\nVirtual environment belum ada, membuat .venv ...")
        create_rc = run_command(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            "Gagal membuat virtual environment. Exit code:",
        )
        if create_rc != 0:
            return create_rc
    else:
        print("\nVirtual environment sudah ada:", VENV_DIR)

    venv_python = get_venv_python_path()
    if not REQUIREMENTS_FILE.exists():
        print("requirements.txt tidak ditemukan:", REQUIREMENTS_FILE)
        return 1

    upgrade_rc = run_command(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrade pip gagal. Exit code:",
    )
    if upgrade_rc != 0:
        return upgrade_rc

    if REQUIREMENTS_FILE.stat().st_size == 0:
        print("requirements.txt kosong. Tidak ada dependency yang diinstall.")
        return 0

    install_rc = run_command(
        [str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        "Instalasi dependency gagal. Exit code:",
    )
    if install_rc == 0:
        print("Instalasi selesai.")
    return install_rc


def run_python_script(script_path: Path, extra_args: List[str] = None) -> int:
    if extra_args is None:
        extra_args = []

    command = [str(get_runtime_python()), str(script_path)] + extra_args
    print("\nMenjalankan:", " ".join(command))
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=build_runtime_env())
    if result.returncode != 0:
        print("Perintah gagal dengan exit code:", result.returncode)
    return result.returncode


def run_streamlit_app(app_path: Path) -> int:
    command = [str(get_runtime_python()), "-m", "streamlit", "run", str(app_path)]
    print("\nMenjalankan:", " ".join(command))
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=build_runtime_env())
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
    max_cpu_usage_percent = ask_int("Target maksimum penggunaan CPU (%)", 70)
    cpu_thread_limit = ask_int("Batas thread CPU absolut (0=otomatis)", 0)
    allow_cpu_fallback = ask_yes_no("Jika GPU penuh/tidak ada, fallback ke CPU?", True)
    mixed_precision = ask_yes_no("Aktifkan mixed precision (hemat memori GPU)?", True)

    safe_cpu_percent = min(100, max(10, max_cpu_usage_percent))
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
        "--max-cpu-usage-percent",
        str(safe_cpu_percent),
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
    if cpu_thread_limit > 0:
        args.extend(["--cpu-thread-limit", str(cpu_thread_limit)])
    if not allow_cpu_fallback:
        args.append("--disable-cpu-fallback")
    if mixed_precision:
        args.append("--mixed-precision")
    return args


def find_existing_model_run(model_name: str) -> Optional[Path]:
    model_root = PROJECT_ROOT / "trained_models" / model_name
    if not model_root.exists() or not model_root.is_dir():
        return None

    def has_model_artifact(run_dir: Path) -> bool:
        return any(run_dir.glob("*.keras")) or any(run_dir.glob("*.pt"))

    latest_run_file = model_root / "latest_run.txt"
    if latest_run_file.exists():
        run_id = latest_run_file.read_text(encoding="utf-8").strip()
        if run_id:
            run_dir = model_root / run_id
            if run_dir.exists() and run_dir.is_dir() and has_model_artifact(run_dir):
                return run_dir

    run_dirs = sorted([path for path in model_root.iterdir() if path.is_dir()], reverse=True)
    for run_dir in run_dirs:
        if has_model_artifact(run_dir):
            return run_dir
    return None


def should_train_model(model_name: str) -> bool:
    existing_run = find_existing_model_run(model_name)
    if existing_run is None:
        return True

    print("\nModel {} sudah pernah di-training.".format(model_name))
    print("Lokasi model sebelumnya:", existing_run)
    return ask_yes_no("Lanjutkan training ulang {}?".format(model_name), False)


def run_training_script(script_path: Path, model_name: str, args: List[str]) -> Optional[int]:
    if not should_train_model(model_name):
        print("Training {} dilewati.".format(model_name))
        return None
    return run_python_script(script_path, args)


def print_menu() -> None:
    print("\n" + "=" * 60)
    print("Pipeline Klasifikasi Parkinson")
    print("=" * 60)
    print("1. Instalasi dependency + virtual environment")
    print("2. Check distribusi dataset")
    print("3. Augmentasi data (resize + rotasi)")
    print("4. Split data (train/testing/validation)")
    print("5. Training MobileNetV2")
    print("6. Training ResNet50")
    print("7. Training YOLOv8")
    print("8. Training semua model (MobileNetV2 + ResNet50 + YOLOv8)")
    print("9. Jalankan pipeline penuh (2 -> 8)")
    print("10. Jalankan Dashboard Streamlit")
    print("0. Keluar")


def main() -> None:
    check_script = PROJECT_ROOT / "1.check_dataset.py"
    augment_script = PROJECT_ROOT / "2.augmentasi.py"
    split_script = PROJECT_ROOT / "3.split_data_testing.py"
    mobilenet_script = PROJECT_ROOT / "model" / "mobilenetv2.py"
    resnet_script = PROJECT_ROOT / "model" / "resnet50.py"
    yolov8_script = PROJECT_ROOT / "model" / "yolov8.py"
    streamlit_app = PROJECT_ROOT / "web" / "app.py"

    while True:
        print_menu()
        choice = input("Pilih aksi: ").strip()

        if choice == "1":
            install_environment()
        elif choice == "2":
            run_python_script(check_script)
        elif choice == "3":
            run_python_script(augment_script)
        elif choice == "4":
            run_python_script(split_script)
        elif choice == "5":
            args = build_training_args()
            run_training_script(mobilenet_script, "mobilenetv2", args)
        elif choice == "6":
            args = build_training_args()
            run_training_script(resnet_script, "resnet50", args)
        elif choice == "7":
            args = build_training_args()
            run_training_script(yolov8_script, "yolov8", args)
        elif choice == "8":
            args = build_training_args()
            first_rc = run_training_script(mobilenet_script, "mobilenetv2", args)
            if first_rc is not None and first_rc != 0:
                print("Training MobileNetV2 gagal/dibatalkan, model berikutnya tidak dijalankan.")
                continue
            second_rc = run_training_script(resnet_script, "resnet50", args)
            if second_rc is not None and second_rc != 0:
                print("Training ResNet50 gagal/dibatalkan, YOLOv8 tidak dijalankan.")
                continue
            run_training_script(yolov8_script, "yolov8", args)
        elif choice == "9":
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
            first_rc = run_training_script(mobilenet_script, "mobilenetv2", args)
            if first_rc is not None and first_rc != 0:
                print("Training MobileNetV2 gagal/dibatalkan, model berikutnya tidak dijalankan.")
                continue
            second_rc = run_training_script(resnet_script, "resnet50", args)
            if second_rc is not None and second_rc != 0:
                print("Training ResNet50 gagal/dibatalkan, YOLOv8 tidak dijalankan.")
                continue
            run_training_script(yolov8_script, "yolov8", args)
        elif choice == "10":
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
