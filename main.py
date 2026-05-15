from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from src.datasets.registry import DatasetRegistry
from src.models.registry import ModelRegistry
from src.training.strategies import TrainingMethodRegistry
from src.utils.paths import PROJECT_ROOT
from src.utils.runtime import build_runtime_env, get_runtime_python, get_venv_python_path


REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
TRAINING_DIR = PROJECT_ROOT / "training"


def _resolve_script(script_name: str) -> Path:
    candidates = [
        TRAINING_DIR / script_name,
        PROJECT_ROOT / script_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Script tidak ditemukan: {}. Sudah cek: {}".format(
            script_name,
            ", ".join(str(item) for item in candidates),
        )
    )


def run_command(command: List[str], cwd: Optional[Path] = None) -> int:
    print("\nMenjalankan:", " ".join(command))
    result = subprocess.run(command, cwd=str(cwd or PROJECT_ROOT), env=build_runtime_env())
    if result.returncode != 0:
        print("Perintah gagal dengan exit code:", result.returncode)
    return int(result.returncode)


def install_environment() -> int:
    venv_python = get_venv_python_path()
    if not venv_python.exists():
        print("\nVirtual environment belum ada, membuat .venv ...")
        rc = run_command([sys.executable, "-m", "venv", str(PROJECT_ROOT / ".venv")])
        if rc != 0:
            return rc

    if not REQUIREMENTS_FILE.exists():
        print("requirements.txt tidak ditemukan:", REQUIREMENTS_FILE)
        return 1

    venv_python = get_venv_python_path()
    rc = run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    if rc != 0:
        return rc

    if REQUIREMENTS_FILE.stat().st_size == 0:
        print("requirements.txt kosong. Tidak ada dependency yang diinstall.")
        return 0

    rc = run_command([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
    if rc == 0:
        print("Instalasi selesai.")
    return rc


def ask_choice(prompt: str, options: List[str], default_index: int = 0) -> str:
    if not options:
        raise ValueError("Tidak ada opsi")

    print()
    for idx, option in enumerate(options, start=1):
        default_label = " (default)" if idx - 1 == default_index else ""
        print(f"{idx}. {option}{default_label}")

    raw = input(f"{prompt} [{default_index + 1}]: ").strip()
    if not raw:
        return options[default_index]

    try:
        selected = int(raw)
        if 1 <= selected <= len(options):
            return options[selected - 1]
    except ValueError:
        pass

    print("Input tidak valid, gunakan default.")
    return options[default_index]


def ask_yes_no(prompt: str, default_yes: bool = False) -> bool:
    default_label = "Y/n" if default_yes else "y/N"
    raw = input(f"{prompt} [{default_label}]: ").strip().lower()
    if not raw:
        return default_yes
    if raw in {"y", "yes"}:
        return True
    if raw in {"n", "no"}:
        return False
    print("Input tidak valid, gunakan default.")
    return default_yes


def ask_optional_int(prompt: str) -> Optional[int]:
    raw = input(f"{prompt} [kosong=default]: ").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        print("Input tidak valid, pakai default.")
        return None


def ask_preprocessing_mode() -> str:
    options = [
        "augment (gunakan augmentasi on-the-fly)",
        "no_augment (tanpa augmentasi)",
        "both (jalankan keduanya)",
    ]
    selected = ask_choice("Pilih mode preprocessing", options, default_index=0)
    return selected.split(" ", 1)[0].strip()


def build_train_command(
    dataset_id: str,
    models: str,
    method_id: Optional[str],
    all_methods: bool,
    check_first: bool,
    split_first: bool,
    augment_info: bool,
    preprocessing_mode: str,
) -> List[str]:
    train_script = _resolve_script("train.py")
    command = [str(get_runtime_python()), str(train_script), "--dataset", dataset_id, "--models", models]
    command.extend(["--preprocessing-mode", preprocessing_mode])

    if method_id:
        command.extend(["--method", method_id])
    if all_methods:
        command.append("--all-methods")
    if check_first:
        command.append("--check-first")
    if split_first:
        command.append("--split-first")
    if augment_info:
        command.append("--augment-info")

    epochs = ask_optional_int("Override epoch stage-1")
    batch_size = ask_optional_int("Override batch size")
    fine_tune_epochs = ask_optional_int("Override fine-tune epochs")

    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    if fine_tune_epochs is not None:
        command.extend(["--fine-tune-epochs", str(fine_tune_epochs)])

    if ask_yes_no("Aktifkan mixed precision?", False):
        command.append("--mixed-precision")

    if ask_yes_no("Nonaktifkan fallback CPU?", False):
        command.append("--disable-cpu-fallback")

    return command


def run_check_dataset(dataset_id: str) -> int:
    script = _resolve_script("1.check_dataset.py")
    command = [str(get_runtime_python()), str(script), "--dataset", dataset_id]
    return run_command(command)


def run_split_dataset(dataset_id: str) -> int:
    script = _resolve_script("2.split_data_testing.py")
    command = [str(get_runtime_python()), str(script), "--dataset", dataset_id]
    return run_command(command)


def run_augment_info(dataset_split_dir: str) -> int:
    script = _resolve_script("3.augmentasi.py")
    command = [
        str(get_runtime_python()),
        str(script),
        "--dataset-dir",
        str(dataset_split_dir),
    ]
    return run_command(command)


def run_streamlit_dashboard() -> int:
    command = [str(get_runtime_python()), "-m", "streamlit", "run", str(PROJECT_ROOT / "web" / "app.py")]
    return run_command(command)


def print_menu() -> None:
    print("\n" + "=" * 68)
    print("Pipeline Klasifikasi Parkinson (Dynamic Registry Mode)")
    print("=" * 68)
    print("1. Instalasi dependency + virtual environment")
    print("2. Check distribusi dataset")
    print("3. Split dataset")
    print("4. Info kebijakan augmentasi train")
    print("5. Training satu model")
    print("6. Training semua model (satu method)")
    print("7. Training semua model + semua method")
    print("8. Pipeline penuh (check -> split -> augment -> train all models)")
    print("9. Jalankan dashboard Streamlit")
    print("0. Keluar")


def main() -> None:
    while True:
        dataset_registry = DatasetRegistry()
        model_registry = ModelRegistry()
        method_registry = TrainingMethodRegistry()

        dataset_ids = dataset_registry.list_dataset_ids()
        model_ids = model_registry.list_model_ids(enabled_only=True)
        method_ids = method_registry.list_method_ids()

        print_menu()
        choice = input("Pilih aksi: ").strip()

        if choice == "0":
            print("Selesai.")
            break

        if choice == "1":
            install_environment()
            continue

        selected_dataset = ask_choice("Pilih dataset", dataset_ids, default_index=dataset_ids.index(dataset_registry.default_dataset))
        dataset_cfg = dataset_registry.get(selected_dataset)

        if choice == "2":
            run_check_dataset(selected_dataset)
        elif choice == "3":
            run_split_dataset(selected_dataset)
        elif choice == "4":
            run_augment_info(str(dataset_cfg.split_path))
        elif choice == "5":
            selected_model = ask_choice("Pilih model", model_ids)
            selected_method = ask_choice("Pilih method", method_ids, default_index=method_ids.index(method_registry.default_method))
            command = build_train_command(
                dataset_id=selected_dataset,
                models=selected_model,
                method_id=selected_method,
                all_methods=False,
                check_first=False,
                split_first=False,
                augment_info=False,
                preprocessing_mode=ask_preprocessing_mode(),
            )
            run_command(command)
        elif choice == "6":
            selected_method = ask_choice("Pilih method", method_ids, default_index=method_ids.index(method_registry.default_method))
            command = build_train_command(
                dataset_id=selected_dataset,
                models="all",
                method_id=selected_method,
                all_methods=False,
                check_first=False,
                split_first=False,
                augment_info=False,
                preprocessing_mode=ask_preprocessing_mode(),
            )
            run_command(command)
        elif choice == "7":
            command = build_train_command(
                dataset_id=selected_dataset,
                models="all",
                method_id=None,
                all_methods=True,
                check_first=False,
                split_first=False,
                augment_info=False,
                preprocessing_mode=ask_preprocessing_mode(),
            )
            run_command(command)
        elif choice == "8":
            use_all_methods = ask_yes_no("Gunakan semua method training?", True)
            if use_all_methods:
                method_id = None
            else:
                method_id = ask_choice(
                    "Pilih method",
                    method_ids,
                    default_index=method_ids.index(method_registry.default_method),
                )

            command = build_train_command(
                dataset_id=selected_dataset,
                models="all",
                method_id=method_id,
                all_methods=use_all_methods,
                check_first=True,
                split_first=True,
                augment_info=True,
                preprocessing_mode=ask_preprocessing_mode(),
            )
            run_command(command)
        elif choice == "9":
            run_streamlit_dashboard()
        else:
            print("Pilihan tidak dikenali. Coba lagi.")


if __name__ == "__main__":
    main()
