from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.datasets.registry import DatasetRegistry
from src.models.registry import ModelRegistry
from src.training.augmentations import TrainingAugmentationRegistry
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


def _parse_csv_selection(raw: str, options: List[str]) -> List[str]:
    normalized = str(raw).strip().lower()
    if normalized in {"all", "*"}:
        return list(options)
    selected = [item.strip() for item in str(raw).split(",") if item.strip()]
    return [item for item in selected if item in options]


def ask_method_runtime_overrides(method_ids: List[str]) -> Dict[str, Dict[str, int]]:
    if not method_ids:
        return {}

    print("\n=== Konfigurasi Runtime per Method ===")
    print("Kosongkan input jika ingin pakai default method.")
    overrides: Dict[str, Dict[str, int]] = {}

    for method_id in method_ids:
        print(f"\nMethod: {method_id}")
        epochs = ask_optional_int("Epoch stage-1")
        batch_size = ask_optional_int("Batch size")
        fine_tune_epochs = ask_optional_int("Fine-tune epochs")

        method_override: Dict[str, int] = {}
        if epochs is not None:
            method_override["epochs"] = epochs
        if batch_size is not None:
            method_override["batch_size"] = batch_size
        if fine_tune_epochs is not None:
            method_override["fine_tune_epochs"] = fine_tune_epochs

        if method_override:
            overrides[method_id] = method_override

    return overrides


def ask_runtime_toggles() -> Dict[str, bool]:
    return {
        "mixed_precision": ask_yes_no("Aktifkan mixed precision?", False),
        "disable_cpu_fallback": ask_yes_no("Nonaktifkan fallback CPU?", False),
    }


def ask_shutdown_on_finish() -> bool:
    return ask_yes_no(
        "Aktifkan auto-shutdown perangkat setelah workflow training selesai?",
        False,
    )


def ask_csv_or_all(prompt: str, options: List[str], default: str = "all") -> str:
    print("\nOpsi tersedia:")
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    print("Ketik 'all' untuk semua opsi.")
    print("Bisa juga pilih beberapa sekaligus, contoh: 1,3 atau mobilenetv2,resnet50")

    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default

    normalized = raw.lower()
    if normalized in {"all", "*"}:
        return "all"

    selected: List[str] = []
    for token in [item.strip() for item in raw.split(",") if item.strip()]:
        if token.isdigit():
            index = int(token)
            if 1 <= index <= len(options):
                selected.append(options[index - 1])
                continue
            print(f"Index {index} di luar rentang, diabaikan.")
            continue

        if token in options:
            selected.append(token)
            continue

        print(f"Opsi '{token}' tidak dikenal, diabaikan.")

    if not selected:
        print("Tidak ada opsi valid, gunakan default.")
        return default

    deduped: List[str] = []
    seen = set()
    for item in selected:
        if item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return ",".join(deduped)


def ask_preprocessing_mode() -> str:
    options = [
        "augment (gunakan augmentasi on-the-fly)",
        "no_augment (tanpa augmentasi)",
        "both (jalankan keduanya)",
    ]
    selected = ask_choice("Pilih mode preprocessing", options, default_index=0)
    return selected.split(" ", 1)[0].strip()


def ask_on_existing_mode() -> str:
    options = [
        "ask (tanya sekali di awal jika ada kombinasi existing)",
        "retrain (langsung training ulang semua kombinasi yang sudah ada)",
        "skip (lewati kombinasi yang sudah pernah training)",
    ]
    selected = ask_choice("Perilaku jika kombinasi sudah pernah ditraining", options, default_index=0)
    return selected.split(" ", 1)[0].strip()


def ask_on_existing_split_mode() -> str:
    options = [
        "ask (tanya jika folder split sudah ada)",
        "resplit (langsung split ulang)",
        "skip (pakai split lama tanpa split ulang)",
    ]
    selected = ask_choice("Perilaku jika folder split sudah ada", options, default_index=0)
    return selected.split(" ", 1)[0].strip()


def ask_dataset_target(dataset_registry: DatasetRegistry) -> str:
    dataset_ids = dataset_registry.list_dataset_ids()
    option_all = "all (jalankan berurutan semua dataset)"
    options = dataset_ids + [option_all]
    default_index = dataset_ids.index(dataset_registry.default_dataset)
    selected = ask_choice("Pilih dataset", options, default_index=default_index)
    if selected == option_all:
        return "all"
    return selected


def resolve_dataset_targets(dataset_registry: DatasetRegistry, selected_dataset: str) -> List[str]:
    if str(selected_dataset).strip().lower() in {"all", "*"}:
        return dataset_registry.list_dataset_ids()
    return [selected_dataset]


def build_train_command(
    dataset_id: str,
    models: str,
    method_id: Optional[str],
    all_methods: bool,
    check_first: bool,
    split_first: bool,
    augment_info: bool,
    preprocessing_mode: str,
    augmentations: Optional[str] = None,
    on_existing: str = "ask",
    on_existing_split: str = "ask",
    method_overrides: Optional[Dict[str, Dict[str, int]]] = None,
    model_families: Optional[str] = None,
    mixed_precision: bool = False,
    disable_cpu_fallback: bool = False,
    shutdown_on_finish: bool = False,
) -> List[str]:
    train_script = _resolve_script("train.py")
    command = [str(get_runtime_python()), str(train_script), "--dataset", dataset_id, "--models", models]
    if augmentations:
        command.extend(["--augmentations", augmentations])
    else:
        command.extend(["--preprocessing-mode", preprocessing_mode])

    if method_id:
        command.extend(["--method", method_id])
    if model_families:
        command.extend(["--model-families", model_families])
    if all_methods:
        command.append("--all-methods")
    if check_first:
        command.append("--check-first")
    if split_first:
        command.append("--split-first")
        command.extend(["--on-existing-split", on_existing_split])
    if augment_info:
        command.append("--augment-info")
    command.extend(["--on-existing", on_existing])

    if method_overrides:
        command.extend(["--method-overrides-json", json.dumps(method_overrides)])

    if mixed_precision:
        command.append("--mixed-precision")

    if disable_cpu_fallback:
        command.append("--disable-cpu-fallback")

    if shutdown_on_finish:
        command.append("--shutdown-on-finish")

    return command


def run_check_dataset(dataset_id: str) -> int:
    script = _resolve_script("1.check_dataset.py")
    command = [str(get_runtime_python()), str(script), "--dataset", dataset_id]
    return run_command(command)


def run_split_dataset(dataset_id: str, on_existing_split: str = "ask") -> int:
    script = _resolve_script("2.split_data_testing.py")
    command = [
        str(get_runtime_python()),
        str(script),
        "--dataset",
        dataset_id,
        "--on-existing-split",
        on_existing_split,
    ]
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
    print("10. Workflow training fleksibel (multi dataset/augmentasi/method/model)")
    print("11. Pipeline penuh model CNN")
    print("12. Pipeline penuh model Transformer")
    print("0. Keluar")


def main() -> None:
    while True:
        dataset_registry = DatasetRegistry()
        model_registry = ModelRegistry()
        method_registry = TrainingMethodRegistry()
        augmentation_registry = TrainingAugmentationRegistry()

        model_ids = model_registry.list_model_ids(enabled_only=True)
        family_ids = model_registry.list_family_ids(enabled_only=True)
        method_ids = method_registry.list_method_ids()
        augmentation_ids = augmentation_registry.list_augmentation_ids()

        print_menu()
        choice = input("Pilih aksi: ").strip()

        if choice == "0":
            print("Selesai.")
            break

        if choice == "1":
            install_environment()
            continue

        selected_dataset = ""
        target_datasets: List[str] = []
        if choice in {"2", "3", "4", "5", "6", "7", "8", "11", "12"}:
            selected_dataset = ask_dataset_target(dataset_registry)
            target_datasets = resolve_dataset_targets(dataset_registry, selected_dataset)

        if choice == "2":
            for dataset_id in target_datasets:
                run_check_dataset(dataset_id)
        elif choice == "3":
            on_existing_split_mode = ask_on_existing_split_mode()
            for dataset_id in target_datasets:
                run_split_dataset(dataset_id, on_existing_split=on_existing_split_mode)
        elif choice == "4":
            for dataset_id in target_datasets:
                dataset_cfg = dataset_registry.get(dataset_id)
                run_augment_info(str(dataset_cfg.split_path))
        elif choice == "5":
            selected_model = ask_choice("Pilih model", model_ids)
            selected_method = ask_choice("Pilih method", method_ids, default_index=method_ids.index(method_registry.default_method))
            on_existing_mode = ask_on_existing_mode()
            method_overrides = ask_method_runtime_overrides([selected_method])
            runtime_toggles = ask_runtime_toggles()
            command = build_train_command(
                dataset_id=selected_dataset,
                models=selected_model,
                method_id=selected_method,
                all_methods=False,
                check_first=False,
                split_first=False,
                augment_info=False,
                preprocessing_mode=ask_preprocessing_mode(),
                on_existing=on_existing_mode,
                method_overrides=method_overrides,
                mixed_precision=runtime_toggles["mixed_precision"],
                disable_cpu_fallback=runtime_toggles["disable_cpu_fallback"],
                shutdown_on_finish=ask_shutdown_on_finish(),
            )
            run_command(command)
        elif choice == "6":
            selected_method = ask_choice("Pilih method", method_ids, default_index=method_ids.index(method_registry.default_method))
            on_existing_mode = ask_on_existing_mode()
            method_overrides = ask_method_runtime_overrides([selected_method])
            runtime_toggles = ask_runtime_toggles()
            command = build_train_command(
                dataset_id=selected_dataset,
                models="all",
                method_id=selected_method,
                all_methods=False,
                check_first=False,
                split_first=False,
                augment_info=False,
                preprocessing_mode=ask_preprocessing_mode(),
                on_existing=on_existing_mode,
                method_overrides=method_overrides,
                mixed_precision=runtime_toggles["mixed_precision"],
                disable_cpu_fallback=runtime_toggles["disable_cpu_fallback"],
                shutdown_on_finish=ask_shutdown_on_finish(),
            )
            run_command(command)
        elif choice == "7":
            on_existing_mode = ask_on_existing_mode()
            method_overrides = ask_method_runtime_overrides(method_ids)
            runtime_toggles = ask_runtime_toggles()
            command = build_train_command(
                dataset_id=selected_dataset,
                models="all",
                method_id=None,
                all_methods=True,
                check_first=False,
                split_first=False,
                augment_info=False,
                preprocessing_mode=ask_preprocessing_mode(),
                on_existing=on_existing_mode,
                method_overrides=method_overrides,
                mixed_precision=runtime_toggles["mixed_precision"],
                disable_cpu_fallback=runtime_toggles["disable_cpu_fallback"],
                shutdown_on_finish=ask_shutdown_on_finish(),
            )
            run_command(command)
        elif choice == "8":
            use_all_methods = ask_yes_no("Gunakan semua method training?", True)
            if use_all_methods:
                method_id = None
                selected_method_ids = method_ids
            else:
                method_id = ask_choice(
                    "Pilih method",
                    method_ids,
                    default_index=method_ids.index(method_registry.default_method),
                )
                selected_method_ids = [method_id]

            on_existing_split_mode = ask_on_existing_split_mode()
            on_existing_mode = ask_on_existing_mode()
            method_overrides = ask_method_runtime_overrides(selected_method_ids)
            runtime_toggles = ask_runtime_toggles()
            command = build_train_command(
                dataset_id=selected_dataset,
                models="all",
                method_id=method_id,
                all_methods=use_all_methods,
                check_first=True,
                split_first=True,
                augment_info=True,
                preprocessing_mode=ask_preprocessing_mode(),
                on_existing=on_existing_mode,
                on_existing_split=on_existing_split_mode,
                method_overrides=method_overrides,
                mixed_precision=runtime_toggles["mixed_precision"],
                disable_cpu_fallback=runtime_toggles["disable_cpu_fallback"],
                shutdown_on_finish=ask_shutdown_on_finish(),
            )
            run_command(command)
        elif choice == "9":
            run_streamlit_dashboard()
        elif choice == "10":
            dataset_arg = ask_csv_or_all(
                "Pilih dataset target",
                dataset_registry.list_dataset_ids(),
                default=dataset_registry.default_dataset,
            )
            augmentations_arg = ask_csv_or_all(
                "Pilih augmentasi",
                augmentation_ids,
                default="all",
            )
            methods_arg = ask_csv_or_all(
                "Pilih method training",
                method_ids,
                default=method_registry.default_method,
            )
            models_arg = ask_csv_or_all(
                "Pilih model",
                model_ids,
                default="all",
            )
            model_families_arg = ask_csv_or_all(
                "Pilih family model (opsional)",
                family_ids,
                default="all",
            )
            selected_method_ids = _parse_csv_selection(methods_arg, method_ids)
            method_overrides = ask_method_runtime_overrides(selected_method_ids)
            on_existing_mode = ask_on_existing_mode()
            split_first = ask_yes_no("Jalankan split dataset dulu?", False)
            on_existing_split_mode = "ask"
            if split_first:
                on_existing_split_mode = ask_on_existing_split_mode()
            runtime_toggles = ask_runtime_toggles()

            command = build_train_command(
                dataset_id=dataset_arg,
                models=models_arg,
                method_id=methods_arg,
                all_methods=False,
                check_first=ask_yes_no("Jalankan dataset check dulu?", False),
                split_first=split_first,
                augment_info=ask_yes_no("Tampilkan info augmentasi sebelum training?", False),
                preprocessing_mode="augment",
                augmentations=augmentations_arg,
                on_existing=on_existing_mode,
                on_existing_split=on_existing_split_mode,
                method_overrides=method_overrides,
                model_families=model_families_arg,
                mixed_precision=runtime_toggles["mixed_precision"],
                disable_cpu_fallback=runtime_toggles["disable_cpu_fallback"],
                shutdown_on_finish=ask_shutdown_on_finish(),
            )
            run_command(command)
        elif choice in {"11", "12"}:
            target_family = "cnn" if choice == "11" else "transformer"
            if target_family not in family_ids:
                print(f"Family model '{target_family}' tidak tersedia di registry.")
                continue

            use_all_methods = ask_yes_no("Gunakan semua method training?", True)
            if use_all_methods:
                method_id = None
                selected_method_ids = method_ids
            else:
                method_id = ask_choice(
                    "Pilih method",
                    method_ids,
                    default_index=method_ids.index(method_registry.default_method),
                )
                selected_method_ids = [method_id]

            on_existing_split_mode = ask_on_existing_split_mode()
            on_existing_mode = ask_on_existing_mode()
            method_overrides = ask_method_runtime_overrides(selected_method_ids)
            runtime_toggles = ask_runtime_toggles()

            command = build_train_command(
                dataset_id=selected_dataset,
                models="all",
                method_id=method_id,
                all_methods=use_all_methods,
                check_first=True,
                split_first=True,
                augment_info=True,
                preprocessing_mode=ask_preprocessing_mode(),
                on_existing=on_existing_mode,
                on_existing_split=on_existing_split_mode,
                method_overrides=method_overrides,
                model_families=target_family,
                mixed_precision=runtime_toggles["mixed_precision"],
                disable_cpu_fallback=runtime_toggles["disable_cpu_fallback"],
                shutdown_on_finish=ask_shutdown_on_finish(),
            )
            run_command(command)
        else:
            print("Pilihan tidak dikenali. Coba lagi.")


if __name__ == "__main__":
    main()
