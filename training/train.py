from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.registry import DatasetConfig, DatasetRegistry
from src.datasets.splitter import split_dataset, validate_balanced_split_manifest
from src.datasets.transforms import print_augmentation_summary
from src.datasets.validator import discover_class_directories, list_image_files
from src.models.registry import ModelConfig, ModelRegistry
from src.reporting.report_reader import build_experiment_index
from src.training.augmentations import TrainingAugmentation, TrainingAugmentationRegistry
from src.training.strategies import TrainingMethod, TrainingMethodRegistry, build_training_params
from src.training.trainer import run_model_training
from src.utils.paths import REPORT_ROOT, TRAINED_MODELS_ROOT


ON_EXISTING_CHOICES = ("ask", "retrain", "skip")
ON_EXISTING_SPLIT_CHOICES = ("ask", "resplit", "skip")


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value).strip())
    sanitized = sanitized.strip("-")
    return sanitized or "unknown"


def _build_experiment_id(
    dataset_id: str,
    augmentation_id: str,
    method_id: str,
    model_id: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return "exp_{}_{}_{}_{}_{}".format(
        _sanitize_token(dataset_id),
        _sanitize_token(augmentation_id),
        _sanitize_token(method_id),
        _sanitize_token(model_id),
        timestamp,
    )


def _parse_models_arg(models_arg: str, registry: ModelRegistry) -> List[str]:
    value = str(models_arg).strip().lower()
    if value in {"all", "*"}:
        return registry.list_model_ids(enabled_only=True)

    requested = [item.strip() for item in models_arg.split(",") if item.strip()]
    if not requested:
        raise ValueError("Argumen --models kosong")

    available = set(registry.list_model_ids(enabled_only=False))
    unknown = [m for m in requested if m not in available]
    if unknown:
        raise ValueError("Model tidak ditemukan: {}".format(", ".join(unknown)))
    return requested


def _resolve_dataset_ids(dataset_arg: Optional[str], registry: DatasetRegistry) -> List[str]:
    if dataset_arg is None or not str(dataset_arg).strip():
        return [registry.default_dataset]

    normalized = str(dataset_arg).strip().lower()
    if normalized in {"all", "*"}:
        return registry.list_dataset_ids()

    requested = [item.strip() for item in str(dataset_arg).split(",") if item.strip()]
    if not requested:
        raise ValueError("Argumen --dataset kosong")

    available = set(registry.list_dataset_ids())
    unknown = [dataset_id for dataset_id in requested if dataset_id not in available]
    if unknown:
        raise ValueError("Dataset tidak ditemukan: {}".format(", ".join(unknown)))
    return requested


def _resolve_method_ids(args: argparse.Namespace, registry: TrainingMethodRegistry) -> List[str]:
    if args.all_methods:
        return registry.list_method_ids()

    if args.method is None or not str(args.method).strip():
        return [registry.default_method]

    normalized = str(args.method).strip().lower()
    if normalized in {"all", "*"}:
        return registry.list_method_ids()

    requested = [item.strip() for item in str(args.method).split(",") if item.strip()]
    if not requested:
        raise ValueError("Argumen --method kosong")

    available = set(registry.list_method_ids())
    unknown = [method_id for method_id in requested if method_id not in available]
    if unknown:
        raise ValueError("Method tidak ditemukan: {}".format(", ".join(unknown)))
    return requested


def _resolve_augmentation_ids(
    args: argparse.Namespace,
    augmentation_registry: TrainingAugmentationRegistry,
) -> List[str]:
    if args.augmentations is not None and str(args.augmentations).strip():
        return augmentation_registry.resolve_ids(args.augmentations)

    return augmentation_registry.resolve_from_preprocessing_mode(args.preprocessing_mode)


def _print_dataset_preview(dataset_id: str, dataset_dir: Path, class_mode: str, extensions: List[str]) -> None:
    entries = discover_class_directories(
        dataset_root=dataset_dir,
        class_mode=class_mode,
        extensions=extensions,
    )
    total = sum(entry.image_count for entry in entries)
    print("\n=== Dataset Check ===")
    print("Dataset ID   :", dataset_id)
    print("Original dir :", dataset_dir)
    print("Class mode   :", class_mode)
    print("Jumlah kelas :", len(entries))
    print("Total gambar :", total)


def _run_augment_info(split_dir: Path) -> int:
    try:
        print_augmentation_summary(split_dir)
        return 0
    except Exception as exc:
        print("Gagal menampilkan augmentasi:", exc)
        return 1


def _split_manifest_path(split_dir: Path) -> Path:
    return split_dir / "_metadata" / "split_manifest.json"


def _load_split_manifest(split_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = _split_manifest_path(split_dir)
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return payload
        return None
    except Exception:
        return None


def _split_dir_has_required_structure(split_dir: Path, extensions: List[str]) -> bool:
    required_splits = ["train", "testing", "validation"]
    for split_name in required_splits:
        split_path = split_dir / split_name
        if not split_path.exists() or not split_path.is_dir():
            return False

        class_dirs = [path for path in split_path.iterdir() if path.is_dir()]
        if not class_dirs:
            return False

        has_images = any(list_image_files(class_dir, extensions) for class_dir in class_dirs)
        if not has_images:
            return False
    return True


def _ask_resplit_for_dataset(dataset_id: str, split_dir: Path) -> bool:
    prompt = (
        f"\nFolder split untuk dataset '{dataset_id}' sudah ada di:\n"
        f"{split_dir}\n"
        "Perlu dilakukan split ulang? [y/N]: "
    )
    answer = input(prompt).strip().lower()
    return answer in {"y", "yes"}


def _should_run_split(dataset_cfg: DatasetConfig, on_existing_split: str) -> bool:
    split_dir = dataset_cfg.split_path
    if not split_dir.exists():
        return True

    if on_existing_split == "resplit":
        print("[INFO] `--on-existing-split resplit` aktif. Split akan dijalankan ulang.")
        return True

    if on_existing_split == "skip":
        print("[INFO] `--on-existing-split skip` aktif. Split lama akan digunakan.")
        return False

    if not sys.stdin.isatty():
        print(
            "[INFO] Split folder sudah ada, tetapi sesi non-interaktif tidak bisa bertanya. "
            "Split lama akan digunakan. Gunakan `--on-existing-split resplit` untuk memaksa split ulang."
        )
        return False

    return _ask_resplit_for_dataset(dataset_cfg.dataset_id, split_dir)


def _assert_split_balance(split_manifest: Dict[str, Any], source_label: str) -> None:
    validation = split_manifest.get("balance_validation")
    if not isinstance(validation, dict):
        validation = validate_balanced_split_manifest(split_manifest)

    if not validation.get("is_balanced", False):
        raise RuntimeError(
            "Split {} tidak seimbang antar kelas. "
            "Hentikan training dan periksa dataset/split manifest.".format(source_label)
        )

    print(
        "[VALIDASI] Split {} seimbang: after_counts/train/testing/validation balanced.".format(source_label)
    )


def _collect_user_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    overridable_keys = [
        "image_size",
        "batch_size",
        "epochs",
        "fine_tune_epochs",
        "fine_tune_freeze_ratio",
        "learning_rate",
        "fine_tune_learning_rate",
        "dropout",
        "early_stopping_patience",
        "seed",
        "max_per_class",
        "max_train_per_class",
        "max_validation_per_class",
        "max_test_per_class",
        "train_batch_limit",
        "validation_batch_limit",
        "test_batch_limit",
        "shuffle_buffer_size",
        "num_parallel_calls",
        "prefetch_buffer",
        "gpu_memory_limit_mb",
        "max_cpu_usage_percent",
        "cpu_thread_limit",
        "yolo_size",
    ]
    for key in overridable_keys:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    if args.disable_cpu_fallback:
        overrides["disable_cpu_fallback"] = True
    if args.mixed_precision:
        overrides["mixed_precision"] = True
    if args.no_pretrained:
        overrides["no_pretrained"] = True

    return overrides


def load_dataset(dataset_registry: DatasetRegistry, dataset_id: str) -> DatasetConfig:
    return dataset_registry.get(dataset_id)


def apply_augmentation(
    training_params: Dict[str, Any],
    augmentation_cfg: TrainingAugmentation,
) -> Dict[str, Any]:
    scoped_params = dict(training_params)
    scoped_params["disable_augmentation"] = bool(augmentation_cfg.disable_augmentation)
    return scoped_params


def select_training_method(
    method_registry: TrainingMethodRegistry,
    method_id: str,
    user_overrides: Dict[str, Any],
) -> Tuple[TrainingMethod, Dict[str, Any]]:
    method_cfg = method_registry.get(method_id)
    training_params = build_training_params(method=method_cfg, user_overrides=user_overrides)
    return method_cfg, training_params


def train_model(
    model_cfg: ModelConfig,
    dataset_cfg: DatasetConfig,
    method_id: str,
    augmentation_cfg: TrainingAugmentation,
    training_params: Dict[str, Any],
    experiment_id: str,
) -> int:
    return run_model_training(
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        method_id=method_id,
        training_params=training_params,
        report_root=REPORT_ROOT,
        models_root=TRAINED_MODELS_ROOT,
        augmentation_id=augmentation_cfg.augmentation_id,
        augmentation_label=augmentation_cfg.label,
        experiment_id=experiment_id,
    )


def evaluate_model(
    dataset_id: str,
    augmentation_id: str,
    method_id: str,
    model_id: str,
    expected_experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    records = build_experiment_index(REPORT_ROOT)
    filtered_records = [
        record
        for record in records
        if record.get("dataset") == dataset_id
        and record.get("augmentation") == augmentation_id
        and record.get("method") == method_id
        and record.get("model") == model_id
    ]
    if not filtered_records:
        return {}

    if expected_experiment_id:
        for record in filtered_records:
            if str(record.get("experiment_id", "")).strip() == str(expected_experiment_id).strip():
                return record

    return filtered_records[0]


def _find_existing_runs(
    report_index: List[Dict[str, Any]],
    dataset_id: str,
    augmentation_id: str,
    method_id: str,
    model_id: str,
) -> List[Dict[str, Any]]:
    return [
        record
        for record in report_index
        if record.get("dataset") == dataset_id
        and record.get("augmentation") == augmentation_id
        and record.get("method") == method_id
        and record.get("model") == model_id
    ]


def _ask_retrain_for_existing_combo(
    dataset_id: str,
    augmentation_id: str,
    method_id: str,
    model_id: str,
    latest_run_id: str,
) -> bool:
    prompt = (
        "\nKombinasi sudah pernah ditraining: "
        f"dataset={dataset_id}, augmentasi={augmentation_id}, method={method_id}, model={model_id}\n"
        f"Run terbaru: {latest_run_id}\n"
        "Training ulang akan membuat run/model baru dengan tanggal training baru.\n"
        "Lanjut training ulang? [y/N]: "
    )
    answer = input(prompt).strip().lower()
    return answer in {"y", "yes"}


def _should_run_combo(
    on_existing: str,
    existing_runs: List[Dict[str, Any]],
    dataset_id: str,
    augmentation_id: str,
    method_id: str,
    model_id: str,
) -> bool:
    if not existing_runs:
        return True

    latest_run_id = str(existing_runs[0].get("run_id", "unknown_run"))
    if on_existing == "retrain":
        print(
            "[INFO] Ditemukan run sebelumnya untuk kombinasi ini. "
            "Lanjut retrain dan buat run baru:",
            latest_run_id,
        )
        return True

    if on_existing == "skip":
        print(
            "[SKIP] Kombinasi sudah pernah ditraining dan `--on-existing skip` aktif. "
            "Run terbaru:",
            latest_run_id,
        )
        return False

    if not sys.stdin.isatty():
        print(
            "[SKIP] Kombinasi sudah pernah ditraining, tetapi sesi non-interaktif "
            "tidak bisa meminta konfirmasi retrain. "
            "Gunakan `--on-existing retrain` jika ingin memaksa training ulang."
        )
        return False

    return _ask_retrain_for_existing_combo(
        dataset_id=dataset_id,
        augmentation_id=augmentation_id,
        method_id=method_id,
        model_id=model_id,
        latest_run_id=latest_run_id,
    )


def save_result(results: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
    results.append(result)


def generate_report(results: List[Dict[str, Any]]) -> Optional[Path]:
    if not results:
        return None

    output_root = REPORT_ROOT / "_workflow_runs"
    output_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_root / f"workflow_summary_{stamp}.json"
    csv_path = output_root / f"workflow_summary_{stamp}.csv"
    latest_path = output_root / "latest_workflow_summary.csv"

    import json

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    header = [
        "experiment_id",
        "dataset",
        "augmentation",
        "method",
        "model",
        "status",
        "run_id",
        "train_accuracy",
        "val_accuracy",
        "train_loss",
        "val_loss",
        "test_accuracy",
        "test_f1_score",
        "training_time_seconds",
        "model_path",
        "report_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in header})

    with open(latest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in header})

    return csv_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic training orchestrator untuk project Parkinson.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="ID dataset dari configs/datasets.yaml, daftar dipisah koma, atau 'all'.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="ID training method, daftar dipisah koma, atau 'all'.",
    )
    parser.add_argument("--all-methods", action="store_true", help="Jalankan semua method yang terdaftar")
    parser.add_argument("--models", type=str, default="all", help="List model dipisah koma, atau 'all'")

    parser.add_argument(
        "--augmentations",
        type=str,
        default=None,
        help="ID augmentasi dipisah koma, atau 'all'.",
    )
    parser.add_argument(
        "--preprocessing-mode",
        type=str,
        default="augment",
        choices=["augment", "no_augment", "both"],
        help="Kompatibilitas lama: augment, no_augment, atau both.",
    )

    parser.add_argument("--check-first", action="store_true", help="Jalankan validasi dataset sebelum training")
    parser.add_argument("--split-first", action="store_true", help="Jalankan proses split dataset sebelum training")
    parser.add_argument("--augment-info", action="store_true", help="Cetak info augmentasi train on-the-fly")

    parser.add_argument("--stop-on-error", action="store_true", help="Hentikan jika ada model gagal")
    parser.add_argument(
        "--on-existing",
        type=str,
        default="ask",
        choices=list(ON_EXISTING_CHOICES),
        help="Perilaku jika kombinasi dataset+augmentasi+method+model sudah pernah ditraining: ask, retrain, atau skip.",
    )
    parser.add_argument(
        "--on-existing-split",
        type=str,
        default="ask",
        choices=list(ON_EXISTING_SPLIT_CHOICES),
        help="Perilaku jika folder split dataset sudah ada: ask, resplit, atau skip.",
    )

    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--fine-tune-epochs", type=int, default=None)
    parser.add_argument("--fine-tune-freeze-ratio", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-validation-per-class", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=None)

    parser.add_argument("--train-batch-limit", type=int, default=None)
    parser.add_argument("--validation-batch-limit", type=int, default=None)
    parser.add_argument("--test-batch-limit", type=int, default=None)

    parser.add_argument("--shuffle-buffer-size", type=int, default=None)
    parser.add_argument("--num-parallel-calls", type=int, default=None)
    parser.add_argument("--prefetch-buffer", type=int, default=None)

    parser.add_argument("--gpu-memory-limit-mb", type=int, default=None)
    parser.add_argument("--max-cpu-usage-percent", type=int, default=None)
    parser.add_argument("--cpu-thread-limit", type=int, default=None)

    parser.add_argument("--disable-cpu-fallback", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--yolo-size", type=str, default=None, choices=["n", "s", "m", "l", "x"])

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_registry = DatasetRegistry()
    method_registry = TrainingMethodRegistry()
    model_registry = ModelRegistry()
    augmentation_registry = TrainingAugmentationRegistry()

    dataset_ids = _resolve_dataset_ids(args.dataset, dataset_registry)
    method_ids = _resolve_method_ids(args, method_registry)
    model_ids = _parse_models_arg(args.models, model_registry)
    selected_augmentation_ids = _resolve_augmentation_ids(args, augmentation_registry)

    model_cfgs = [model_registry.get(model_id) for model_id in model_ids if model_registry.get(model_id).enabled]

    user_overrides = _collect_user_overrides(args)

    print("\n=== Rencana Eksekusi Kombinasi ===")
    print("Dataset    :", ", ".join(dataset_ids))
    print("Augmentasi :", ", ".join(selected_augmentation_ids))
    print("Method     :", ", ".join(method_ids))
    print("Model      :", ", ".join([m.model_id for m in model_cfgs]))

    overall_failed: List[str] = []
    workflow_results: List[Dict[str, Any]] = []
    report_index = build_experiment_index(REPORT_ROOT)
    if report_index:
        print("\n[INFO] Ditemukan {} run historis pada folder report.".format(len(report_index)))
    else:
        print("\n[INFO] Belum ada run historis pada folder report.")

    for dataset_id in dataset_ids:
        dataset_cfg = load_dataset(dataset_registry, dataset_id)
        dataset_augmentations = augmentation_registry.resolve_for_dataset(dataset_cfg, selected_augmentation_ids)
        if not dataset_augmentations:
            print("\n[SKIP] Dataset '{}' tidak memiliki augmentasi sesuai pilihan user.".format(dataset_cfg.dataset_id))
            continue

        print("\n" + "=" * 72)
        print("Dataset Aktif:", dataset_cfg.dataset_id)
        print("Augmentasi tersedia untuk dataset:", ", ".join([item.augmentation_id for item in dataset_augmentations]))
        print("=" * 72)

        if args.check_first:
            _print_dataset_preview(
                dataset_id=dataset_cfg.dataset_id,
                dataset_dir=dataset_cfg.original_path,
                class_mode=dataset_cfg.class_mode,
                extensions=dataset_cfg.valid_extensions,
            )

        if args.split_first:
            should_resplit = _should_run_split(dataset_cfg=dataset_cfg, on_existing_split=args.on_existing_split)
            if should_resplit:
                print("\n=== Split Dataset ===")
                split_manifest = split_dataset(
                    original_dir=dataset_cfg.original_path,
                    split_dir=dataset_cfg.split_path,
                    class_mode=dataset_cfg.class_mode,
                    extensions=dataset_cfg.valid_extensions,
                    split_cfg=dataset_cfg.split,
                    resize_cfg=dataset_cfg.resize,
                    seed=int(user_overrides.get("seed", dataset_cfg.seed)),
                )
                source_label = "baru"
            else:
                print("\n=== Gunakan Split Existing ===")
                if not _split_dir_has_required_structure(dataset_cfg.split_path, dataset_cfg.valid_extensions):
                    raise RuntimeError(
                        "Folder split existing tidak valid/kurang lengkap. "
                        "Jalankan split ulang (`--on-existing-split resplit`)."
                    )
                split_manifest = _load_split_manifest(dataset_cfg.split_path)
                if split_manifest is None:
                    raise RuntimeError(
                        "split_manifest.json tidak ditemukan pada split existing. "
                        "Jalankan split ulang (`--on-existing-split resplit`)."
                    )
                source_label = "existing"

            print("Split manifest:", dataset_cfg.split_path / "_metadata" / "split_manifest.json")
            split_stats = split_manifest.get("split_stats", {})
            for split_name in ["train", "testing", "validation"]:
                total = sum(int(v) for v in split_stats.get(split_name, {}).values())
                print(f"- {split_name:10s}: {total} gambar")
            balancing_info = split_manifest.get("class_balancing", {})
            if balancing_info.get("applied"):
                print("Balancing  : aktif, total augmentasi =", balancing_info.get("total_generated", 0))
            else:
                print("Balancing  : tidak perlu (kelas sudah seimbang)")
            _assert_split_balance(split_manifest=split_manifest, source_label=source_label)
        else:
            if not _split_dir_has_required_structure(dataset_cfg.split_path, dataset_cfg.valid_extensions):
                raise RuntimeError(
                    "Folder split dataset tidak valid/kurang lengkap. "
                    "Gunakan `--split-first` untuk membuat split yang siap training."
                )
            existing_manifest = _load_split_manifest(dataset_cfg.split_path)
            if existing_manifest is not None:
                _assert_split_balance(split_manifest=existing_manifest, source_label="existing")
            else:
                print(
                    "[INFO] split_manifest.json belum tersedia. "
                    "Validasi balance split dilewati. "
                    "Gunakan `--split-first` agar split tervalidasi otomatis."
                )

        if args.augment_info:
            rc = _run_augment_info(dataset_cfg.split_path)
            if rc != 0:
                raise SystemExit(rc)

        for augmentation_cfg in dataset_augmentations:
            print(
                "\n=== Skenario Augmentasi: {} ({}) ===".format(
                    augmentation_cfg.augmentation_id,
                    augmentation_cfg.label,
                )
            )
            for method_id in method_ids:
                method_cfg, base_training_params = select_training_method(
                    method_registry=method_registry,
                    method_id=method_id,
                    user_overrides=user_overrides,
                )
                training_params = apply_augmentation(base_training_params, augmentation_cfg)

                print("\n--- Method: {} ---".format(method_id))
                print("Deskripsi:", method_cfg.description)

                for model_cfg in model_cfgs:
                    existing_runs = _find_existing_runs(
                        report_index=report_index,
                        dataset_id=dataset_cfg.dataset_id,
                        augmentation_id=augmentation_cfg.augmentation_id,
                        method_id=method_id,
                        model_id=model_cfg.model_id,
                    )
                    should_run = _should_run_combo(
                        on_existing=args.on_existing,
                        existing_runs=existing_runs,
                        dataset_id=dataset_cfg.dataset_id,
                        augmentation_id=augmentation_cfg.augmentation_id,
                        method_id=method_id,
                        model_id=model_cfg.model_id,
                    )
                    if not should_run:
                        latest_existing = existing_runs[0] if existing_runs else {}
                        save_result(
                            workflow_results,
                            {
                                "experiment_id": latest_existing.get("experiment_id"),
                                "dataset": dataset_cfg.dataset_id,
                                "augmentation": augmentation_cfg.augmentation_id,
                                "method": method_id,
                                "model": model_cfg.model_id,
                                "status": "skipped_existing",
                                "run_id": latest_existing.get("run_id"),
                                "train_accuracy": latest_existing.get("train_accuracy"),
                                "val_accuracy": latest_existing.get("val_accuracy"),
                                "train_loss": latest_existing.get("train_loss"),
                                "val_loss": latest_existing.get("val_loss"),
                                "test_accuracy": latest_existing.get("accuracy"),
                                "test_f1_score": latest_existing.get("f1_score"),
                                "training_time_seconds": latest_existing.get("training_time_seconds"),
                                "model_path": latest_existing.get("final_model_path"),
                                "report_path": latest_existing.get("run_dir"),
                            },
                        )
                        continue

                    experiment_id = _build_experiment_id(
                        dataset_id=dataset_cfg.dataset_id,
                        augmentation_id=augmentation_cfg.augmentation_id,
                        method_id=method_id,
                        model_id=model_cfg.model_id,
                    )
                    print(
                        "\n[RUN] dataset={} | augmentasi={} | method={} | model={} | experiment_id={}".format(
                            dataset_cfg.dataset_id,
                            augmentation_cfg.augmentation_id,
                            method_id,
                            model_cfg.model_id,
                            experiment_id,
                        )
                    )
                    rc = train_model(
                        model_cfg=model_cfg,
                        dataset_cfg=dataset_cfg,
                        method_id=method_id,
                        augmentation_cfg=augmentation_cfg,
                        training_params=training_params,
                        experiment_id=experiment_id,
                    )

                    eval_result = evaluate_model(
                        dataset_id=dataset_cfg.dataset_id,
                        augmentation_id=augmentation_cfg.augmentation_id,
                        method_id=method_id,
                        model_id=model_cfg.model_id,
                        expected_experiment_id=experiment_id,
                    )
                    metrics = eval_result.get("metrics", {}) if isinstance(eval_result, dict) else {}

                    result_row = {
                        "experiment_id": experiment_id,
                        "dataset": dataset_cfg.dataset_id,
                        "augmentation": augmentation_cfg.augmentation_id,
                        "method": method_id,
                        "model": model_cfg.model_id,
                        "status": "success" if rc == 0 else "failed",
                        "run_id": eval_result.get("run_id"),
                        "train_accuracy": metrics.get("train_accuracy"),
                        "val_accuracy": metrics.get("val_accuracy"),
                        "train_loss": metrics.get("train_loss"),
                        "val_loss": metrics.get("val_loss"),
                        "test_accuracy": metrics.get("accuracy"),
                        "test_f1_score": metrics.get("f1_score"),
                        "training_time_seconds": eval_result.get("duration_seconds") or metrics.get("training_time_seconds"),
                        "model_path": eval_result.get("final_model_path"),
                        "report_path": eval_result.get("run_dir"),
                    }
                    save_result(workflow_results, result_row)
                    report_index = build_experiment_index(REPORT_ROOT)

                    if rc != 0:
                        failed_id = "{}:{}:{}:{}".format(
                            dataset_cfg.dataset_id,
                            augmentation_cfg.augmentation_id,
                            method_id,
                            model_cfg.model_id,
                        )
                        overall_failed.append(failed_id)
                        print("[FAILED]", failed_id)
                        if args.stop_on_error:
                            break

                if args.stop_on_error and overall_failed:
                    break

            if args.stop_on_error and overall_failed:
                break

        if args.stop_on_error and overall_failed:
            break

    summary_path = generate_report(workflow_results)
    if summary_path is not None:
        print("\nRingkasan workflow tersimpan di:", summary_path)

    if overall_failed:
        print("\nTraining selesai dengan kegagalan:")
        for item in overall_failed:
            print("-", item)
        raise SystemExit(1)

    print("\nTraining selesai tanpa error.")


if __name__ == "__main__":
    main()
