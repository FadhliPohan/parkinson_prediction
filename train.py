from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from src.datasets.registry import DatasetRegistry
from src.datasets.splitter import split_dataset
from src.datasets.transforms import print_augmentation_summary
from src.datasets.validator import discover_class_directories
from src.models.registry import ModelRegistry
from src.training.strategies import TrainingMethodRegistry, build_training_params
from src.training.trainer import pick_failed_models, run_training_jobs
from src.utils.paths import REPORT_ROOT, TRAINED_MODELS_ROOT


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic training orchestrator untuk project Parkinson.")
    parser.add_argument("--dataset", type=str, default=None, help="ID dataset dari configs/datasets.yaml")
    parser.add_argument("--method", type=str, default=None, help="ID training method")
    parser.add_argument("--all-methods", action="store_true", help="Jalankan semua method yang terdaftar")
    parser.add_argument("--models", type=str, default="all", help="List model dipisah koma, atau 'all'")

    parser.add_argument("--check-first", action="store_true", help="Jalankan validasi dataset sebelum training")
    parser.add_argument("--split-first", action="store_true", help="Jalankan proses split dataset sebelum training")
    parser.add_argument("--augment-info", action="store_true", help="Cetak info augmentasi train on-the-fly")

    parser.add_argument("--stop-on-error", action="store_true", help="Hentikan jika ada model gagal")

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

    dataset_id = args.dataset or dataset_registry.default_dataset
    dataset_cfg = dataset_registry.get(dataset_id)

    model_ids = _parse_models_arg(args.models, model_registry)
    model_cfgs = [model_registry.get(model_id) for model_id in model_ids if model_registry.get(model_id).enabled]

    method_ids = method_registry.list_method_ids() if args.all_methods else [args.method or method_registry.default_method]

    user_overrides = _collect_user_overrides(args)

    if args.check_first:
        _print_dataset_preview(
            dataset_id=dataset_cfg.dataset_id,
            dataset_dir=dataset_cfg.original_path,
            class_mode=dataset_cfg.class_mode,
            extensions=dataset_cfg.valid_extensions,
        )

    if args.split_first:
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
        print("Split selesai. Manifest:", dataset_cfg.split_path / "_metadata" / "split_manifest.json")
        split_stats = split_manifest.get("split_stats", {})
        for split_name in ["train", "testing", "validation"]:
            total = sum(int(v) for v in split_stats.get(split_name, {}).values())
            print(f"- {split_name:10s}: {total} gambar")

    if args.augment_info:
        rc = _run_augment_info(dataset_cfg.split_path)
        if rc != 0:
            raise SystemExit(rc)

    print("\n=== Eksekusi Training ===")
    print("Dataset :", dataset_cfg.dataset_id)
    print("Models  :", ", ".join([m.model_id for m in model_cfgs]))
    print("Methods :", ", ".join(method_ids))

    overall_failed: List[str] = []

    for method_id in method_ids:
        method_cfg = method_registry.get(method_id)
        training_params = build_training_params(method=method_cfg, user_overrides=user_overrides)

        print("\n--- Method:", method_id, "---")
        status_map = run_training_jobs(
            models=model_cfgs,
            dataset_cfg=dataset_cfg,
            method_id=method_id,
            training_params=training_params,
            report_root=REPORT_ROOT,
            models_root=TRAINED_MODELS_ROOT,
            stop_on_error=args.stop_on_error,
        )

        failed = pick_failed_models(status_map)
        if failed:
            overall_failed.extend([f"{method_id}:{model_id}" for model_id in failed])
            print("Model gagal:", ", ".join(failed))
            if args.stop_on_error:
                break
        else:
            print("Semua model sukses untuk method", method_id)

    if overall_failed:
        print("\nTraining selesai dengan kegagalan:")
        for item in overall_failed:
            print("-", item)
        raise SystemExit(1)

    print("\nTraining selesai tanpa error.")


if __name__ == "__main__":
    main()
