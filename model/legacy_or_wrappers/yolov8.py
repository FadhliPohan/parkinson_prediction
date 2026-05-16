import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.report_writer import (
    build_artifact_dirs,
    generate_run_id,
    write_latest_run_marker,
    write_run_manifest,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cpu_runtime(
    max_cpu_usage_percent: int,
    cpu_thread_limit: Optional[int] = None,
) -> int:
    total_logical_cpu = os.cpu_count() or 1
    safe_percent = min(100, max(10, int(max_cpu_usage_percent)))
    auto_thread_limit = max(1, int(total_logical_cpu * (safe_percent / 100.0)))
    thread_limit = auto_thread_limit
    if cpu_thread_limit is not None and cpu_thread_limit > 0:
        thread_limit = max(1, min(total_logical_cpu, int(cpu_thread_limit)))

    try:
        torch.set_num_threads(thread_limit)
    except Exception as exc:
        print("Peringatan: gagal mengatur jumlah thread CPU -> {}".format(exc))

    try:
        interop_threads = max(1, min(4, thread_limit // 2 if thread_limit > 1 else 1))
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        # Aman diabaikan jika runtime tidak mengizinkan perubahan thread inter-op.
        pass

    print("\n=== Kontrol CPU ===")
    print("Logical CPU terdeteksi    :", total_logical_cpu)
    print("Target maksimal CPU (%)   :", safe_percent)
    print("Batas thread CPU aktif    :", thread_limit)
    return thread_limit


def configure_compute_device(disable_cpu_fallback: bool) -> str:
    if torch.cuda.is_available():
        print("\nGPU terdeteksi. Training akan menggunakan GPU:")
        total_gpu = torch.cuda.device_count()
        for idx in range(total_gpu):
            print("- cuda:{} ({})".format(idx, torch.cuda.get_device_name(idx)))
        return "0"

    if disable_cpu_fallback:
        print("\n[TRAINING DIBATALKAN] GPU tidak terdeteksi.")
        print("Aktifkan CPU fallback atau pastikan CUDA tersedia.")
        raise SystemExit(1)

    print("\n[CPU Fallback] GPU tidak terdeteksi.")
    print("Training dilanjutkan di CPU.")
    return "cpu"


def is_gpu_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    memory_keywords = [
        "out of memory",
        "cuda out of memory",
        "cudnn_status_alloc_failed",
        "cuda_error_out_of_memory",
        "resource exhausted",
    ]
    return any(keyword in message for keyword in memory_keywords)


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


def ensure_split_structure(dataset_dir: Path) -> None:
    required_splits = ["train", "testing", "validation"]
    missing: List[str] = []
    for split_name in required_splits:
        split_path = dataset_dir / split_name
        if not split_path.exists() or not split_path.is_dir():
            missing.append(str(split_path))
    if missing:
        raise FileNotFoundError("Folder split tidak lengkap: {}".format(", ".join(missing)))


def get_class_names(dataset_dir: Path) -> List[str]:
    train_dir = dataset_dir / "train"
    class_names = sorted([path.name for path in train_dir.iterdir() if path.is_dir()])
    if len(class_names) < 2:
        raise ValueError("Minimal harus ada 2 kelas. Saat ini ditemukan: {}".format(class_names))
    return class_names


def pick_split_limit(default_limit: Optional[int], split_limit: Optional[int]) -> Optional[int]:
    if split_limit is not None and split_limit > 0:
        return split_limit
    return default_limit


def collect_split_files(
    split_dir: Path,
    class_names: List[str],
    max_per_class: Optional[int],
    seed: int,
) -> Dict[str, List[Path]]:
    rng = random.Random(seed)
    split_files: Dict[str, List[Path]] = {}

    for class_name in class_names:
        class_dir = split_dir / class_name
        if not class_dir.exists() or not class_dir.is_dir():
            raise FileNotFoundError("Folder kelas tidak ditemukan: {}".format(class_dir))
        files = list_image_files(class_dir)
        if max_per_class is not None and max_per_class > 0:
            rng.shuffle(files)
            files = files[:max_per_class]
            files = sorted(files)
        split_files[class_name] = files

    return split_files


def apply_sample_cap(
    split_files: Dict[str, List[Path]],
    sample_cap: Optional[int],
    seed: int,
) -> Dict[str, List[Path]]:
    if sample_cap is None or sample_cap <= 0:
        return split_files

    all_items: List[Tuple[str, Path]] = []
    for class_name, files in split_files.items():
        for file_path in files:
            all_items.append((class_name, file_path))

    if len(all_items) <= sample_cap:
        return split_files

    rng = random.Random(seed)
    rng.shuffle(all_items)
    selected_items = sorted(all_items[:sample_cap], key=lambda x: (x[0], str(x[1])))

    capped: Dict[str, List[Path]] = dict((class_name, []) for class_name in split_files.keys())
    for class_name, file_path in selected_items:
        capped[class_name].append(file_path)
    return capped


def safe_link_or_copy(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(str(src), str(dst))


def materialize_dataset_view(
    dataset_dir: Path,
    class_names: List[str],
    train_files: Dict[str, List[Path]],
    val_files: Dict[str, List[Path]],
    test_files: Dict[str, List[Path]],
    target_root: Path,
) -> Dict[str, Dict[str, int]]:
    if target_root.exists():
        shutil.rmtree(str(target_root))
    target_root.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }
    split_counts: Dict[str, Dict[str, int]] = {}

    for split_name, file_map in split_map.items():
        split_counts[split_name] = {}
        for class_name in class_names:
            class_target_dir = target_root / split_name / class_name
            class_target_dir.mkdir(parents=True, exist_ok=True)

            files = file_map.get(class_name, [])
            split_counts[split_name][class_name] = len(files)
            for idx, src_path in enumerate(files):
                dst_name = "{:06d}_{}".format(idx, src_path.name)
                safe_link_or_copy(src=src_path, dst=class_target_dir / dst_name)

    print("\n=== Ringkasan Dataset untuk YOLOv8 ===")
    print("Sumber dataset split :", dataset_dir)
    print("Dataset view YOLO    :", target_root)
    for split_name in ["train", "val", "test"]:
        total_count = sum(split_counts[split_name].values())
        print("- {}: {} sampel".format(split_name, total_count))
    return split_counts


def plot_training_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    if history_df.empty:
        return

    history_df = history_df.copy()
    history_df.columns = [str(col).strip() for col in history_df.columns]
    epoch_values = history_df["epoch"] if "epoch" in history_df.columns else np.arange(len(history_df))

    loss_candidates = ["train/loss", "train/cls_loss", "loss"]
    val_loss_candidates = ["val/loss", "val/cls_loss", "val_loss"]
    acc_candidates = ["metrics/accuracy_top1", "accuracy", "train/accuracy"]
    val_acc_candidates = ["val/accuracy_top1", "val_accuracy", "metrics/accuracy_top1"]

    def first_existing(candidates: List[str]) -> Optional[str]:
        for col in candidates:
            if col in history_df.columns:
                return col
        return None

    loss_col = first_existing(loss_candidates)
    val_loss_col = first_existing(val_loss_candidates)
    acc_col = first_existing(acc_candidates)
    val_acc_col = first_existing(val_acc_candidates)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if loss_col:
        plt.plot(epoch_values, history_df[loss_col], label="train_loss")
    if val_loss_col:
        plt.plot(epoch_values, history_df[val_loss_col], label="val_loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if loss_col or val_loss_col:
        plt.legend()
    plt.grid(alpha=0.3, linestyle="--")

    plt.subplot(1, 2, 2)
    if acc_col:
        plt.plot(epoch_values, history_df[acc_col], label="train_accuracy")
    if val_acc_col:
        plt.plot(epoch_values, history_df[val_acc_col], label="val_accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    if acc_col or val_acc_col:
        plt.legend()
    plt.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def plot_split_distribution(split_counts: Dict[str, Dict[str, int]], output_path: Path) -> None:
    dataframe = pd.DataFrame(split_counts).T
    dataframe = dataframe.sort_index()
    ax = dataframe.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Distribusi Data per Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Jumlah Gambar")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def plot_metrics_table(metrics: Dict[str, float], output_path: Path) -> None:
    table_df = pd.DataFrame(
        [{"Metric": key, "Value": value} for key, value in metrics.items()]
    )
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.6 * len(table_df))))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    plt.title("Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def plot_roc_curve_binary(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC (AUC={:.4f})".format(auc_value))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    return float(auc_value)


def plot_roc_curve_multiclass(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")

    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    auc_value = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")

    plt.figure(figsize=(7, 6))
    for class_index, class_name in enumerate(class_names):
        if len(np.unique(y_true_bin[:, class_index])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], y_prob[:, class_index])
        class_auc = roc_auc_score(y_true_bin[:, class_index], y_prob[:, class_index])
        plt.plot(fpr, tpr, label="{} (AUC={:.4f})".format(class_name, class_auc))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (OvR)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    return float(auc_value)


def get_yolo_model_source(model_size: str, no_pretrained: bool) -> str:
    safe_size = model_size.lower().strip()
    if safe_size not in {"n", "s", "m", "l", "x"}:
        raise ValueError("Ukuran YOLOv8 tidak valid: {}".format(model_size))
    suffix = "yaml" if no_pretrained else "pt"
    return "yolov8{}-cls.{}".format(safe_size, suffix)


def train_yolov8_once(
    data_dir: Path,
    run_model_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> Path:
    model_source = get_yolo_model_source(model_size=args.yolo_size, no_pretrained=args.no_pretrained)
    model = YOLO(model_source)

    total_epochs = max(1, int(args.epochs) + max(0, int(args.fine_tune_epochs)))
    mixed_precision_enabled = bool(args.mixed_precision and device != "cpu")
    use_train_augmentation = not bool(getattr(args, "disable_augmentation", False))

    print("\n=== Training YOLOv8 ===")
    print("Model source             :", model_source)
    print("Total epoch              :", total_epochs)
    print("Batch size               :", args.batch_size)
    print("Image size               :", args.image_size)
    print("Learning rate            :", args.learning_rate)
    print("Mixed precision          :", mixed_precision_enabled)
    print("Augmentasi train         :", "aktif" if use_train_augmentation else "nonaktif")
    print("Device aktif             :", "GPU" if device != "cpu" else "CPU")
    if args.gpu_memory_limit_mb:
        print("Catatan: --gpu-memory-limit-mb belum didukung langsung oleh YOLOv8/PyTorch.")
    if args.prefetch_buffer:
        print("Catatan: --prefetch-buffer dikelola internal oleh DataLoader YOLOv8.")
    if args.shuffle_buffer_size:
        print("Catatan: --shuffle-buffer-size dikelola internal oleh DataLoader YOLOv8.")

    train_result = model.train(
        data=str(data_dir),
        epochs=total_epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        lr0=args.learning_rate,
        dropout=args.dropout,
        patience=args.early_stopping_patience,
        workers=max(1, int(args.num_parallel_calls)),
        seed=args.seed,
        project=str(run_model_dir),
        name="ultralytics",
        exist_ok=True,
        device=device,
        pretrained=not args.no_pretrained,
        amp=mixed_precision_enabled,
        augment=use_train_augmentation,
        verbose=True,
    )
    return Path(train_result.save_dir).resolve()


def copy_weights(train_save_dir: Path, run_model_dir: Path) -> Tuple[Path, Path]:
    weights_dir = train_save_dir / "weights"
    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"

    if not best_src.exists() and not last_src.exists():
        raise FileNotFoundError("Weights YOLOv8 tidak ditemukan di {}".format(weights_dir))

    best_dst = run_model_dir / "best_model.pt"
    final_dst = run_model_dir / "final_model.pt"

    if best_src.exists():
        shutil.copy2(str(best_src), str(best_dst))
    elif last_src.exists():
        shutil.copy2(str(last_src), str(best_dst))

    if last_src.exists():
        shutil.copy2(str(last_src), str(final_dst))
    else:
        shutil.copy2(str(best_dst), str(final_dst))

    return best_dst, final_dst


def collect_eval_targets(test_dir: Path, class_names: List[str]) -> Tuple[List[Path], np.ndarray]:
    paths: List[Path] = []
    labels: List[int] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = test_dir / class_name
        for image_path in list_image_files(class_dir):
            paths.append(image_path)
            labels.append(class_index)
    return paths, np.array(labels, dtype=np.int32)


def predict_probabilities(
    model_path: Path,
    image_paths: List[Path],
    class_count: int,
    image_size: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model = YOLO(str(model_path))
    sources = [str(path) for path in image_paths]
    stream = model.predict(
        source=sources,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        verbose=False,
        stream=True,
    )

    probabilities: List[np.ndarray] = []
    for result in stream:
        if result.probs is None:
            continue
        probs = result.probs.data.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if probs.size < class_count:
            probs = np.pad(probs, (0, class_count - probs.size), mode="constant")
        elif probs.size > class_count:
            probs = probs[:class_count]
        prob_sum = float(np.sum(probs))
        if prob_sum > 0.0:
            probs = probs / prob_sum
        probabilities.append(probs)

    if not probabilities:
        return np.zeros((0, class_count), dtype=np.float32)
    return np.vstack(probabilities)


def load_training_history(train_save_dir: Path) -> pd.DataFrame:
    history_path = train_save_dir / "results.csv"
    if not history_path.exists():
        return pd.DataFrame()
    try:
        history_df = pd.read_csv(str(history_path))
        history_df.columns = [str(col).strip() for col in history_df.columns]
        return history_df
    except Exception:
        return pd.DataFrame()


def run_pipeline(args: argparse.Namespace) -> Dict[str, Path]:
    run_started_at = datetime.now().astimezone().isoformat()
    set_global_seed(args.seed)
    configure_cpu_runtime(
        max_cpu_usage_percent=args.max_cpu_usage_percent,
        cpu_thread_limit=args.cpu_thread_limit,
    )
    compute_device = configure_compute_device(disable_cpu_fallback=args.disable_cpu_fallback)
    final_compute_device = compute_device

    dataset_dir = (PROJECT_ROOT / args.dataset_dir).resolve()
    ensure_split_structure(dataset_dir)

    class_names = get_class_names(dataset_dir)
    num_classes = len(class_names)
    is_binary = num_classes == 2

    report_root = (PROJECT_ROOT / args.report_root).resolve()
    models_root = (PROJECT_ROOT / args.models_root).resolve()
    dataset_name = str(args.dataset_name).strip() if str(args.dataset_name).strip() else "default_dataset"
    run_id = generate_run_id()
    report_model_root, models_model_root, run_report_dir, run_model_dir = build_artifact_dirs(
        report_root=report_root,
        models_root=models_root,
        dataset_name=dataset_name,
        model_name="yolov8",
        run_id=run_id,
    )

    train_limit = pick_split_limit(args.max_per_class, args.max_train_per_class)
    val_limit = pick_split_limit(args.max_per_class, args.max_validation_per_class)
    test_limit = pick_split_limit(args.max_per_class, args.max_test_per_class)

    train_files = collect_split_files(
        split_dir=dataset_dir / "train",
        class_names=class_names,
        max_per_class=train_limit,
        seed=args.seed,
    )
    val_files = collect_split_files(
        split_dir=dataset_dir / "validation",
        class_names=class_names,
        max_per_class=val_limit,
        seed=args.seed + 1,
    )
    test_files = collect_split_files(
        split_dir=dataset_dir / "testing",
        class_names=class_names,
        max_per_class=test_limit,
        seed=args.seed + 2,
    )

    train_sample_cap = (
        int(args.train_batch_limit) * int(args.batch_size)
        if args.train_batch_limit is not None and args.train_batch_limit > 0
        else None
    )
    val_sample_cap = (
        int(args.validation_batch_limit) * int(args.batch_size)
        if args.validation_batch_limit is not None and args.validation_batch_limit > 0
        else None
    )
    test_sample_cap = (
        int(args.test_batch_limit) * int(args.batch_size)
        if args.test_batch_limit is not None and args.test_batch_limit > 0
        else None
    )

    train_files = apply_sample_cap(train_files, train_sample_cap, args.seed + 10)
    val_files = apply_sample_cap(val_files, val_sample_cap, args.seed + 11)
    test_files = apply_sample_cap(test_files, test_sample_cap, args.seed + 12)

    dataset_view_dir = run_model_dir / "_dataset_view"
    split_counts = materialize_dataset_view(
        dataset_dir=dataset_dir,
        class_names=class_names,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        target_root=dataset_view_dir,
    )

    train_total = sum(split_counts["train"].values())
    val_total = sum(split_counts["val"].values())
    test_total = sum(split_counts["test"].values())
    if train_total == 0 or val_total == 0 or test_total == 0:
        raise ValueError("Data train/validation/testing kosong setelah proses sampling.")

    try:
        train_save_dir = train_yolov8_once(
            data_dir=dataset_view_dir,
            run_model_dir=run_model_dir,
            args=args,
            device=compute_device,
        )
    except Exception as exc:
        can_fallback_to_cpu = (
            compute_device != "cpu"
            and not args.disable_cpu_fallback
            and is_gpu_memory_error(exc)
        )
        if not can_fallback_to_cpu:
            raise

        print("\n[PERINGATAN] GPU penuh saat training: {}".format(exc))
        print("Mencoba ulang otomatis menggunakan CPU...")
        torch.cuda.empty_cache()
        fallback_train_dir = run_model_dir / "ultralytics"
        if fallback_train_dir.exists():
            shutil.rmtree(str(fallback_train_dir))
        final_compute_device = "cpu"
        train_save_dir = train_yolov8_once(
            data_dir=dataset_view_dir,
            run_model_dir=run_model_dir,
            args=args,
            device="cpu",
        )

    best_model_path, final_model_path = copy_weights(train_save_dir, run_model_dir)

    history_df = load_training_history(train_save_dir)
    history_df.to_csv(str(run_report_dir / "training_history.csv"), index=False)

    eval_paths, y_true = collect_eval_targets(dataset_view_dir / "test", class_names)
    y_prob = predict_probabilities(
        model_path=best_model_path,
        image_paths=eval_paths,
        class_count=num_classes,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=final_compute_device,
    )

    aligned_count = min(len(y_true), len(y_prob))
    y_true = y_true[:aligned_count]
    y_prob = y_prob[:aligned_count]
    if aligned_count == 0:
        raise ValueError("Prediksi testing gagal: tidak ada output probabilitas.")

    y_pred = np.argmax(y_prob, axis=1).astype(np.int32)
    test_accuracy = accuracy_score(y_true, y_pred)
    average_mode = "binary" if is_binary else "macro"
    test_f1 = f1_score(y_true, y_pred, average=average_mode)

    if is_binary:
        positive_prob = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.reshape(-1)
        try:
            test_roc_auc = plot_roc_curve_binary(
                y_true=y_true,
                y_prob=positive_prob,
                output_path=run_report_dir / "roc_curve.png",
            )
        except Exception:
            test_roc_auc = float("nan")
    else:
        try:
            test_roc_auc = plot_roc_curve_multiclass(
                y_true=y_true,
                y_prob=y_prob,
                class_names=class_names,
                output_path=run_report_dir / "roc_curve.png",
            )
        except Exception:
            test_roc_auc = float("nan")

    label_indices = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)
    cls_report = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics_summary = {
        "device_used": "cpu" if final_compute_device == "cpu" else "gpu",
        "accuracy": float(test_accuracy),
        "f1_score": float(test_f1),
        "roc_auc": float(test_roc_auc) if not np.isnan(test_roc_auc) else float("nan"),
        "test_samples": int(len(y_true)),
        "train_samples_used_per_epoch": int(train_total),
        "validation_samples_used": int(val_total),
        "train_samples_total": int(train_total),
        "validation_samples_total": int(val_total),
        "test_samples_total": int(test_total),
        "epochs_total": int(max(1, int(args.epochs) + max(0, int(args.fine_tune_epochs)))),
        "yolo_variant": "yolov8{}-cls".format(args.yolo_size),
    }

    classification_df = pd.DataFrame(cls_report).transpose()
    classification_df.to_csv(str(run_report_dir / "classification_report.csv"))

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(str(run_report_dir / "confusion_matrix.csv"))

    split_df = pd.DataFrame(
        {
            "train": split_counts["train"],
            "validation": split_counts["val"],
            "testing": split_counts["test"],
        }
    ).transpose()
    split_df.to_csv(str(run_report_dir / "split_distribution.csv"))

    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(str(run_report_dir / "evaluation_metrics.csv"), index=False)

    plot_training_curves(history_df, run_report_dir / "training_curves.png")
    plot_confusion_matrix(cm, class_names, run_report_dir / "confusion_matrix.png")
    plot_split_distribution(
        {
            "train": split_counts["train"],
            "validation": split_counts["val"],
            "testing": split_counts["test"],
        },
        run_report_dir / "split_distribution.png",
    )
    plot_metrics_table(metrics_summary, run_report_dir / "evaluation_table.png")

    with open(str(run_report_dir / "evaluation_metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics_summary, fp, indent=2)

    with open(str(run_model_dir / "class_names.json"), "w", encoding="utf-8") as fp:
        json.dump(class_names, fp, indent=2)

    run_finished_at = datetime.now().astimezone().isoformat()

    run_summary_lines = [
        "Model: yolov8",
        "Dataset Name: {}".format(dataset_name),
        "Run ID: {}".format(run_id),
        "Dataset: {}".format(dataset_dir),
        "Dataset view: {}".format(dataset_view_dir),
        "Report dir: {}".format(run_report_dir),
        "Model dir: {}".format(run_model_dir),
        "Training Method: {}".format(args.training_method),
        "Best model: {}".format(best_model_path),
        "Final model: {}".format(final_model_path),
        "Device: {}".format("CPU" if final_compute_device == "cpu" else "GPU"),
        "Accuracy: {:.4f}".format(metrics_summary["accuracy"]),
        "F1-score: {:.4f}".format(metrics_summary["f1_score"]),
        "ROC-AUC: {}".format(
            "{:.4f}".format(metrics_summary["roc_auc"])
            if not np.isnan(metrics_summary["roc_auc"])
            else "NaN"
        ),
    ]
    with open(str(run_report_dir / "summary.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(run_summary_lines))

    run_manifest = {
        "schema_version": "2.0.0",
        "run_id": run_id,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "dataset": {
            "dataset_name": dataset_name,
            "dataset_dir": str(dataset_dir),
            "class_names": class_names,
            "class_count": int(num_classes),
            "split_counts": {
                "train": split_counts["train"],
                "testing": split_counts["test"],
                "validation": split_counts["val"],
            },
        },
        "model": {
            "model_name": "yolov8",
            "framework": "yolo",
            "variant": "yolov8{}-cls".format(args.yolo_size),
        },
        "training": {
            "method": args.training_method,
            "parameters": vars(args),
            "device_used": "cpu" if final_compute_device == "cpu" else "gpu",
        },
        "artifacts": {
            "report_dir": str(run_report_dir),
            "model_dir": str(run_model_dir),
            "best_model_path": str(best_model_path),
            "final_model_path": str(final_model_path),
        },
    }
    write_run_manifest(run_report_dir=run_report_dir, manifest=run_manifest)

    write_latest_run_marker(report_model_root, run_id)
    write_latest_run_marker(models_model_root, run_id)

    print("\n=== Ringkasan Evaluasi YOLOv8 ===")
    print("Device   : {}".format("CPU" if final_compute_device == "cpu" else "GPU"))
    print("Accuracy : {:.4f}".format(metrics_summary["accuracy"]))
    print("F1-score : {:.4f}".format(metrics_summary["f1_score"]))
    if np.isnan(metrics_summary["roc_auc"]):
        print("ROC-AUC  : NaN")
    else:
        print("ROC-AUC  : {:.4f}".format(metrics_summary["roc_auc"]))
    print("Report   : {}".format(run_report_dir))
    print("Model    : {}".format(run_model_dir))

    return {
        "report_dir": run_report_dir,
        "model_dir": run_model_dir,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training klasifikasi Parkinson dengan YOLOv8.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/split/parkinson_merder",
        help="Folder dataset hasil split.",
    )
    parser.add_argument("--dataset-name", type=str, default="default_dataset", help="ID dataset (untuk path artifact).")
    parser.add_argument(
        "--training-method",
        type=str,
        default="transfer_learning",
        help="ID metode training (untuk metadata run).",
    )
    parser.add_argument(
        "--report-root",
        type=str,
        default="report",
        help="Root folder report.",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="trained_models",
        help="Root folder model artifact.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Ukuran gambar input model.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size saat training.")
    parser.add_argument("--epochs", type=int, default=8, help="Jumlah epoch stage 1.")
    parser.add_argument("--fine-tune-epochs", type=int, default=2, help="Tambahan epoch (digabung ke total epoch).")
    parser.add_argument(
        "--fine-tune-freeze-ratio",
        type=float,
        default=0.7,
        help="Kompatibilitas argumen dengan model lain (tidak dipakai langsung).",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate awal.")
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=1e-5,
        help="Kompatibilitas argumen dengan model lain (tidak dipakai langsung).",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout classifier head YOLOv8.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help="Patience early stopping.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Batasi jumlah data per kelas untuk semua split.",
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=None,
        help="Batasi jumlah data per kelas khusus split train.",
    )
    parser.add_argument(
        "--max-validation-per-class",
        type=int,
        default=None,
        help="Batasi jumlah data per kelas khusus split validation.",
    )
    parser.add_argument(
        "--max-test-per-class",
        type=int,
        default=None,
        help="Batasi jumlah data per kelas khusus split testing.",
    )
    parser.add_argument(
        "--train-batch-limit",
        type=int,
        default=None,
        help="Maksimal batch train (dikoversi menjadi batas jumlah sampel train).",
    )
    parser.add_argument(
        "--validation-batch-limit",
        type=int,
        default=None,
        help="Maksimal batch validation (dikoversi menjadi batas jumlah sampel val).",
    )
    parser.add_argument(
        "--test-batch-limit",
        type=int,
        default=None,
        help="Maksimal batch testing (dikoversi menjadi batas jumlah sampel test).",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=2048,
        help="Kompatibilitas argumen (shuffle dikelola internal YOLOv8).",
    )
    parser.add_argument(
        "--num-parallel-calls",
        type=int,
        default=2,
        help="Jumlah worker DataLoader YOLOv8.",
    )
    parser.add_argument(
        "--prefetch-buffer",
        type=int,
        default=1,
        help="Kompatibilitas argumen (prefetch dikelola internal YOLOv8).",
    )
    parser.add_argument(
        "--gpu-memory-limit-mb",
        type=int,
        default=None,
        help="Kompatibilitas argumen (belum didukung langsung pada YOLOv8/PyTorch).",
    )
    parser.add_argument(
        "--max-cpu-usage-percent",
        type=int,
        default=70,
        help="Target maksimum penggunaan logical CPU dalam persen.",
    )
    parser.add_argument(
        "--cpu-thread-limit",
        type=int,
        default=None,
        help="Batas absolut thread CPU (opsional, override persen).",
    )
    parser.add_argument(
        "--disable-cpu-fallback",
        action="store_true",
        help="Nonaktifkan fallback otomatis ke CPU saat GPU tidak tersedia/penuh.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Aktifkan mixed precision untuk training GPU.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Gunakan arsitektur YAML tanpa bobot pretrained.",
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Nonaktifkan augmentasi train bawaan YOLOv8.",
    )
    parser.add_argument(
        "--yolo-size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="Ukuran backbone YOLOv8 classification.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
