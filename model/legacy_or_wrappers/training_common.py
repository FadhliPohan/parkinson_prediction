import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.datasets.transforms import (
    build_training_augmentation,
    describe_augmentation_policy,
)

from src.reporting.report_writer import (
    build_artifact_dirs,
    generate_run_id,
    write_latest_run_marker,
    write_run_manifest,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_train_augmentation_bundle() -> Tuple[tf.keras.layers.Layer, List[str]]:
    augmenter = build_training_augmentation()
    if not isinstance(augmenter, tf.keras.layers.Layer):
        raise TypeError("build_training_augmentation() harus mengembalikan tf.keras.layers.Layer")

    policy_lines = describe_augmentation_policy()
    return augmenter, [str(line) for line in policy_lines]


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

    inter_op_threads = max(1, min(4, thread_limit // 2 if thread_limit > 1 else 1))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(thread_limit)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    except Exception as exc:
        print("Peringatan: gagal mengatur batas thread CPU -> {}".format(exc))

    print("\n=== Kontrol CPU ===")
    print("Logical CPU terdeteksi    :", total_logical_cpu)
    print("Target maksimal CPU (%)   :", safe_percent)
    print("Batas thread CPU aktif    :", thread_limit)
    return thread_limit


def activate_cpu_mode(reason: str, cpu_thread_limit: int, max_cpu_usage_percent: int) -> None:
    print("\n[CPU Fallback] {}".format(reason))
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        # Aman diabaikan jika runtime sudah inisialisasi.
        pass
    print("Training dilanjutkan di CPU.")
    print(
        "Batas CPU: {} thread (target <= {}% logical CPU).".format(
            cpu_thread_limit, min(100, max(10, int(max_cpu_usage_percent)))
        )
    )


def configure_compute_device(
    gpu_memory_limit_mb: Optional[int],
    allow_cpu_fallback: bool,
    cpu_thread_limit: int,
    max_cpu_usage_percent: int,
) -> str:
    gpu_devices = tf.config.list_physical_devices("GPU")
    if not gpu_devices:
        if allow_cpu_fallback:
            activate_cpu_mode(
                "GPU tidak terdeteksi.",
                cpu_thread_limit=cpu_thread_limit,
                max_cpu_usage_percent=max_cpu_usage_percent,
            )
            return "cpu"
        print("\n[TRAINING DIBATALKAN] GPU tidak terdeteksi.")
        print("Pastikan TensorFlow GPU dan driver CUDA/cuDNN sudah terpasang dengan benar.")
        raise SystemExit(1)

    if gpu_memory_limit_mb is not None and gpu_memory_limit_mb > 0:
        try:
            tf.config.set_logical_device_configuration(
                gpu_devices[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit_mb)],
            )
        except Exception as exc:
            print("Peringatan: gagal set batas memori GPU -> {}".format(exc))
            # Jika gagal set limit (mis. runtime sudah inisialisasi), fallback ke memory growth.
            for device in gpu_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except Exception:
                    pass
    else:
        for device in gpu_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except Exception:
                # Aman diabaikan jika memory growth tidak bisa diubah pada environment tertentu.
                pass

    print("\nGPU terdeteksi. Training akan menggunakan GPU:")
    for device in gpu_devices:
        print("- {}".format(device.name))
    if gpu_memory_limit_mb is not None and gpu_memory_limit_mb > 0:
        print("Batas memori GPU: {} MB".format(gpu_memory_limit_mb))
    return "gpu"


def is_gpu_memory_error(exc: Exception) -> bool:
    if isinstance(exc, tf.errors.ResourceExhaustedError):
        return True
    message = str(exc).lower()
    memory_keywords = [
        "resource exhausted",
        "out of memory",
        "oom",
        "cuda_error_out_of_memory",
        "failed to allocate",
        "cudnn_status_alloc_failed",
    ]
    return any(keyword in message for keyword in memory_keywords)


def apply_runtime_optimizations(enable_mixed_precision: bool, using_gpu: bool) -> None:
    if not enable_mixed_precision:
        return
    if not using_gpu:
        print("Mixed precision dilewati karena mode CPU.")
        return

    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision aktif (float16) untuk mengurangi pemakaian memori GPU.")
    except Exception as exc:
        print("Peringatan: gagal mengaktifkan mixed precision -> {}".format(exc))


def pick_split_limit(default_limit: Optional[int], split_limit: Optional[int]) -> Optional[int]:
    if split_limit is not None and split_limit > 0:
        return split_limit
    return default_limit


def effective_sample_count(
    total_samples: int,
    batch_size: int,
    batch_limit: Optional[int],
) -> int:
    if batch_limit is not None and batch_limit > 0:
        return min(total_samples, batch_limit * batch_size)
    return total_samples


def ensure_split_structure(dataset_dir: Path) -> None:
    required_splits = ["train", "testing", "validation"]
    missing = []
    for split_name in required_splits:
        split_path = dataset_dir / split_name
        if not split_path.exists() or not split_path.is_dir():
            missing.append(str(split_path))
    if missing:
        raise FileNotFoundError("Folder split tidak lengkap: {}".format(", ".join(missing)))


def get_class_names(dataset_dir: Path) -> List[str]:
    train_dir = dataset_dir / "train"
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if len(class_names) < 2:
        raise ValueError("Minimal harus ada 2 kelas. Saat ini ditemukan: {}".format(class_names))
    return class_names


def list_image_files(folder: Path) -> List[Path]:
    return sorted(
        [
            f
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def collect_split_files(
    split_dir: Path,
    class_names: List[str],
    class_to_index: Dict[str, int],
    max_per_class: Optional[int],
    seed: int,
) -> Tuple[List[str], List[int], Dict[str, int]]:
    rng = random.Random(seed)
    filepaths: List[str] = []
    labels: List[int] = []
    class_counts: Dict[str, int] = {}

    for class_name in class_names:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError("Folder kelas tidak ditemukan: {}".format(class_dir))

        files = list_image_files(class_dir)
        if max_per_class is not None and max_per_class > 0:
            rng.shuffle(files)
            files = files[:max_per_class]

        class_counts[class_name] = len(files)
        for image_path in files:
            filepaths.append(str(image_path))
            labels.append(class_to_index[class_name])

    return filepaths, labels, class_counts


def _decode_and_resize_image(path: tf.Tensor, label: tf.Tensor, image_size: Tuple[int, int], binary: bool):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size, method="bilinear")
    image = tf.cast(image, tf.float32)
    if binary:
        label = tf.cast(label, tf.float32)
    else:
        label = tf.cast(label, tf.int32)
    return image, label


def _apply_batch_augmentation(
    images: tf.Tensor,
    labels: tf.Tensor,
    train_augmenter: tf.keras.layers.Layer,
):
    augmented_images = train_augmenter(images, training=True)
    augmented_images = tf.cast(augmented_images, tf.float32)
    augmented_images = tf.clip_by_value(augmented_images, 0.0, 255.0)
    return augmented_images, labels


def make_dataset(
    filepaths: List[str],
    labels: List[int],
    image_size: Tuple[int, int],
    batch_size: int,
    seed: int,
    shuffle: bool,
    binary: bool,
    shuffle_buffer_size: int,
    num_parallel_calls: int,
    prefetch_buffer: int,
    batch_limit: Optional[int],
    apply_train_augmentation: bool = False,
    train_augmenter: Optional[tf.keras.layers.Layer] = None,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    options = tf.data.Options()
    options.experimental_deterministic = not shuffle
    dataset = dataset.with_options(options)

    if shuffle:
        buffer_size = min(max(1, shuffle_buffer_size), max(1, len(filepaths)))
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    map_calls = max(1, int(num_parallel_calls))
    dataset = dataset.map(
        lambda p, l: _decode_and_resize_image(p, l, image_size, binary),
        num_parallel_calls=map_calls,
    )
    dataset = dataset.batch(batch_size)

    if batch_limit is not None and batch_limit > 0:
        dataset = dataset.take(batch_limit)

    if apply_train_augmentation:
        if train_augmenter is None:
            raise ValueError("train_augmenter wajib diisi saat apply_train_augmentation=True.")
        dataset = dataset.map(
            lambda images, batch_labels: _apply_batch_augmentation(images, batch_labels, train_augmenter),
            num_parallel_calls=map_calls,
        )

    dataset = dataset.prefetch(max(1, int(prefetch_buffer)))
    return dataset


def build_model(
    backbone_builder: Callable,
    preprocess_fn: Callable,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float,
    use_pretrained: bool,
    learning_rate: float,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    weights = "imagenet" if use_pretrained else None
    try:
        base_model = backbone_builder(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
    except Exception as exc:
        if use_pretrained:
            print(
                "Peringatan: gagal memuat bobot ImageNet ({}). Fallback ke bobot acak.".format(exc)
            )
            base_model = backbone_builder(
                include_top=False,
                weights=None,
                input_shape=input_shape,
            )
        else:
            raise

    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = preprocess_fn(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)

    if num_classes == 2:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier_model")
    loss, metrics = build_loss_and_metrics(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model, base_model


def build_loss_and_metrics(num_classes: int):
    if num_classes == 2:
        return (
            tf.keras.losses.BinaryCrossentropy(),
            [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
    return (
        tf.keras.losses.SparseCategoricalCrossentropy(),
        [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )


def merge_histories(history_a: Dict[str, List[float]], history_b: Dict[str, List[float]]) -> Dict[str, List[float]]:
    merged: Dict[str, List[float]] = {}
    all_keys = set(history_a.keys()).union(set(history_b.keys()))
    for key in all_keys:
        merged[key] = list(history_a.get(key, [])) + list(history_b.get(key, []))
    return merged


def plot_training_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    if history_df.empty:
        return
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if "loss" in history_df:
        plt.plot(history_df["loss"], label="train_loss")
    if "val_loss" in history_df:
        plt.plot(history_df["val_loss"], label="val_loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if "loss" in history_df or "val_loss" in history_df:
        plt.legend()
    plt.grid(alpha=0.3, linestyle="--")

    plt.subplot(1, 2, 2)
    if "accuracy" in history_df:
        plt.plot(history_df["accuracy"], label="train_accuracy")
    if "val_accuracy" in history_df:
        plt.plot(history_df["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    if "accuracy" in history_df or "val_accuracy" in history_df:
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
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    auc_value = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")

    plt.figure(figsize=(7, 6))
    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], y_prob[:, class_index])
        plt.plot(fpr, tpr, label="{} (AUC={:.4f})".format(class_name, roc_auc_score(y_true_bin[:, class_index], y_prob[:, class_index])))
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


def _first_existing_series(history_df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    for col in candidates:
        if col in history_df.columns:
            return history_df[col]
    return None


def extract_final_training_metrics(history_df: pd.DataFrame) -> Dict[str, float]:
    if history_df.empty:
        return {
            "train_accuracy": float("nan"),
            "val_accuracy": float("nan"),
            "train_loss": float("nan"),
            "val_loss": float("nan"),
            "epochs_trained": 0,
        }

    train_acc = _first_existing_series(
        history_df,
        ["accuracy", "train_accuracy", "train/accuracy", "metrics/accuracy_top1"],
    )
    val_acc = _first_existing_series(
        history_df,
        ["val_accuracy", "validation_accuracy", "val/accuracy_top1", "val_accuracy_top1"],
    )
    train_loss = _first_existing_series(
        history_df,
        ["loss", "train_loss", "train/loss", "train/cls_loss"],
    )
    val_loss = _first_existing_series(
        history_df,
        ["val_loss", "validation_loss", "val/loss", "val/cls_loss"],
    )

    return {
        "train_accuracy": float(train_acc.iloc[-1]) if train_acc is not None and len(train_acc) else float("nan"),
        "val_accuracy": float(val_acc.iloc[-1]) if val_acc is not None and len(val_acc) else float("nan"),
        "train_loss": float(train_loss.iloc[-1]) if train_loss is not None and len(train_loss) else float("nan"),
        "val_loss": float(val_loss.iloc[-1]) if val_loss is not None and len(val_loss) else float("nan"),
        "epochs_trained": int(len(history_df)),
    }


def run_training_pipeline(
    model_name: str,
    backbone_builder: Callable,
    preprocess_fn: Callable,
    args: argparse.Namespace,
    custom_objects: Optional[Dict[str, object]] = None,
) -> Dict[str, Path]:
    run_started_dt = datetime.now().astimezone()
    run_started_at = run_started_dt.isoformat()
    set_global_seed(args.seed)
    cpu_thread_limit = configure_cpu_runtime(
        max_cpu_usage_percent=args.max_cpu_usage_percent,
        cpu_thread_limit=args.cpu_thread_limit,
    )
    compute_device = configure_compute_device(
        gpu_memory_limit_mb=args.gpu_memory_limit_mb,
        allow_cpu_fallback=not args.disable_cpu_fallback,
        cpu_thread_limit=cpu_thread_limit,
        max_cpu_usage_percent=args.max_cpu_usage_percent,
    )
    apply_runtime_optimizations(
        enable_mixed_precision=args.mixed_precision,
        using_gpu=compute_device == "gpu",
    )

    dataset_dir = (PROJECT_ROOT / args.dataset_dir).resolve()
    ensure_split_structure(dataset_dir)

    class_names = get_class_names(dataset_dir)
    class_to_index = dict((name, idx) for idx, name in enumerate(class_names))
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
        augmentation_id=args.augmentation_id,
        method_id=args.training_method,
        model_name=model_name,
        run_id=run_id,
    )

    split_paths = {
        "train": dataset_dir / "train",
        "testing": dataset_dir / "testing",
        "validation": dataset_dir / "validation",
    }

    train_limit = pick_split_limit(args.max_per_class, args.max_train_per_class)
    test_limit = pick_split_limit(args.max_per_class, args.max_test_per_class)
    val_limit = pick_split_limit(args.max_per_class, args.max_validation_per_class)

    train_paths, train_labels, train_counts = collect_split_files(
        split_paths["train"], class_names, class_to_index, train_limit, args.seed
    )
    test_paths, test_labels, test_counts = collect_split_files(
        split_paths["testing"], class_names, class_to_index, test_limit, args.seed + 1
    )
    val_paths, val_labels, val_counts = collect_split_files(
        split_paths["validation"], class_names, class_to_index, val_limit, args.seed + 2
    )

    if len(train_paths) == 0 or len(test_paths) == 0 or len(val_paths) == 0:
        raise ValueError("Data train/testing/validation kosong. Jalankan split data terlebih dahulu.")

    split_counts = {
        "train": train_counts,
        "testing": test_counts,
        "validation": val_counts,
    }

    train_batch_limit = args.train_batch_limit if args.train_batch_limit and args.train_batch_limit > 0 else None
    validation_batch_limit = (
        args.validation_batch_limit if args.validation_batch_limit and args.validation_batch_limit > 0 else None
    )
    test_batch_limit = args.test_batch_limit if args.test_batch_limit and args.test_batch_limit > 0 else None

    effective_train_samples = effective_sample_count(len(train_labels), args.batch_size, train_batch_limit)
    effective_validation_samples = effective_sample_count(len(val_labels), args.batch_size, validation_batch_limit)
    effective_test_samples = effective_sample_count(len(test_labels), args.batch_size, test_batch_limit)

    print("\n=== Mode Hemat Resource ===")
    print("Batch size                :", args.batch_size)
    print("Train batch limit/epoch   :", train_batch_limit if train_batch_limit else "full")
    print("Validation batch limit    :", validation_batch_limit if validation_batch_limit else "full")
    print("Test batch limit          :", test_batch_limit if test_batch_limit else "full")
    print("Shuffle buffer            :", args.shuffle_buffer_size)
    print("Parallel map calls        :", args.num_parallel_calls)
    print("Prefetch buffer           :", args.prefetch_buffer)
    print("Device aktif              :", "GPU" if compute_device == "gpu" else "CPU")
    print("Sample train dipakai      :", effective_train_samples, "dari", len(train_labels))
    print("Sample validation dipakai :", effective_validation_samples, "dari", len(val_labels))
    print("Sample test dipakai       :", effective_test_samples, "dari", len(test_labels))

    use_train_augmentation = not bool(getattr(args, "disable_augmentation", False))
    train_augmenter = None
    augmentation_policy: List[str] = []
    print("\n=== Augmentasi Train On-the-Fly ===")
    if use_train_augmentation:
        train_augmenter, augmentation_policy = load_train_augmentation_bundle()
        if augmentation_policy:
            for description in augmentation_policy:
                print("- {}".format(description))
        else:
            print("- Augmentasi train aktif dari src/datasets/transforms.py")
    else:
        print("- Dinonaktifkan (menggunakan data train asli tanpa augmentasi).")

    input_shape = (args.image_size, args.image_size, 3)
    train_ds = make_dataset(
        train_paths,
        train_labels,
        (args.image_size, args.image_size),
        args.batch_size,
        args.seed,
        shuffle=True,
        binary=is_binary,
        shuffle_buffer_size=args.shuffle_buffer_size,
        num_parallel_calls=args.num_parallel_calls,
        prefetch_buffer=args.prefetch_buffer,
        batch_limit=train_batch_limit,
        apply_train_augmentation=use_train_augmentation,
        train_augmenter=train_augmenter,
    )
    val_ds = make_dataset(
        val_paths,
        val_labels,
        (args.image_size, args.image_size),
        args.batch_size,
        args.seed,
        shuffle=False,
        binary=is_binary,
        shuffle_buffer_size=args.shuffle_buffer_size,
        num_parallel_calls=args.num_parallel_calls,
        prefetch_buffer=args.prefetch_buffer,
        batch_limit=validation_batch_limit,
    )
    test_ds = make_dataset(
        test_paths,
        test_labels,
        (args.image_size, args.image_size),
        args.batch_size,
        args.seed,
        shuffle=False,
        binary=is_binary,
        shuffle_buffer_size=args.shuffle_buffer_size,
        num_parallel_calls=args.num_parallel_calls,
        prefetch_buffer=args.prefetch_buffer,
        batch_limit=test_batch_limit,
    )

    best_model_path = run_model_dir / "best_model.keras"
    final_model_path = run_model_dir / "final_model.keras"
    final_compute_device = compute_device

    def _run_training_once(force_cpu: bool) -> Tuple[tf.keras.Model, Dict[str, List[float]]]:
        device_context = tf.device("/CPU:0") if force_cpu else nullcontext()

        with device_context:
            model, base_model = build_model(
                backbone_builder=backbone_builder,
                preprocess_fn=preprocess_fn,
                input_shape=input_shape,
                num_classes=num_classes,
                dropout_rate=args.dropout,
                use_pretrained=not args.no_pretrained,
                learning_rate=args.learning_rate,
            )

            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=str(best_model_path),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            )
            callbacks = [
                checkpoint_callback,
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=args.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.3,
                    patience=max(1, args.early_stopping_patience // 2),
                    verbose=1,
                ),
            ]

            device_label = "CPU" if force_cpu else "GPU"
            print("\n=== Training {} (Stage 1 - {}) ===".format(model_name, device_label))
            history_stage1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=1,
            ).history

            full_history = history_stage1

            if args.fine_tune_epochs > 0:
                print("\n=== Fine Tuning {} (Stage 2 - {}) ===".format(model_name, device_label))
                base_model.trainable = True
                freeze_until = int(len(base_model.layers) * args.fine_tune_freeze_ratio)
                for layer in base_model.layers[:freeze_until]:
                    layer.trainable = False

                fine_loss, fine_metrics = build_loss_and_metrics(num_classes)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_learning_rate),
                    loss=fine_loss,
                    metrics=fine_metrics,
                )

                history_stage2 = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=args.fine_tune_epochs,
                    callbacks=callbacks,
                    verbose=1,
                ).history
                full_history = merge_histories(history_stage1, history_stage2)

            model.save(str(final_model_path))
            if best_model_path.exists():
                model = tf.keras.models.load_model(
                    str(best_model_path),
                    custom_objects=custom_objects,
                    compile=False,
                )
            return model, full_history

    try:
        model, full_history = _run_training_once(force_cpu=compute_device == "cpu")
    except Exception as exc:
        can_fallback_to_cpu = (
            compute_device == "gpu"
            and not args.disable_cpu_fallback
            and is_gpu_memory_error(exc)
        )
        if not can_fallback_to_cpu:
            raise

        print("\n[PERINGATAN] GPU penuh saat training: {}".format(exc))
        print("Mencoba ulang otomatis menggunakan CPU...")
        tf.keras.backend.clear_session()
        if best_model_path.exists():
            best_model_path.unlink()
        if final_model_path.exists():
            final_model_path.unlink()
        activate_cpu_mode(
            "GPU kehabisan memori (OOM).",
            cpu_thread_limit=cpu_thread_limit,
            max_cpu_usage_percent=args.max_cpu_usage_percent,
        )
        apply_runtime_optimizations(enable_mixed_precision=args.mixed_precision, using_gpu=False)
        final_compute_device = "cpu"
        model, full_history = _run_training_once(force_cpu=True)

    print("\n=== Evaluasi {} ===".format(model_name))
    y_true = np.array(test_labels[:effective_test_samples], dtype=np.int32)
    predict_context = tf.device("/CPU:0") if final_compute_device == "cpu" else nullcontext()
    with predict_context:
        y_pred_prob = model.predict(test_ds, verbose=0)
    if len(y_pred_prob) != len(y_true):
        aligned_count = min(len(y_pred_prob), len(y_true))
        y_pred_prob = y_pred_prob[:aligned_count]
        y_true = y_true[:aligned_count]

    if is_binary:
        y_prob = y_pred_prob.reshape(-1)
        y_pred = (y_prob >= 0.5).astype(np.int32)
        test_f1 = f1_score(y_true, y_pred, average="binary")
        try:
            test_roc_auc = plot_roc_curve_binary(
                y_true=y_true,
                y_prob=y_prob,
                output_path=run_report_dir / "roc_curve.png",
            )
        except Exception:
            test_roc_auc = float("nan")
    else:
        y_prob = y_pred_prob
        y_pred = np.argmax(y_prob, axis=1)
        test_f1 = f1_score(y_true, y_pred, average="macro")
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
    test_accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)
    cls_report = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    history_df = pd.DataFrame(full_history)
    final_train_metrics = extract_final_training_metrics(history_df)
    run_finished_dt = datetime.now().astimezone()
    run_finished_at = run_finished_dt.isoformat()
    training_time_seconds = float((run_finished_dt - run_started_dt).total_seconds())

    metrics_summary = {
        "device_used": final_compute_device,
        "accuracy": float(test_accuracy),
        "f1_score": float(test_f1),
        "roc_auc": float(test_roc_auc) if not np.isnan(test_roc_auc) else float("nan"),
        "test_samples": int(len(y_true)),
        "train_samples_used_per_epoch": int(effective_train_samples),
        "validation_samples_used": int(effective_validation_samples),
        "train_samples_total": int(len(train_labels)),
        "validation_samples_total": int(len(val_labels)),
        "test_samples_total": int(len(test_labels)),
        "train_accuracy": final_train_metrics.get("train_accuracy"),
        "val_accuracy": final_train_metrics.get("val_accuracy"),
        "train_loss": final_train_metrics.get("train_loss"),
        "val_loss": final_train_metrics.get("val_loss"),
        "epochs_trained": final_train_metrics.get("epochs_trained"),
        "training_time_seconds": training_time_seconds,
    }

    history_df.to_csv(str(run_report_dir / "training_history.csv"), index=False)

    classification_df = pd.DataFrame(cls_report).transpose()
    classification_df.to_csv(str(run_report_dir / "classification_report.csv"))

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(str(run_report_dir / "confusion_matrix.csv"))

    split_df = pd.DataFrame(split_counts).transpose()
    split_df.to_csv(str(run_report_dir / "split_distribution.csv"))

    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(str(run_report_dir / "evaluation_metrics.csv"), index=False)

    plot_training_curves(history_df, run_report_dir / "training_curves.png")
    plot_confusion_matrix(cm, class_names, run_report_dir / "confusion_matrix.png")
    plot_split_distribution(split_counts, run_report_dir / "split_distribution.png")
    plot_metrics_table(metrics_summary, run_report_dir / "evaluation_table.png")

    with open(str(run_report_dir / "evaluation_metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics_summary, fp, indent=2)

    with open(str(run_model_dir / "class_names.json"), "w", encoding="utf-8") as fp:
        json.dump(class_names, fp, indent=2)

    run_summary_lines = [
        "Model: {}".format(model_name),
        "Dataset Name: {}".format(dataset_name),
        "Augmentation ID: {}".format(args.augmentation_id),
        "Augmentation Label: {}".format(args.augmentation_label),
        "Experiment ID: {}".format(args.experiment_id),
        "Training Method: {}".format(args.training_method),
        "Run ID: {}".format(run_id),
        "Dataset: {}".format(dataset_dir),
        "Report dir: {}".format(run_report_dir),
        "Model dir: {}".format(run_model_dir),
        "Best model: {}".format(best_model_path),
        "Final model: {}".format(final_model_path),
        "Device: {}".format(final_compute_device),
        "Train Accuracy (last): {}".format(
            "{:.4f}".format(metrics_summary["train_accuracy"])
            if not np.isnan(metrics_summary["train_accuracy"])
            else "NaN"
        ),
        "Validation Accuracy (last): {}".format(
            "{:.4f}".format(metrics_summary["val_accuracy"])
            if not np.isnan(metrics_summary["val_accuracy"])
            else "NaN"
        ),
        "Train Loss (last): {}".format(
            "{:.4f}".format(metrics_summary["train_loss"])
            if not np.isnan(metrics_summary["train_loss"])
            else "NaN"
        ),
        "Validation Loss (last): {}".format(
            "{:.4f}".format(metrics_summary["val_loss"])
            if not np.isnan(metrics_summary["val_loss"])
            else "NaN"
        ),
        "Accuracy: {:.4f}".format(metrics_summary["accuracy"]),
        "F1-score: {:.4f}".format(metrics_summary["f1_score"]),
        "Training Time (s): {:.2f}".format(training_time_seconds),
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
        "experiment_id": args.experiment_id,
        "run_id": run_id,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "dataset": {
            "dataset_name": dataset_name,
            "dataset_dir": str(dataset_dir),
            "class_names": class_names,
            "class_count": int(num_classes),
            "split_counts": split_counts,
        },
        "model": {
            "model_name": model_name,
            "framework": "tensorflow",
            "custom_objects_used": bool(custom_objects),
        },
        "training": {
            "method": args.training_method,
            "augmentation_id": args.augmentation_id,
            "augmentation_label": args.augmentation_label,
            "experiment_id": args.experiment_id,
            "parameters": vars(args),
            "device_used": final_compute_device,
            "duration_seconds": training_time_seconds,
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

    print("\n=== Ringkasan Evaluasi {} ===".format(model_name))
    print("Device   : {}".format(final_compute_device.upper()))
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


def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
        "--augmentation-id",
        type=str,
        default="augment_on_the_fly",
        help="ID skenario augmentasi (untuk metadata run).",
    )
    parser.add_argument(
        "--augmentation-label",
        type=str,
        default="augmentasi_on_the_fly",
        help="Label skenario augmentasi (untuk metadata run).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="",
        help="ID unik eksperimen untuk kombinasi dataset/augmentasi/method/model.",
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
    parser.add_argument("--fine-tune-epochs", type=int, default=2, help="Jumlah epoch fine-tuning.")
    parser.add_argument(
        "--fine-tune-freeze-ratio",
        type=float,
        default=0.7,
        help="Rasio layer backbone yang dibekukan saat fine-tuning (0-1).",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate stage 1.")
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate stage 2 (fine tuning).",
    )
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout head classifier.")
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
        help="Maksimal jumlah batch train per epoch.",
    )
    parser.add_argument(
        "--validation-batch-limit",
        type=int,
        default=None,
        help="Maksimal jumlah batch validation.",
    )
    parser.add_argument(
        "--test-batch-limit",
        type=int,
        default=None,
        help="Maksimal jumlah batch testing.",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=2048,
        help="Ukuran buffer shuffle untuk mengontrol RAM usage.",
    )
    parser.add_argument(
        "--num-parallel-calls",
        type=int,
        default=2,
        help="Jumlah worker paralel saat decode/resize data.",
    )
    parser.add_argument(
        "--prefetch-buffer",
        type=int,
        default=1,
        help="Ukuran prefetch buffer (lebih kecil = lebih hemat memori).",
    )
    parser.add_argument(
        "--gpu-memory-limit-mb",
        type=int,
        default=None,
        help="Batas maksimum memori GPU dalam MB (opsional).",
    )
    parser.add_argument(
        "--max-cpu-usage-percent",
        type=int,
        default=70,
        help="Target maksimum penggunaan logical CPU dalam persen saat training.",
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
        help="Aktifkan mixed precision untuk menghemat memori GPU.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Gunakan bobot acak (tanpa pretrained ImageNet).",
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Nonaktifkan augmentasi on-the-fly pada split train.",
    )
    return parser
