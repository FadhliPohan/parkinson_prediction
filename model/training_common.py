import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from augmentation_selected.registry import get_profile_choices, load_profile, resolve_profile_names
from experiment_utils import (
    build_run_name,
    dump_json,
    ensure_run_directories,
    update_latest_run_pointer,
    write_experiment_summary,
    write_run_metadata,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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
        pass
    print("Training dilanjutkan di CPU.")
    print(
        "Batas CPU: {} thread (target <= {}% logical CPU).".format(
            cpu_thread_limit,
            min(100, max(10, int(max_cpu_usage_percent))),
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


def make_base_image_dataset(
    filepaths: List[str],
    labels: List[int],
    image_size: Tuple[int, int],
    binary: bool,
    num_parallel_calls: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)

    map_calls = max(1, int(num_parallel_calls))
    dataset = dataset.map(
        lambda p, l: _decode_and_resize_image(p, l, image_size, binary),
        num_parallel_calls=map_calls,
    )
    return dataset.cache()


def make_static_train_dataset(
    filepaths: List[str],
    labels: List[int],
    image_size: Tuple[int, int],
    batch_size: int,
    seed: int,
    binary: bool,
    shuffle_buffer_size: int,
    num_parallel_calls: int,
    prefetch_buffer: int,
    batch_limit: Optional[int],
    augmentation_profile: Dict[str, object],
    augmentation_copies: int,
) -> Tuple[tf.data.Dataset, int]:
    base_dataset = make_base_image_dataset(
        filepaths=filepaths,
        labels=labels,
        image_size=image_size,
        binary=binary,
        num_parallel_calls=num_parallel_calls,
    )
    combined_dataset = base_dataset
    total_samples = len(labels)
    map_calls = max(1, int(num_parallel_calls))

    if bool(augmentation_profile.get("enabled")):
        build_fn = augmentation_profile.get("build_fn")
        if not callable(build_fn):
            raise ValueError("Profile augmentasi {} tidak memiliki build_fn yang valid.".format(augmentation_profile["name"]))
        copy_count = max(1, int(augmentation_copies))
        for copy_index in range(copy_count):
            augmenter = build_fn(seed=seed + (copy_index * 97))
            augmented_dataset = base_dataset.batch(batch_size)
            augmented_dataset = augmented_dataset.map(
                lambda images, batch_labels: _apply_batch_augmentation(images, batch_labels, augmenter),
                num_parallel_calls=map_calls,
            )
            augmented_dataset = augmented_dataset.unbatch().cache()
            combined_dataset = combined_dataset.concatenate(augmented_dataset)
            total_samples += len(labels)

    buffer_size = min(max(1, shuffle_buffer_size), max(1, total_samples))
    combined_dataset = combined_dataset.shuffle(
        buffer_size=buffer_size,
        seed=seed,
        reshuffle_each_iteration=True,
    )
    combined_dataset = combined_dataset.batch(batch_size)
    if batch_limit is not None and batch_limit > 0:
        combined_dataset = combined_dataset.take(batch_limit)
    combined_dataset = combined_dataset.prefetch(max(1, int(prefetch_buffer)))
    return combined_dataset, total_samples


def make_eval_dataset(
    filepaths: List[str],
    labels: List[int],
    image_size: Tuple[int, int],
    batch_size: int,
    binary: bool,
    num_parallel_calls: int,
    prefetch_buffer: int,
    batch_limit: Optional[int],
) -> tf.data.Dataset:
    dataset = make_base_image_dataset(
        filepaths=filepaths,
        labels=labels,
        image_size=image_size,
        binary=binary,
        num_parallel_calls=num_parallel_calls,
    )
    dataset = dataset.batch(batch_size)
    if batch_limit is not None and batch_limit > 0:
        dataset = dataset.take(batch_limit)
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
            tf.keras.metrics.AUC(name="auc", multi_label=True),
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


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str,
) -> None:
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
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def plot_roc_curve_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str,
) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC (AUC={:.4f})".format(auc_value))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
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
    title: str,
) -> float:
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    auc_value = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")

    plt.figure(figsize=(7, 6))
    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], y_prob[:, class_index])
        class_auc = roc_auc_score(y_true_bin[:, class_index], y_prob[:, class_index])
        plt.plot(fpr, tpr, label="{} (AUC={:.4f})".format(class_name, class_auc))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
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


def plot_metrics_table(metrics: Dict[str, object], output_path: Path) -> None:
    table_df = pd.DataFrame(
        [{"Metric": key, "Value": value} for key, value in metrics.items()]
    )
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.55 * len(table_df))))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    plt.title("Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def evaluate_predictions(
    split_name: str,
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    class_names: List[str],
    output_dir: Path,
) -> Dict[str, object]:
    is_binary = len(class_names) == 2
    aligned_prob = y_pred_prob
    aligned_true = y_true
    if len(aligned_prob) != len(aligned_true):
        aligned_count = min(len(aligned_prob), len(aligned_true))
        aligned_prob = aligned_prob[:aligned_count]
        aligned_true = aligned_true[:aligned_count]

    if is_binary:
        y_prob = aligned_prob.reshape(-1)
        y_pred = (y_prob >= 0.5).astype(np.int32)
        f1_value = f1_score(aligned_true, y_pred, average="binary")
        try:
            roc_auc_value = plot_roc_curve_binary(
                y_true=aligned_true,
                y_prob=y_prob,
                output_path=output_dir / "roc_curve_{}.png".format(split_name),
                title="ROC Curve ({})".format(split_name.title()),
            )
        except Exception:
            roc_auc_value = float("nan")
    else:
        y_prob = aligned_prob
        y_pred = np.argmax(y_prob, axis=1)
        f1_value = f1_score(aligned_true, y_pred, average="macro")
        try:
            roc_auc_value = plot_roc_curve_multiclass(
                y_true=aligned_true,
                y_prob=y_prob,
                class_names=class_names,
                output_path=output_dir / "roc_curve_{}.png".format(split_name),
                title="ROC Curve ({})".format(split_name.title()),
            )
        except Exception:
            roc_auc_value = float("nan")

    label_indices = list(range(len(class_names)))
    accuracy_value = accuracy_score(aligned_true, y_pred)
    cm = confusion_matrix(aligned_true, y_pred, labels=label_indices)
    cls_report = classification_report(
        aligned_true,
        y_pred,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    classification_df = pd.DataFrame(cls_report).transpose()
    classification_df.to_csv(str(output_dir / "classification_report_{}.csv".format(split_name)))

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(str(output_dir / "confusion_matrix_{}.csv".format(split_name)))
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        output_path=output_dir / "confusion_matrix_{}.png".format(split_name),
        title="Confusion Matrix ({})".format(split_name.title()),
    )

    return {
        "metrics": {
            "accuracy": float(accuracy_value),
            "f1_score": float(f1_value),
            "roc_auc": float(roc_auc_value) if not np.isnan(roc_auc_value) else float("nan"),
            "samples": int(len(aligned_true)),
        },
        "classification_df": classification_df,
        "confusion_df": cm_df,
    }


def _run_single_training_pipeline(
    model_name: str,
    backbone_builder: Callable,
    preprocess_fn: Callable,
    args: argparse.Namespace,
    custom_objects: Optional[Dict[str, object]],
    augmentation_profile_name: str,
    compute_device: str,
    cpu_thread_limit: int,
) -> Dict[str, Path]:
    tf.keras.backend.clear_session()
    set_global_seed(args.seed)

    dataset_dir = (PROJECT_ROOT / args.dataset_dir).resolve()
    ensure_split_structure(dataset_dir)

    class_names = get_class_names(dataset_dir)
    class_to_index = dict((name, idx) for idx, name in enumerate(class_names))
    num_classes = len(class_names)
    is_binary = num_classes == 2

    profile = load_profile(augmentation_profile_name)
    run_name = build_run_name(model_name=model_name, augmentation_tag=str(profile["run_tag"]))
    run_dirs = ensure_run_directories(PROJECT_ROOT, model_name, run_name)
    report_root = run_dirs["report_root"]
    models_root = run_dirs["model_root"]
    run_report_dir = run_dirs["report_dir"]
    run_model_dir = run_dirs["model_dir"]

    split_paths = {
        "train": dataset_dir / "train",
        "testing": dataset_dir / "testing",
        "validation": dataset_dir / "validation",
    }

    train_limit = pick_split_limit(args.max_per_class, args.max_train_per_class)
    test_limit = pick_split_limit(args.max_per_class, args.max_test_per_class)
    val_limit = pick_split_limit(args.max_per_class, args.max_validation_per_class)

    train_paths, train_labels, train_counts = collect_split_files(
        split_paths["train"],
        class_names,
        class_to_index,
        train_limit,
        args.seed,
    )
    test_paths, test_labels, test_counts = collect_split_files(
        split_paths["testing"],
        class_names,
        class_to_index,
        test_limit,
        args.seed + 1,
    )
    val_paths, val_labels, val_counts = collect_split_files(
        split_paths["validation"],
        class_names,
        class_to_index,
        val_limit,
        args.seed + 2,
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

    augmentation_copies = 0 if not bool(profile["enabled"]) else max(1, int(args.augmentation_copies))

    train_ds, total_train_after_augmentation = make_static_train_dataset(
        filepaths=train_paths,
        labels=train_labels,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        seed=args.seed,
        binary=is_binary,
        shuffle_buffer_size=args.shuffle_buffer_size,
        num_parallel_calls=args.num_parallel_calls,
        prefetch_buffer=args.prefetch_buffer,
        batch_limit=train_batch_limit,
        augmentation_profile=profile,
        augmentation_copies=augmentation_copies,
    )
    val_ds = make_eval_dataset(
        filepaths=val_paths,
        labels=val_labels,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        binary=is_binary,
        num_parallel_calls=args.num_parallel_calls,
        prefetch_buffer=args.prefetch_buffer,
        batch_limit=validation_batch_limit,
    )
    test_ds = make_eval_dataset(
        filepaths=test_paths,
        labels=test_labels,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        binary=is_binary,
        num_parallel_calls=args.num_parallel_calls,
        prefetch_buffer=args.prefetch_buffer,
        batch_limit=test_batch_limit,
    )

    effective_train_samples = effective_sample_count(total_train_after_augmentation, args.batch_size, train_batch_limit)
    effective_validation_samples = effective_sample_count(len(val_labels), args.batch_size, validation_batch_limit)
    effective_test_samples = effective_sample_count(len(test_labels), args.batch_size, test_batch_limit)

    print("\n=== Konfigurasi Eksperimen {} ===".format(run_name))
    print("Model                     :", model_name)
    print("Profile augmentasi        :", profile["display_name"])
    print("Augmentasi statis aktif   :", "ya" if profile["enabled"] else "tidak")
    print("Jumlah copy augmentasi    :", augmentation_copies)
    print("Batch size                :", args.batch_size)
    print("Train batch limit/epoch   :", train_batch_limit if train_batch_limit else "full")
    print("Validation batch limit    :", validation_batch_limit if validation_batch_limit else "full")
    print("Test batch limit          :", test_batch_limit if test_batch_limit else "full")
    print("Shuffle buffer            :", args.shuffle_buffer_size)
    print("Parallel map calls        :", args.num_parallel_calls)
    print("Prefetch buffer           :", args.prefetch_buffer)
    print("Device aktif              :", "GPU" if compute_device == "gpu" else "CPU")
    print("Sample train asli         :", len(train_labels))
    print("Sample train setelah aug  :", total_train_after_augmentation)
    print("Sample train dipakai      :", effective_train_samples)
    print("Sample validation dipakai :", effective_validation_samples)
    print("Sample test dipakai       :", effective_test_samples)
    print("Kebijakan augmentasi:")
    for description in profile["policy_lines"]:
        print("- {}".format(description))

    input_shape = (args.image_size, args.image_size, 3)
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
            print("\n=== Training {} (Stage 1 - {}) ===".format(run_name, device_label))
            history_stage1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=1,
            ).history

            full_history = history_stage1

            if args.fine_tune_epochs > 0:
                print("\n=== Fine Tuning {} (Stage 2 - {}) ===".format(run_name, device_label))
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

    predict_context = tf.device("/CPU:0") if final_compute_device == "cpu" else nullcontext()
    with predict_context:
        val_pred_prob = model.predict(val_ds, verbose=0)
        test_pred_prob = model.predict(test_ds, verbose=0)

    y_val_true = np.array(val_labels[:effective_validation_samples], dtype=np.int32)
    y_test_true = np.array(test_labels[:effective_test_samples], dtype=np.int32)

    validation_outputs = evaluate_predictions(
        split_name="validation",
        y_true=y_val_true,
        y_pred_prob=val_pred_prob,
        class_names=class_names,
        output_dir=run_report_dir,
    )
    testing_outputs = evaluate_predictions(
        split_name="testing",
        y_true=y_test_true,
        y_pred_prob=test_pred_prob,
        class_names=class_names,
        output_dir=run_report_dir,
    )

    validation_metrics = validation_outputs["metrics"]
    testing_metrics = testing_outputs["metrics"]

    testing_outputs["classification_df"].to_csv(str(run_report_dir / "classification_report.csv"))
    testing_outputs["confusion_df"].to_csv(str(run_report_dir / "confusion_matrix.csv"))
    if (run_report_dir / "roc_curve_testing.png").exists():
        tf.io.gfile.copy(
            str(run_report_dir / "roc_curve_testing.png"),
            str(run_report_dir / "roc_curve.png"),
            overwrite=True,
        )
    if (run_report_dir / "confusion_matrix_testing.png").exists():
        tf.io.gfile.copy(
            str(run_report_dir / "confusion_matrix_testing.png"),
            str(run_report_dir / "confusion_matrix.png"),
            overwrite=True,
        )

    metrics_summary = {
        "run_name": run_name,
        "model_name": model_name,
        "device_used": final_compute_device,
        "augmentation_profile": str(profile["name"]),
        "augmentation_display_name": str(profile["display_name"]),
        "augmentation_enabled": bool(profile["enabled"]),
        "augmentation_copies": int(augmentation_copies),
        "primary_split": "testing",
        "accuracy": float(testing_metrics["accuracy"]),
        "f1_score": float(testing_metrics["f1_score"]),
        "roc_auc": float(testing_metrics["roc_auc"]) if not np.isnan(testing_metrics["roc_auc"]) else float("nan"),
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "validation_f1_score": float(validation_metrics["f1_score"]),
        "validation_roc_auc": float(validation_metrics["roc_auc"]) if not np.isnan(validation_metrics["roc_auc"]) else float("nan"),
        "test_samples": int(testing_metrics["samples"]),
        "validation_samples": int(validation_metrics["samples"]),
        "train_samples_total_before_augmentation": int(len(train_labels)),
        "train_samples_total_after_augmentation": int(total_train_after_augmentation),
        "train_samples_used_per_epoch": int(effective_train_samples),
        "validation_samples_used": int(effective_validation_samples),
        "train_samples_total": int(len(train_labels)),
        "validation_samples_total": int(len(val_labels)),
        "test_samples_total": int(len(test_labels)),
        "split_metrics": {
            "validation": validation_metrics,
            "testing": testing_metrics,
        },
    }

    history_df = pd.DataFrame(full_history)
    history_df.to_csv(str(run_report_dir / "training_history.csv"), index=False)

    split_df = pd.DataFrame(split_counts).transpose()
    split_df.to_csv(str(run_report_dir / "split_distribution.csv"))

    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(str(run_report_dir / "evaluation_metrics.csv"), index=False)

    plot_training_curves(history_df, run_report_dir / "training_curves.png")
    plot_split_distribution(split_counts, run_report_dir / "split_distribution.png")
    plot_metrics_table(
        {
            "testing_accuracy": metrics_summary["accuracy"],
            "testing_f1_score": metrics_summary["f1_score"],
            "testing_roc_auc": metrics_summary["roc_auc"],
            "validation_accuracy": metrics_summary["validation_accuracy"],
            "validation_f1_score": metrics_summary["validation_f1_score"],
            "validation_roc_auc": metrics_summary["validation_roc_auc"],
            "train_samples_before_aug": metrics_summary["train_samples_total_before_augmentation"],
            "train_samples_after_aug": metrics_summary["train_samples_total_after_augmentation"],
            "device_used": metrics_summary["device_used"],
        },
        run_report_dir / "evaluation_table.png",
    )

    dump_json(run_report_dir / "evaluation_metrics.json", metrics_summary)
    dump_json(
        run_report_dir / "split_metrics.json",
        {
            "validation": validation_metrics,
            "testing": testing_metrics,
        },
    )

    with open(str(run_model_dir / "class_names.json"), "w", encoding="utf-8") as fp:
        json.dump(class_names, fp, indent=2)

    metadata = {
        "run_name": run_name,
        "model_name": model_name,
        "model_family": "deep_learning",
        "model_category": "cnn_or_transformer",
        "classifier_name": model_name,
        "feature_extractor": None,
        "augmentation_profile": str(profile["name"]),
        "augmentation_display_name": str(profile["display_name"]),
        "augmentation_enabled": bool(profile["enabled"]),
        "augmentation_copies": int(augmentation_copies),
        "dataset_dir": str(dataset_dir),
        "image_size": int(args.image_size),
        "class_names": class_names,
        "split_counts": split_counts,
        "training_config": dict(vars(args)),
        "artifact_paths": {
            "report_dir": str(run_report_dir),
            "model_dir": str(run_model_dir),
            "best_model": str(best_model_path),
            "final_model": str(final_model_path),
        },
        "pipeline_steps": [
            "Baca split train/validation/testing dari dataset/split.",
            "Terapkan augmentasi statis hanya pada train sesuai profile yang dipilih.",
            "Latih backbone transfer learning pada data train.",
            "Opsional fine-tuning backbone pada stage kedua.",
            "Evaluasi validation dan testing tanpa augmentasi acak.",
        ],
    }
    write_run_metadata(run_report_dir, run_model_dir, metadata)
    write_experiment_summary(run_report_dir, metadata, metrics_summary)

    run_summary_lines = [
        "Run Name: {}".format(run_name),
        "Model: {}".format(model_name),
        "Model Family: deep_learning",
        "Augmentasi: {} ({})".format(profile["display_name"], profile["name"]),
        "Dataset: {}".format(dataset_dir),
        "Report dir: {}".format(run_report_dir),
        "Model dir: {}".format(run_model_dir),
        "Best model: {}".format(best_model_path),
        "Final model: {}".format(final_model_path),
        "Device: {}".format(final_compute_device),
        "Train samples sebelum augmentasi: {}".format(len(train_labels)),
        "Train samples sesudah augmentasi: {}".format(total_train_after_augmentation),
        "Validation accuracy: {:.4f}".format(metrics_summary["validation_accuracy"]),
        "Validation F1-score: {:.4f}".format(metrics_summary["validation_f1_score"]),
        "Validation ROC-AUC: {}".format(
            "{:.4f}".format(metrics_summary["validation_roc_auc"])
            if not np.isnan(metrics_summary["validation_roc_auc"])
            else "NaN"
        ),
        "Testing accuracy: {:.4f}".format(metrics_summary["accuracy"]),
        "Testing F1-score: {:.4f}".format(metrics_summary["f1_score"]),
        "Testing ROC-AUC: {}".format(
            "{:.4f}".format(metrics_summary["roc_auc"])
            if not np.isnan(metrics_summary["roc_auc"])
            else "NaN"
        ),
        "",
        "Alur Eksperimen:",
        "1. Split data dibaca dari folder train, validation, dan testing.",
        "2. Augmentasi hanya diterapkan pada data train secara statis per run.",
        "3. Model dilatih menggunakan transfer learning dan fine-tuning opsional.",
        "4. Validation dan testing dievaluasi tanpa augmentasi acak.",
    ]
    with open(str(run_report_dir / "summary.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(run_summary_lines))

    update_latest_run_pointer(report_root, run_name)
    update_latest_run_pointer(models_root, run_name)

    print("\n=== Ringkasan Evaluasi {} ===".format(run_name))
    print("Device               : {}".format(final_compute_device.upper()))
    print("Validation accuracy  : {:.4f}".format(metrics_summary["validation_accuracy"]))
    print("Validation F1-score  : {:.4f}".format(metrics_summary["validation_f1_score"]))
    if np.isnan(metrics_summary["validation_roc_auc"]):
        print("Validation ROC-AUC   : NaN")
    else:
        print("Validation ROC-AUC   : {:.4f}".format(metrics_summary["validation_roc_auc"]))
    print("Testing accuracy     : {:.4f}".format(metrics_summary["accuracy"]))
    print("Testing F1-score     : {:.4f}".format(metrics_summary["f1_score"]))
    if np.isnan(metrics_summary["roc_auc"]):
        print("Testing ROC-AUC      : NaN")
    else:
        print("Testing ROC-AUC      : {:.4f}".format(metrics_summary["roc_auc"]))
    print("Report               : {}".format(run_report_dir))
    print("Model                : {}".format(run_model_dir))

    return {
        "report_dir": run_report_dir,
        "model_dir": run_model_dir,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
    }


def run_training_pipeline(
    model_name: str,
    backbone_builder: Callable,
    preprocess_fn: Callable,
    args: argparse.Namespace,
    custom_objects: Optional[Dict[str, object]] = None,
) -> Dict[str, Path]:
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

    selected_profiles = resolve_profile_names(args.augmentation_profile)
    results: Dict[str, Path] = {}
    last_result: Dict[str, Path] = {}

    for profile_name in selected_profiles:
        result = _run_single_training_pipeline(
            model_name=model_name,
            backbone_builder=backbone_builder,
            preprocess_fn=preprocess_fn,
            args=args,
            custom_objects=custom_objects,
            augmentation_profile_name=profile_name,
            compute_device=compute_device,
            cpu_thread_limit=cpu_thread_limit,
        )
        results[profile_name] = result["model_dir"]
        last_result = result

    return last_result


def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-dir", type=str, default="dataset/split", help="Folder dataset hasil split.")
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
        "--augmentation-profile",
        type=str,
        default="mycostum_augment",
        choices=get_profile_choices(),
        help="Pilih profile augmentasi: tanpa augmentasi, augmentasi kustom, atau jalankan semua profile.",
    )
    parser.add_argument(
        "--augmentation-copies",
        type=int,
        default=1,
        help="Jumlah copy augmentasi statis yang ditambahkan ke data train saat augmentasi aktif.",
    )
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
    return parser
