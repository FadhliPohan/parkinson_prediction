import argparse
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from skimage.feature import hog
except Exception as exc:
    raise ImportError(
        "scikit-image belum terinstall. Jalankan instalasi dependency terbaru sebelum memakai pipeline HOG/GHOG."
    ) from exc


MODULE_DIR = Path(__file__).resolve().parent
MODEL_DIR = MODULE_DIR.parent
PROJECT_ROOT = MODULE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from augmentation_selected.registry import load_profile, resolve_profile_names
from experiment_utils import (
    build_run_name,
    dump_json,
    ensure_run_directories,
    update_latest_run_pointer,
    write_experiment_summary,
    write_run_metadata,
)
from training_common import (
    ensure_split_structure,
    evaluate_predictions,
    get_class_names,
    pick_split_limit,
    plot_metrics_table,
    plot_split_distribution,
    collect_split_files,
)


DEFAULT_IMAGE_SIZE = 224
DEFAULT_HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_classical_cpu_runtime(max_cpu_usage_percent: int, cpu_thread_limit: Optional[int]) -> int:
    total_logical_cpu = os.cpu_count() or 1
    safe_percent = min(100, max(10, int(max_cpu_usage_percent)))
    auto_thread_limit = max(1, int(total_logical_cpu * (safe_percent / 100.0)))
    thread_limit = auto_thread_limit
    if cpu_thread_limit is not None and cpu_thread_limit > 0:
        thread_limit = max(1, min(total_logical_cpu, int(cpu_thread_limit)))

    os.environ["OMP_NUM_THREADS"] = str(thread_limit)
    os.environ["MKL_NUM_THREADS"] = str(thread_limit)
    print("\n=== Kontrol CPU Classical ML ===")
    print("Logical CPU terdeteksi    :", total_logical_cpu)
    print("Target maksimal CPU (%)   :", safe_percent)
    print("Batas thread CPU aktif    :", thread_limit)
    return thread_limit


def load_rgb_image(image_path: str, image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB").resize((image_size, image_size))
    return np.asarray(rgb_image, dtype=np.float32)


def load_rgb_image_from_bytes(image_bytes: bytes, image_size: int) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb_image = image.convert("RGB").resize((image_size, image_size))
    return np.asarray(rgb_image, dtype=np.float32)


def rgb_to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    grayscale = np.dot(image_rgb[..., :3], np.array([0.2989, 0.5870, 0.1140], dtype=np.float32))
    grayscale = np.clip(grayscale / 255.0, 0.0, 1.0)
    return grayscale.astype(np.float32)


def compute_gradient_magnitude(grayscale_image: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(grayscale_image)
    grad_magnitude = np.sqrt((grad_x ** 2) + (grad_y ** 2))
    max_value = float(np.max(grad_magnitude))
    if max_value > 0.0:
        grad_magnitude = grad_magnitude / max_value
    return grad_magnitude.astype(np.float32)


def compute_hog_descriptor(grayscale_image: np.ndarray, feature_config: Dict[str, object]) -> np.ndarray:
    descriptor = hog(
        grayscale_image,
        orientations=int(feature_config["orientations"]),
        pixels_per_cell=tuple(feature_config["pixels_per_cell"]),
        cells_per_block=tuple(feature_config["cells_per_block"]),
        block_norm=str(feature_config["block_norm"]),
        transform_sqrt=bool(feature_config["transform_sqrt"]),
        feature_vector=True,
    )
    return np.asarray(descriptor, dtype=np.float32)


def extract_feature_vector(
    image_rgb: np.ndarray,
    feature_extractor: str,
    feature_config: Dict[str, object],
) -> np.ndarray:
    grayscale = rgb_to_grayscale(image_rgb)
    if feature_extractor == "hog":
        return compute_hog_descriptor(grayscale, feature_config)
    if feature_extractor == "ghog":
        base_hog = compute_hog_descriptor(grayscale, feature_config)
        gradient_hog = compute_hog_descriptor(compute_gradient_magnitude(grayscale), feature_config)
        return np.concatenate([base_hog, gradient_hog], axis=0).astype(np.float32)
    raise ValueError("Feature extractor tidak dikenal: {}".format(feature_extractor))


def apply_augmenter_to_image(image_rgb: np.ndarray, augmenter: tf.keras.layers.Layer) -> np.ndarray:
    input_batch = tf.convert_to_tensor(image_rgb[None, ...], dtype=tf.float32)
    augmented = augmenter(input_batch, training=True)
    augmented = tf.clip_by_value(augmented, 0.0, 255.0)
    return np.asarray(augmented[0], dtype=np.float32)


def build_feature_matrix(
    filepaths: List[str],
    labels: List[int],
    image_size: int,
    feature_extractor: str,
    feature_config: Dict[str, object],
    augmentation_profile: Dict[str, object],
    augmentation_copies: int,
    seed: int,
    is_train_split: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    features: List[np.ndarray] = []
    targets: List[int] = []
    source_counts = {"original": 0, "augmented": 0}

    augmenters: List[tf.keras.layers.Layer] = []
    if is_train_split and bool(augmentation_profile.get("enabled")):
        build_fn = augmentation_profile.get("build_fn")
        if not callable(build_fn):
            raise ValueError("Profile augmentasi {} tidak memiliki build_fn.".format(augmentation_profile["name"]))
        for copy_index in range(max(1, int(augmentation_copies))):
            augmenters.append(build_fn(seed=seed + (copy_index * 97)))

    for item_index, (image_path, label) in enumerate(zip(filepaths, labels)):
        image_rgb = load_rgb_image(image_path=image_path, image_size=image_size)
        features.append(extract_feature_vector(image_rgb, feature_extractor, feature_config))
        targets.append(int(label))
        source_counts["original"] += 1

        for copy_index, augmenter in enumerate(augmenters):
            augmented_image = apply_augmenter_to_image(image_rgb, augmenter)
            feature_vector = extract_feature_vector(augmented_image, feature_extractor, feature_config)
            features.append(feature_vector)
            targets.append(int(label))
            source_counts["augmented"] += 1

    feature_matrix = np.asarray(features, dtype=np.float32)
    target_vector = np.asarray(targets, dtype=np.int32)
    return feature_matrix, target_vector, source_counts


def select_feature_names(feature_mode: str) -> List[str]:
    normalized = str(feature_mode).strip().lower()
    if normalized == "all":
        return ["hog", "ghog"]
    if normalized in {"hog", "ghog"}:
        return [normalized]
    raise ValueError("Mode feature tidak dikenal: {}".format(feature_mode))


def build_classifier(classifier_name: str, args, thread_limit: int):
    if classifier_name == "svm":
        return SVC(
            C=float(args.svm_c),
            kernel=str(args.svm_kernel),
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=args.seed,
        )
    if classifier_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(args.knn_neighbors),
            weights=str(args.knn_weights),
            metric="minkowski",
        )
    if classifier_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(args.rf_estimators),
            max_depth=None if int(args.rf_max_depth) <= 0 else int(args.rf_max_depth),
            min_samples_split=int(args.rf_min_samples_split),
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=max(1, int(thread_limit)),
        )
    if classifier_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(int(args.mlp_hidden_1), int(args.mlp_hidden_2)),
            activation="relu",
            solver="adam",
            alpha=float(args.mlp_alpha),
            batch_size=int(args.mlp_batch_size),
            learning_rate_init=float(args.mlp_learning_rate),
            max_iter=int(args.mlp_max_iter),
            early_stopping=True,
            random_state=args.seed,
        )
    raise ValueError("Classifier tidak dikenal: {}".format(classifier_name))


def predict_proba_matrix(estimator, features: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    return np.asarray(probabilities, dtype=np.float32)


def run_feature_experiment(
    classifier_name: str,
    feature_extractor: str,
    augmentation_profile_name: str,
    args,
    thread_limit: int,
) -> Dict[str, Path]:
    set_global_seed(args.seed)
    dataset_dir = (PROJECT_ROOT / args.dataset_dir).resolve()
    ensure_split_structure(dataset_dir)

    class_names = get_class_names(dataset_dir)
    class_to_index = dict((name, idx) for idx, name in enumerate(class_names))
    profile = load_profile(augmentation_profile_name)
    model_name = "{}_{}".format(classifier_name, feature_extractor)
    run_name = build_run_name(model_name=model_name, augmentation_tag=str(profile["run_tag"]))
    run_dirs = ensure_run_directories(PROJECT_ROOT, model_name, run_name)
    report_root = run_dirs["report_root"]
    models_root = run_dirs["model_root"]
    run_report_dir = run_dirs["report_dir"]
    run_model_dir = run_dirs["model_dir"]

    train_limit = pick_split_limit(args.max_per_class, args.max_train_per_class)
    test_limit = pick_split_limit(args.max_per_class, args.max_test_per_class)
    val_limit = pick_split_limit(args.max_per_class, args.max_validation_per_class)

    train_paths, train_labels, train_counts = collect_split_files(
        dataset_dir / "train",
        class_names,
        class_to_index,
        train_limit,
        args.seed,
    )
    test_paths, test_labels, test_counts = collect_split_files(
        dataset_dir / "testing",
        class_names,
        class_to_index,
        test_limit,
        args.seed + 1,
    )
    val_paths, val_labels, val_counts = collect_split_files(
        dataset_dir / "validation",
        class_names,
        class_to_index,
        val_limit,
        args.seed + 2,
    )

    feature_config = {
        "orientations": int(args.hog_orientations),
        "pixels_per_cell": (int(args.hog_pixels_per_cell), int(args.hog_pixels_per_cell)),
        "cells_per_block": (int(args.hog_cells_per_block), int(args.hog_cells_per_block)),
        "block_norm": "L2-Hys",
        "transform_sqrt": True,
    }
    augmentation_copies = 0 if not bool(profile["enabled"]) else max(1, int(args.augmentation_copies))

    x_train, y_train, train_source_counts = build_feature_matrix(
        filepaths=train_paths,
        labels=train_labels,
        image_size=int(args.image_size),
        feature_extractor=feature_extractor,
        feature_config=feature_config,
        augmentation_profile=profile,
        augmentation_copies=augmentation_copies,
        seed=args.seed,
        is_train_split=True,
    )
    x_val, y_val, _ = build_feature_matrix(
        filepaths=val_paths,
        labels=val_labels,
        image_size=int(args.image_size),
        feature_extractor=feature_extractor,
        feature_config=feature_config,
        augmentation_profile=profile,
        augmentation_copies=0,
        seed=args.seed,
        is_train_split=False,
    )
    x_test, y_test, _ = build_feature_matrix(
        filepaths=test_paths,
        labels=test_labels,
        image_size=int(args.image_size),
        feature_extractor=feature_extractor,
        feature_config=feature_config,
        augmentation_profile=profile,
        augmentation_copies=0,
        seed=args.seed,
        is_train_split=False,
    )

    print("\n=== Konfigurasi Eksperimen {} ===".format(run_name))
    print("Classifier               :", classifier_name)
    print("Feature extractor        :", feature_extractor.upper())
    print("Profile augmentasi       :", profile["display_name"])
    print("Augmentasi statis aktif  :", "ya" if profile["enabled"] else "tidak")
    print("Jumlah copy augmentasi   :", augmentation_copies)
    print("Train sample asli        :", len(train_labels))
    print("Train sample augmented   :", train_source_counts["augmented"])
    print("Train sample final       :", len(y_train))
    print("Validation sample        :", len(y_val))
    print("Testing sample           :", len(y_test))
    print("Panjang fitur            :", x_train.shape[1] if x_train.ndim == 2 else 0)
    print("Kebijakan augmentasi:")
    for description in profile["policy_lines"]:
        print("- {}".format(description))

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    estimator = build_classifier(classifier_name=classifier_name, args=args, thread_limit=thread_limit)
    estimator.fit(x_train_scaled, y_train)

    validation_outputs = evaluate_predictions(
        split_name="validation",
        y_true=y_val,
        y_pred_prob=predict_proba_matrix(estimator, x_val_scaled),
        class_names=class_names,
        output_dir=run_report_dir,
    )
    testing_outputs = evaluate_predictions(
        split_name="testing",
        y_true=y_test,
        y_pred_prob=predict_proba_matrix(estimator, x_test_scaled),
        class_names=class_names,
        output_dir=run_report_dir,
    )

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

    split_counts = {
        "train": train_counts,
        "validation": val_counts,
        "testing": test_counts,
    }
    split_df = pd.DataFrame(split_counts).transpose()
    split_df.to_csv(str(run_report_dir / "split_distribution.csv"))
    plot_split_distribution(split_counts, run_report_dir / "split_distribution.png")

    feature_summary = {
        "feature_extractor": feature_extractor,
        "feature_length": int(x_train.shape[1]),
        "image_size": int(args.image_size),
        "hog_config": feature_config,
        "train_source_counts": train_source_counts,
        "scaler": "StandardScaler",
    }
    dump_json(run_report_dir / "feature_summary.json", feature_summary)

    validation_metrics = validation_outputs["metrics"]
    testing_metrics = testing_outputs["metrics"]
    metrics_summary = {
        "run_name": run_name,
        "model_name": model_name,
        "device_used": "cpu",
        "augmentation_profile": str(profile["name"]),
        "augmentation_display_name": str(profile["display_name"]),
        "augmentation_enabled": bool(profile["enabled"]),
        "augmentation_copies": int(augmentation_copies),
        "feature_extractor": feature_extractor,
        "classifier_name": classifier_name,
        "primary_split": "testing",
        "accuracy": float(testing_metrics["accuracy"]),
        "f1_score": float(testing_metrics["f1_score"]),
        "roc_auc": float(testing_metrics["roc_auc"]),
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "validation_f1_score": float(validation_metrics["f1_score"]),
        "validation_roc_auc": float(validation_metrics["roc_auc"]),
        "test_samples": int(testing_metrics["samples"]),
        "validation_samples": int(validation_metrics["samples"]),
        "train_samples_total_before_augmentation": int(len(train_labels)),
        "train_samples_total_after_augmentation": int(len(y_train)),
        "train_samples_used_per_epoch": int(len(y_train)),
        "validation_samples_used": int(len(y_val)),
        "train_samples_total": int(len(train_labels)),
        "validation_samples_total": int(len(val_labels)),
        "test_samples_total": int(len(test_labels)),
        "split_metrics": {
            "validation": validation_metrics,
            "testing": testing_metrics,
        },
    }
    dump_json(run_report_dir / "evaluation_metrics.json", metrics_summary)
    dump_json(
        run_report_dir / "split_metrics.json",
        {
            "validation": validation_metrics,
            "testing": testing_metrics,
        },
    )
    pd.DataFrame([metrics_summary]).to_csv(str(run_report_dir / "evaluation_metrics.csv"), index=False)

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
            "feature_length": feature_summary["feature_length"],
        },
        run_report_dir / "evaluation_table.png",
    )

    bundle_payload = {
        "model": estimator,
        "scaler": scaler,
        "class_names": class_names,
        "feature_extractor": feature_extractor,
        "feature_config": feature_config,
        "image_size": int(args.image_size),
        "augmentation_profile": str(profile["name"]),
        "classifier_name": classifier_name,
    }
    best_model_path = run_model_dir / "best_model.joblib"
    final_model_path = run_model_dir / "final_model.joblib"
    joblib.dump(bundle_payload, str(best_model_path))
    joblib.dump(bundle_payload, str(final_model_path))

    with open(str(run_model_dir / "class_names.json"), "w", encoding="utf-8") as fp:
        json.dump(class_names, fp, indent=2)

    metadata = {
        "run_name": run_name,
        "model_name": model_name,
        "model_family": "classical_ml",
        "model_category": "hog_ghog",
        "classifier_name": classifier_name,
        "feature_extractor": feature_extractor,
        "augmentation_profile": str(profile["name"]),
        "augmentation_display_name": str(profile["display_name"]),
        "augmentation_enabled": bool(profile["enabled"]),
        "augmentation_copies": int(augmentation_copies),
        "dataset_dir": str(dataset_dir),
        "image_size": int(args.image_size),
        "class_names": class_names,
        "split_counts": split_counts,
        "feature_config": feature_config,
        "training_config": dict(vars(args)),
        "artifact_paths": {
            "report_dir": str(run_report_dir),
            "model_dir": str(run_model_dir),
            "best_model": str(best_model_path),
            "final_model": str(final_model_path),
        },
        "pipeline_steps": [
            "Baca split train/validation/testing dari dataset/split.",
            "Augmentasi hanya diterapkan pada data train sesuai profile yang dipilih.",
            "Ekstraksi fitur HOG atau GHOG dari train asli dan train hasil augmentasi.",
            "Scaling fitur dengan StandardScaler.",
            "Training classifier classical ML.",
            "Evaluasi validation dan testing tanpa augmentasi acak.",
        ],
    }
    write_run_metadata(run_report_dir, run_model_dir, metadata)
    write_experiment_summary(run_report_dir, metadata, metrics_summary)

    summary_lines = [
        "Run Name: {}".format(run_name),
        "Model: {}".format(model_name),
        "Model Family: classical_ml",
        "Classifier: {}".format(classifier_name),
        "Feature Extractor: {}".format(feature_extractor.upper()),
        "Augmentasi: {} ({})".format(profile["display_name"], profile["name"]),
        "Train samples sebelum augmentasi: {}".format(len(train_labels)),
        "Train samples sesudah augmentasi: {}".format(len(y_train)),
        "Validation accuracy: {:.4f}".format(metrics_summary["validation_accuracy"]),
        "Validation F1-score: {:.4f}".format(metrics_summary["validation_f1_score"]),
        "Validation ROC-AUC: {:.4f}".format(metrics_summary["validation_roc_auc"]),
        "Testing accuracy: {:.4f}".format(metrics_summary["accuracy"]),
        "Testing F1-score: {:.4f}".format(metrics_summary["f1_score"]),
        "Testing ROC-AUC: {:.4f}".format(metrics_summary["roc_auc"]),
        "",
        "Alur Eksperimen:",
        "1. Split data dibaca dari folder train, validation, dan testing.",
        "2. Augmentasi hanya diterapkan pada data train.",
        "3. Fitur diekstrak dari train asli dan train hasil augmentasi.",
        "4. Fitur dinormalisasi dengan StandardScaler.",
        "5. Classifier dilatih lalu diuji pada validation dan testing tanpa augmentasi acak.",
    ]
    with open(str(run_report_dir / "summary.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(summary_lines))

    update_latest_run_pointer(report_root, run_name)
    update_latest_run_pointer(models_root, run_name)

    print("\n=== Ringkasan Evaluasi {} ===".format(run_name))
    print("Validation accuracy  : {:.4f}".format(metrics_summary["validation_accuracy"]))
    print("Validation F1-score  : {:.4f}".format(metrics_summary["validation_f1_score"]))
    print("Validation ROC-AUC   : {:.4f}".format(metrics_summary["validation_roc_auc"]))
    print("Testing accuracy     : {:.4f}".format(metrics_summary["accuracy"]))
    print("Testing F1-score     : {:.4f}".format(metrics_summary["f1_score"]))
    print("Testing ROC-AUC      : {:.4f}".format(metrics_summary["roc_auc"]))
    print("Report               : {}".format(run_report_dir))
    print("Model                : {}".format(run_model_dir))

    return {
        "report_dir": run_report_dir,
        "model_dir": run_model_dir,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
    }


def run_feature_training_pipeline(classifier_name: str, args) -> Dict[str, Path]:
    set_global_seed(args.seed)
    thread_limit = configure_classical_cpu_runtime(
        max_cpu_usage_percent=args.max_cpu_usage_percent,
        cpu_thread_limit=args.cpu_thread_limit,
    )

    selected_features = select_feature_names(args.feature_extractor)
    selected_profiles = resolve_profile_names(args.augmentation_profile)

    last_result: Dict[str, Path] = {}
    for feature_name in selected_features:
        for profile_name in selected_profiles:
            last_result = run_feature_experiment(
                classifier_name=classifier_name,
                feature_extractor=feature_name,
                augmentation_profile_name=profile_name,
                args=args,
                thread_limit=thread_limit,
            )
    return last_result


def build_feature_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-dir", type=str, default="dataset/split", help="Folder dataset hasil split.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Ukuran gambar untuk ekstraksi fitur.")
    parser.add_argument(
        "--feature-extractor",
        type=str,
        default="all",
        choices=["hog", "ghog", "all"],
        help="Pilih feature extractor HOG, GHOG, atau jalankan keduanya.",
    )
    parser.add_argument(
        "--augmentation-profile",
        type=str,
        default="mycostum_augment",
        choices=["without_augment", "mycostum_augment", "all"],
        help="Pilih profile augmentasi: tanpa augmentasi, augmentasi kustom, atau jalankan semua profile.",
    )
    parser.add_argument(
        "--augmentation-copies",
        type=int,
        default=1,
        help="Jumlah copy augmentasi statis yang ditambahkan ke data train saat augmentasi aktif.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hog-orientations", type=int, default=DEFAULT_HOG_CONFIG["orientations"], help="Jumlah orientation bins HOG.")
    parser.add_argument("--hog-pixels-per-cell", type=int, default=DEFAULT_HOG_CONFIG["pixels_per_cell"][0], help="Ukuran pixels per cell HOG.")
    parser.add_argument("--hog-cells-per-block", type=int, default=DEFAULT_HOG_CONFIG["cells_per_block"][0], help="Jumlah cells per block HOG.")
    parser.add_argument("--max-per-class", type=int, default=None, help="Batasi jumlah data per kelas untuk semua split.")
    parser.add_argument("--max-train-per-class", type=int, default=None, help="Batasi jumlah data per kelas khusus split train.")
    parser.add_argument("--max-validation-per-class", type=int, default=None, help="Batasi jumlah data per kelas khusus split validation.")
    parser.add_argument("--max-test-per-class", type=int, default=None, help="Batasi jumlah data per kelas khusus split testing.")
    parser.add_argument("--max-cpu-usage-percent", type=int, default=70, help="Target maksimum penggunaan logical CPU dalam persen saat training.")
    parser.add_argument("--cpu-thread-limit", type=int, default=None, help="Batas absolut thread CPU (opsional, override persen).")

    parser.add_argument("--svm-c", type=float, default=2.0, help="Nilai C untuk classifier SVM.")
    parser.add_argument("--svm-kernel", type=str, default="rbf", choices=["linear", "rbf", "poly", "sigmoid"], help="Kernel SVM.")
    parser.add_argument("--knn-neighbors", type=int, default=5, help="Jumlah tetangga untuk KNN.")
    parser.add_argument("--knn-weights", type=str, default="distance", choices=["uniform", "distance"], help="Skema bobot KNN.")
    parser.add_argument("--rf-estimators", type=int, default=300, help="Jumlah pohon pada Random Forest.")
    parser.add_argument("--rf-max-depth", type=int, default=0, help="Maksimum kedalaman pohon Random Forest (0=tanpa batas).")
    parser.add_argument("--rf-min-samples-split", type=int, default=2, help="Nilai min_samples_split Random Forest.")
    parser.add_argument("--mlp-hidden-1", type=int, default=256, help="Jumlah unit hidden layer pertama MLP.")
    parser.add_argument("--mlp-hidden-2", type=int, default=128, help="Jumlah unit hidden layer kedua MLP.")
    parser.add_argument("--mlp-alpha", type=float, default=1e-4, help="Nilai regularisasi alpha MLP.")
    parser.add_argument("--mlp-batch-size", type=int, default=64, help="Batch size internal untuk MLP.")
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3, help="Learning rate awal MLP.")
    parser.add_argument("--mlp-max-iter", type=int, default=300, help="Maksimum iterasi training MLP.")
    return parser
