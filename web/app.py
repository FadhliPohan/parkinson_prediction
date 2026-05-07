import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
REPORT_ROOT = PROJECT_ROOT / "report"
TRAINED_MODELS_ROOT = PROJECT_ROOT / "trained_models"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

MODEL_DIR = PROJECT_ROOT / "model"
HOG_GHOG_DIR = MODEL_DIR / "hog_ghog"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(HOG_GHOG_DIR) not in sys.path:
    sys.path.insert(0, str(HOG_GHOG_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from transformer_backbones import TRANSFORMER_CUSTOM_OBJECTS
except Exception:
    TRANSFORMER_CUSTOM_OBJECTS = {}

try:
    from augmentation_selected.registry import list_available_profiles
except Exception:
    list_available_profiles = None

try:
    from common import extract_feature_vector, load_rgb_image_from_bytes
except Exception:
    extract_feature_vector = None
    load_rgb_image_from_bytes = None


def preprocess_transformer_input(inputs: np.ndarray) -> np.ndarray:
    return (inputs.astype(np.float32) / 127.5) - 1.0


MODEL_PREPROCESSORS = {
    "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "vgg19": tf.keras.applications.vgg19.preprocess_input,
    "resnet152": tf.keras.applications.resnet.preprocess_input,
    "inception_googlenet": tf.keras.applications.inception_v3.preprocess_input,
    "efficientnet": tf.keras.applications.efficientnet.preprocess_input,
    "densenet121": tf.keras.applications.densenet.preprocess_input,
    "vit": preprocess_transformer_input,
    "swintransformer": preprocess_transformer_input,
    "deit": preprocess_transformer_input,
}


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


def count_class_distribution(folder: Path) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    if not folder.exists() or not folder.is_dir():
        return distribution
    for class_dir in sorted([p for p in folder.iterdir() if p.is_dir()]):
        distribution[class_dir.name] = len(list_image_files(class_dir))
    return distribution


def summarize_split_distribution(split_root: Path) -> Dict[str, Dict[str, int]]:
    split_names = ["train", "testing", "validation"]
    split_summary: Dict[str, Dict[str, int]] = {}

    for split_name in split_names:
        split_dir = split_root / split_name
        split_summary[split_name] = count_class_distribution(split_dir)
    return split_summary


def get_model_runs(base_dir: Path) -> List[str]:
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    run_dirs = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in run_dirs]


def read_latest_run_id(base_dir: Path) -> Optional[str]:
    latest_file = base_dir / "latest_run.txt"
    if latest_file.exists():
        run_id = latest_file.read_text(encoding="utf-8").strip()
        if run_id and (base_dir / run_id).exists():
            return run_id
    runs = get_model_runs(base_dir)
    if runs:
        return runs[0]
    return None


def resolve_model_file(run_dir: Path) -> Optional[Path]:
    ordered_candidates = [
        run_dir / "best_model.keras",
        run_dir / "best_model.pt",
        run_dir / "best_model.joblib",
        run_dir / "final_model.keras",
        run_dir / "final_model.pt",
        run_dir / "final_model.joblib",
    ]
    for candidate in ordered_candidates:
        if candidate.exists():
            return candidate

    generic_candidates = []
    generic_candidates.extend(sorted(run_dir.glob("*.keras")))
    generic_candidates.extend(sorted(run_dir.glob("*.pt")))
    generic_candidates.extend(sorted(run_dir.glob("*.joblib")))
    if generic_candidates:
        return generic_candidates[0]
    return None


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(str(path), "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def read_report_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(str(path))
    except Exception:
        return pd.DataFrame()


def safe_float(value) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    if np.isnan(result):
        return None
    return result


def format_metric(value) -> str:
    parsed = safe_float(value)
    if parsed is None:
        return "NaN"
    return "{:.4f}".format(parsed)


def build_overview_table(distribution: Dict[str, int], source_name: str) -> pd.DataFrame:
    rows = []
    total = 0
    for class_name in sorted(distribution.keys()):
        count = int(distribution[class_name])
        total += count
        rows.append({"Source": source_name, "Class": class_name, "Count": count})
    rows.append({"Source": source_name, "Class": "Total", "Count": total})
    return pd.DataFrame(rows)


def build_split_table(split_distribution: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    split_df = pd.DataFrame(split_distribution).T
    if split_df.empty:
        return split_df
    split_df["total"] = split_df.sum(axis=1)
    return split_df


def build_model_label(row: Dict[str, object]) -> str:
    model_family = str(row.get("model_family", "legacy"))
    classifier_name = row.get("classifier_name")
    feature_extractor = row.get("feature_extractor")
    if model_family == "classical_ml" and classifier_name and feature_extractor:
        return "{} + {}".format(str(classifier_name).upper(), str(feature_extractor).upper())
    return str(row.get("model_name", "unknown"))


def derive_legacy_metadata(model_name: str, run_id: str) -> Dict[str, object]:
    return {
        "run_name": run_id,
        "model_name": model_name,
        "model_family": "legacy_or_unknown",
        "classifier_name": model_name,
        "feature_extractor": None,
        "augmentation_profile": "legacy_unknown",
        "augmentation_display_name": "Legacy / Unknown",
    }


def read_run_bundle(model_name: str, run_report_dir: Path) -> Dict[str, object]:
    metadata = load_json(run_report_dir / "run_metadata.json") or derive_legacy_metadata(model_name, run_report_dir.name)
    summary = load_json(run_report_dir / "experiment_summary.json") or {}
    metrics = load_json(run_report_dir / "evaluation_metrics.json") or {}

    if summary.get("metrics") and not metrics:
        metrics = summary.get("metrics", {})

    row = {
        "model_name": metadata.get("model_name", model_name),
        "run_name": metadata.get("run_name", run_report_dir.name),
        "run_id": run_report_dir.name,
        "model_family": metadata.get("model_family", "legacy_or_unknown"),
        "classifier_name": metadata.get("classifier_name"),
        "feature_extractor": metadata.get("feature_extractor"),
        "augmentation_profile": metadata.get("augmentation_profile", "legacy_unknown"),
        "augmentation_display_name": metadata.get("augmentation_display_name", "Legacy / Unknown"),
        "accuracy": metrics.get("accuracy"),
        "f1_score": metrics.get("f1_score"),
        "roc_auc": metrics.get("roc_auc"),
        "validation_accuracy": metrics.get("validation_accuracy"),
        "validation_f1_score": metrics.get("validation_f1_score"),
        "validation_roc_auc": metrics.get("validation_roc_auc"),
        "train_samples_total_before_augmentation": metrics.get("train_samples_total_before_augmentation"),
        "train_samples_total_after_augmentation": metrics.get("train_samples_total_after_augmentation"),
        "test_samples": metrics.get("test_samples"),
        "modified_at": run_report_dir.stat().st_mtime,
        "model_label": build_model_label(metadata),
    }
    return {
        "metadata": metadata,
        "summary": summary,
        "metrics": metrics,
        "row": row,
    }


def collect_run_rows() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not REPORT_ROOT.exists():
        return pd.DataFrame()

    for model_dir in sorted([p for p in REPORT_ROOT.iterdir() if p.is_dir()]):
        for run_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
            bundle = read_run_bundle(model_dir.name, run_dir)
            rows.append(bundle["row"])

    if not rows:
        return pd.DataFrame()

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.sort_values(by="modified_at", ascending=False)
    return dataframe


def collect_latest_run_rows() -> pd.DataFrame:
    dataframe = collect_run_rows()
    if dataframe.empty:
        return dataframe
    dedup = dataframe.sort_values(by="modified_at", ascending=False)
    dedup = dedup.drop_duplicates(subset=["model_name", "augmentation_profile"], keep="first")
    return dedup


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_name: str, run_id: str, model_file: str) -> Dict[str, object]:
    run_dir = TRAINED_MODELS_ROOT / model_name / run_id
    class_names_path = run_dir / "class_names.json"
    class_names = load_json(class_names_path)
    framework = "tensorflow"

    if model_file.lower().endswith(".pt"):
        if YOLO is None:
            raise ImportError("ultralytics belum terinstall, model YOLOv8 tidak bisa dimuat.")
        model = YOLO(model_file)
        framework = "yolo"

        if not isinstance(class_names, list) or not class_names:
            yolo_names = getattr(model, "names", None)
            if isinstance(yolo_names, dict):
                def _safe_key(item):
                    key = item[0]
                    try:
                        return (0, int(key))
                    except Exception:
                        return (1, str(key))

                class_names = [name for _, name in sorted(yolo_names.items(), key=_safe_key)]
            elif isinstance(yolo_names, list):
                class_names = yolo_names
            else:
                class_names = []
        return {"model": model, "class_names": class_names, "framework": framework}

    if model_file.lower().endswith(".joblib"):
        bundle = joblib.load(model_file)
        model = bundle.get("model")
        scaler = bundle.get("scaler")
        class_names = bundle.get("class_names", class_names)
        if not isinstance(class_names, list) or not class_names:
            class_names = ["class_0", "class_1"]
        return {
            "model": model,
            "scaler": scaler,
            "class_names": class_names,
            "framework": "classical",
            "feature_extractor": bundle.get("feature_extractor"),
            "feature_config": bundle.get("feature_config"),
            "image_size": bundle.get("image_size", 224),
            "classifier_name": bundle.get("classifier_name"),
        }

    model = tf.keras.models.load_model(
        model_file,
        custom_objects=TRANSFORMER_CUSTOM_OBJECTS,
        compile=False,
    )
    if not isinstance(class_names, list) or not class_names:
        class_names = ["class_0", "class_1"]

    return {"model": model, "class_names": class_names, "framework": framework}


def prepare_input_image(
    image_bytes: bytes, target_size: Tuple[int, int], model_name: str
) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)

    preprocess_fn = MODEL_PREPROCESSORS.get(model_name)
    if preprocess_fn is not None:
        batch = preprocess_fn(batch)
    return batch


def prepare_yolo_input_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def predict_with_model(
    model_name: str,
    run_id: str,
    image_bytes: bytes,
) -> Dict[str, object]:
    run_dir = TRAINED_MODELS_ROOT / model_name / run_id
    model_file = resolve_model_file(run_dir)
    if model_file is None:
        raise FileNotFoundError("File model tidak ditemukan untuk {} / {}.".format(model_name, run_id))

    bundle = load_model_bundle(model_name, run_id, str(model_file))
    model = bundle["model"]
    class_names = bundle["class_names"]
    framework = str(bundle.get("framework", "tensorflow"))

    if framework == "yolo":
        input_image = prepare_yolo_input_image(image_bytes)
        prediction = model.predict(source=input_image, verbose=False)
        if not prediction:
            raise RuntimeError("Prediksi YOLOv8 tidak menghasilkan output.")
        probs_tensor = prediction[0].probs
        if probs_tensor is None:
            raise RuntimeError("Output prediksi YOLOv8 tidak memiliki probabilitas kelas.")
        probs_raw = np.asarray(probs_tensor.data.cpu(), dtype=np.float32).reshape(-1)
    elif framework == "classical":
        if extract_feature_vector is None or load_rgb_image_from_bytes is None:
            raise ImportError("Utility HOG/GHOG belum siap dipakai di dashboard.")
        image_size = int(bundle.get("image_size", 224))
        feature_extractor = str(bundle.get("feature_extractor", "hog"))
        feature_config = bundle.get("feature_config") or {}
        scaler = bundle.get("scaler")
        rgb_image = load_rgb_image_from_bytes(image_bytes, image_size)
        feature_vector = extract_feature_vector(rgb_image, feature_extractor, feature_config).reshape(1, -1)
        if scaler is not None:
            feature_vector = scaler.transform(feature_vector)
        prediction = model.predict_proba(feature_vector)
        probs_raw = np.asarray(prediction, dtype=np.float32).reshape(-1)
    else:
        input_shape = model.input_shape
        target_height = int(input_shape[1]) if len(input_shape) > 2 and input_shape[1] else 224
        target_width = int(input_shape[2]) if len(input_shape) > 2 and input_shape[2] else 224

        input_batch = prepare_input_image(
            image_bytes=image_bytes,
            target_size=(target_width, target_height),
            model_name=model_name,
        )
        prediction = model.predict(input_batch, verbose=0)
        probs_raw = np.asarray(prediction).reshape(-1)

    if probs_raw.size == 1:
        positive_prob = float(probs_raw[0])
        positive_prob = max(0.0, min(1.0, positive_prob))
        class_names_effective = class_names[:2] if len(class_names) >= 2 else ["class_0", "class_1"]
        probabilities = [1.0 - positive_prob, positive_prob]
    else:
        class_names_effective = class_names
        probabilities = probs_raw.astype(float).tolist()
        prob_sum = float(sum(probabilities))
        if prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]

    if len(class_names_effective) != len(probabilities):
        class_names_effective = ["class_{}".format(i) for i in range(len(probabilities))]

    pred_index = int(np.argmax(probabilities))
    pred_label = class_names_effective[pred_index]
    pred_confidence = float(probabilities[pred_index])

    return {
        "model_name": model_name,
        "run_id": run_id,
        "predicted_label": pred_label,
        "confidence": pred_confidence,
        "class_names": class_names_effective,
        "probabilities": probabilities,
    }


def render_dataset_tab() -> None:
    st.subheader("Ringkasan Dataset dan Mode Augmentasi")

    original_dist = count_class_distribution(DATASET_ROOT / "original")
    split_dist = summarize_split_distribution(DATASET_ROOT / "split")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Dataset Original**")
        if original_dist:
            st.dataframe(build_overview_table(original_dist, "original"), use_container_width=True)
            st.bar_chart(pd.Series(original_dist, name="count"))
        else:
            st.info("Folder `dataset/original` belum ditemukan atau masih kosong.")

    with col_right:
        st.markdown("**Dataset Split (train/testing/validation)**")
        split_df = build_split_table(split_dist)
        if not split_df.empty:
            st.dataframe(split_df, use_container_width=True)
            st.bar_chart(split_df.drop(columns=["total"], errors="ignore"))
        else:
            st.info("Folder `dataset/split` belum ditemukan atau masih kosong.")

    st.markdown("**Mode Augmentasi Eksperimen**")
    if callable(list_available_profiles):
        profile_rows = []
        for profile in list_available_profiles():
            profile_rows.append(
                {
                    "profile": profile["name"],
                    "label": profile["display_name"],
                    "run_tag": profile["run_tag"],
                    "augmentasi_aktif": "ya" if profile["enabled"] else "tidak",
                    "ringkasan": " | ".join(profile["policy_lines"][:3]),
                }
            )
        if profile_rows:
            st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)
    st.info(
        "Eksperimen komparasi terbaru memakai augmentasi statis hanya pada split train. Validation dan testing selalu dievaluasi tanpa augmentasi acak."
    )


def render_report_tab() -> None:
    st.subheader("Report Training dan Komparasi Eksperimen")
    latest_df = collect_latest_run_rows()
    all_runs_df = collect_run_rows()
    if latest_df.empty:
        st.info("Belum ada report di folder `report/`.")
        return

    st.markdown("**Ringkasan Run Terbaru per Model dan Mode Augmentasi**")
    latest_table = latest_df[
        [
            "model_label",
            "model_family",
            "augmentation_display_name",
            "validation_accuracy",
            "accuracy",
            "f1_score",
            "roc_auc",
            "run_name",
        ]
    ].rename(
        columns={
            "model_label": "model",
            "model_family": "family",
            "augmentation_display_name": "augmentasi",
            "validation_accuracy": "val_accuracy",
            "accuracy": "test_accuracy",
            "f1_score": "test_f1",
            "roc_auc": "test_roc_auc",
        }
    )
    st.dataframe(latest_table, use_container_width=True)

    family_options = ["Semua"] + sorted([str(value) for value in latest_df["model_family"].dropna().unique()])
    selected_family = st.selectbox("Filter family model", family_options, index=0)

    filtered_df = all_runs_df.copy()
    if selected_family != "Semua":
        filtered_df = filtered_df[filtered_df["model_family"] == selected_family]

    if filtered_df.empty:
        st.warning("Tidak ada run yang cocok dengan filter ini.")
        return

    st.markdown("**Matriks Komparasi Accuracy (run terbaru per model + augmentasi)**")
    comparison_df = filtered_df.sort_values(by="modified_at", ascending=False)
    comparison_df = comparison_df.drop_duplicates(subset=["model_name", "augmentation_profile"], keep="first")
    accuracy_pivot = comparison_df.pivot_table(
        index="model_label",
        columns="augmentation_display_name",
        values="accuracy",
        aggfunc="first",
    )
    if not accuracy_pivot.empty:
        st.dataframe(accuracy_pivot, use_container_width=True)

    st.markdown("**Matriks Komparasi F1-score (run terbaru per model + augmentasi)**")
    f1_pivot = comparison_df.pivot_table(
        index="model_label",
        columns="augmentation_display_name",
        values="f1_score",
        aggfunc="first",
    )
    if not f1_pivot.empty:
        st.dataframe(f1_pivot, use_container_width=True)

    model_options = sorted(filtered_df["model_name"].dropna().unique().tolist())
    selected_model = st.selectbox("Pilih model untuk detail report", model_options, index=0)
    model_report_dir = REPORT_ROOT / selected_model

    run_options = get_model_runs(model_report_dir)
    if not run_options:
        st.warning("Model ini belum memiliki run report.")
        return

    default_run = read_latest_run_id(model_report_dir)
    default_index = run_options.index(default_run) if default_run in run_options else 0
    selected_run = st.selectbox("Pilih run report", run_options, index=default_index)
    run_report_dir = model_report_dir / selected_run
    bundle = read_run_bundle(selected_model, run_report_dir)
    metadata = bundle["metadata"]
    metrics = bundle["metrics"]

    top_cols = st.columns(4)
    top_cols[0].metric("Test Accuracy", format_metric(metrics.get("accuracy")))
    top_cols[1].metric("Test F1-score", format_metric(metrics.get("f1_score")))
    top_cols[2].metric("Val Accuracy", format_metric(metrics.get("validation_accuracy")))
    top_cols[3].metric("Test ROC-AUC", format_metric(metrics.get("roc_auc")))

    detail_cols = st.columns(4)
    detail_cols[0].metric("Family", str(metadata.get("model_family", "legacy_or_unknown")))
    detail_cols[1].metric("Augmentasi", str(metadata.get("augmentation_display_name", "Legacy / Unknown")))
    detail_cols[2].metric(
        "Train Sebelum Aug",
        str(metrics.get("train_samples_total_before_augmentation", "-")),
    )
    detail_cols[3].metric(
        "Train Sesudah Aug",
        str(metrics.get("train_samples_total_after_augmentation", "-")),
    )

    st.markdown("**Metadata Eksperimen**")
    st.json(
        {
            "run_name": metadata.get("run_name", selected_run),
            "model_name": metadata.get("model_name", selected_model),
            "model_family": metadata.get("model_family", "legacy_or_unknown"),
            "classifier_name": metadata.get("classifier_name"),
            "feature_extractor": metadata.get("feature_extractor"),
            "augmentation_profile": metadata.get("augmentation_profile"),
            "augmentation_display_name": metadata.get("augmentation_display_name"),
            "pipeline_steps": metadata.get("pipeline_steps"),
        },
        expanded=False,
    )

    summary_path = run_report_dir / "summary.txt"
    if summary_path.exists():
        st.markdown("**Summary**")
        st.code(summary_path.read_text(encoding="utf-8"))

    split_metrics = load_json(run_report_dir / "split_metrics.json") or {}
    if split_metrics:
        st.markdown("**Split Metrics**")
        st.dataframe(pd.DataFrame(split_metrics).T, use_container_width=True)

    feature_summary = load_json(run_report_dir / "feature_summary.json") or {}
    if feature_summary:
        st.markdown("**Ringkasan Fitur**")
        st.json(feature_summary, expanded=False)

    history_df = read_report_csv(run_report_dir / "training_history.csv")
    if not history_df.empty:
        st.markdown("**Kurva Training (CSV)**")
        cols_to_show = [c for c in ["loss", "val_loss", "accuracy", "val_accuracy"] if c in history_df.columns]
        if cols_to_show:
            st.line_chart(history_df[cols_to_show])
        st.dataframe(history_df, use_container_width=True)

    validation_classification_df = read_report_csv(run_report_dir / "classification_report_validation.csv")
    testing_classification_df = read_report_csv(run_report_dir / "classification_report_testing.csv")
    if not validation_classification_df.empty:
        st.markdown("**Classification Report Validation**")
        st.dataframe(validation_classification_df, use_container_width=True)
    if not testing_classification_df.empty:
        st.markdown("**Classification Report Testing**")
        st.dataframe(testing_classification_df, use_container_width=True)

    split_df = read_report_csv(run_report_dir / "split_distribution.csv")
    if not split_df.empty:
        st.markdown("**Split Distribution (dari report run ini)**")
        st.dataframe(split_df, use_container_width=True)

    image_files = [
        run_report_dir / "training_curves.png",
        run_report_dir / "confusion_matrix_validation.png",
        run_report_dir / "confusion_matrix_testing.png",
        run_report_dir / "roc_curve_validation.png",
        run_report_dir / "roc_curve_testing.png",
        run_report_dir / "split_distribution.png",
        run_report_dir / "evaluation_table.png",
    ]
    existing_images = [img for img in image_files if img.exists()]
    if existing_images:
        st.markdown("**Visualisasi Report**")
        for img_path in existing_images:
            st.image(str(img_path), caption=img_path.name, use_container_width=True)


def render_prediction_tab() -> None:
    st.subheader("Prediksi Gambar dan Perbandingan Antar Model")

    model_options = (
        sorted([p.name for p in TRAINED_MODELS_ROOT.iterdir() if p.is_dir()])
        if TRAINED_MODELS_ROOT.exists()
        else []
    )
    if not model_options:
        st.info("Belum ada model tersimpan di folder `trained_models/`.")
        return

    selected_models = st.multiselect(
        "Pilih model untuk dibandingkan",
        options=model_options,
        default=model_options[: min(6, len(model_options))],
    )
    if not selected_models:
        st.warning("Pilih minimal 1 model.")
        return

    selected_runs: Dict[str, str] = {}
    st.markdown("**Pilih run model**")
    for model_name in selected_models:
        runs = get_model_runs(TRAINED_MODELS_ROOT / model_name)
        if not runs:
            continue
        latest_run = read_latest_run_id(TRAINED_MODELS_ROOT / model_name)
        default_index = runs.index(latest_run) if latest_run in runs else 0
        selected_runs[model_name] = st.selectbox(
            "Run untuk {}".format(model_name),
            options=runs,
            index=default_index,
            key="run_{}".format(model_name),
        )

    uploaded_file = st.file_uploader(
        "Upload gambar (PNG/JPG/JPEG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload gambar untuk mulai prediksi.")
        return

    image_bytes = uploaded_file.getvalue()
    image_preview = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.image(image_preview, caption="Gambar Input", width=320)

    if st.button("Jalankan Prediksi"):
        result_rows = []
        prob_rows = []
        errors = []

        for model_name in selected_models:
            run_id = selected_runs.get(model_name)
            if not run_id:
                errors.append("Model {} tidak punya run yang bisa dipilih.".format(model_name))
                continue

            try:
                prediction = predict_with_model(
                    model_name=model_name,
                    run_id=run_id,
                    image_bytes=image_bytes,
                )
                result_rows.append(
                    {
                        "model": prediction["model_name"],
                        "run_id": prediction["run_id"],
                        "predicted_label": prediction["predicted_label"],
                        "confidence": round(float(prediction["confidence"]) * 100.0, 2),
                    }
                )

                for class_name, prob in zip(prediction["class_names"], prediction["probabilities"]):
                    prob_rows.append(
                        {
                            "model": prediction["model_name"],
                            "class": class_name,
                            "probability": float(prob),
                        }
                    )
            except Exception as exc:
                errors.append("{}: {}".format(model_name, exc))

        if result_rows:
            result_df = pd.DataFrame(result_rows).sort_values(by="confidence", ascending=False)
            st.markdown("**Hasil Prediksi per Model**")
            st.dataframe(result_df, use_container_width=True)

        if prob_rows:
            prob_df = pd.DataFrame(prob_rows)
            st.markdown("**Perbandingan Probabilitas Antar Model**")
            pivot_df = prob_df.pivot(index="class", columns="model", values="probability").fillna(0.0)
            st.bar_chart(pivot_df)
            st.dataframe(prob_df, use_container_width=True)

        if errors:
            st.error("Sebagian model gagal dipakai:\n- " + "\n- ".join(errors))


def main() -> None:
    st.set_page_config(
        page_title="Parkinson Classification Dashboard",
        page_icon="P",
        layout="wide",
    )
    st.title("Parkinson Classification Dashboard")
    st.caption("Monitoring dataset, report training, komparasi eksperimen, dan prediksi gambar lintas model.")

    tab_dataset, tab_report, tab_predict = st.tabs(
        ["Dataset & Praprosesing", "Training Report", "Prediksi Model"]
    )

    with tab_dataset:
        render_dataset_tab()
    with tab_report:
        render_report_tab()
    with tab_predict:
        render_prediction_tab()


if __name__ == "__main__":
    main()
