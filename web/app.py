import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
REPORT_ROOT = PROJECT_ROOT / "report"
TRAINED_MODELS_ROOT = PROJECT_ROOT / "trained_models"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

MODEL_PREPROCESSORS = {
    "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
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
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir()], reverse=True)


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
    best_path = run_dir / "best_model.keras"
    final_path = run_dir / "final_model.keras"

    if best_path.exists():
        return best_path
    if final_path.exists():
        return final_path

    keras_files = sorted(run_dir.glob("*.keras"))
    if keras_files:
        return keras_files[0]
    return None


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(str(path), "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_name: str, run_id: str, model_file: str) -> Dict[str, object]:
    run_dir = TRAINED_MODELS_ROOT / model_name / run_id
    class_names_path = run_dir / "class_names.json"
    class_names = load_json(class_names_path)
    if not isinstance(class_names, list) or not class_names:
        class_names = ["class_0", "class_1"]

    model = tf.keras.models.load_model(model_file)
    return {"model": model, "class_names": class_names}


def build_overview_table(distribution: Dict[str, int], source_name: str) -> pd.DataFrame:
    rows = []
    total = 0
    for class_name in sorted(distribution.keys()):
        count = int(distribution[class_name])
        total += count
        rows.append({"Source": source_name, "Class": class_name, "Count": count})
    rows.append({"Source": source_name, "Class": "Total", "Count": total})
    return pd.DataFrame(rows)


def summarize_reports_latest() -> pd.DataFrame:
    rows = []
    if not REPORT_ROOT.exists():
        return pd.DataFrame()

    for model_dir in sorted([p for p in REPORT_ROOT.iterdir() if p.is_dir()]):
        run_id = read_latest_run_id(model_dir)
        if not run_id:
            continue
        metrics_path = model_dir / run_id / "evaluation_metrics.json"
        metrics = load_json(metrics_path) or {}
        rows.append(
            {
                "model": model_dir.name,
                "run_id": run_id,
                "accuracy": metrics.get("accuracy"),
                "f1_score": metrics.get("f1_score"),
                "roc_auc": metrics.get("roc_auc"),
                "test_samples": metrics.get("test_samples"),
            }
        )
    return pd.DataFrame(rows)


def read_report_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(str(path))
    except Exception:
        return pd.DataFrame()


def build_split_table(split_distribution: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    split_df = pd.DataFrame(split_distribution).T
    if split_df.empty:
        return split_df
    split_df["total"] = split_df.sum(axis=1)
    return split_df


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

    input_shape = model.input_shape
    target_height = int(input_shape[1]) if len(input_shape) > 2 and input_shape[1] else 227
    target_width = int(input_shape[2]) if len(input_shape) > 2 and input_shape[2] else 227

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
    st.subheader("Ringkasan Dataset")

    original_dist = count_class_distribution(DATASET_ROOT / "original")
    preprocess_dist = count_class_distribution(DATASET_ROOT / "praprosesing")
    split_dist = summarize_split_distribution(DATASET_ROOT / "split")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset Original**")
        if original_dist:
            st.dataframe(build_overview_table(original_dist, "original"), use_container_width=True)
            st.bar_chart(pd.Series(original_dist, name="count"))
        else:
            st.info("Folder `dataset/original` belum ditemukan / masih kosong.")

    with col2:
        st.markdown("**Dataset Praprosesing**")
        if preprocess_dist:
            st.dataframe(build_overview_table(preprocess_dist, "praprosesing"), use_container_width=True)
            st.bar_chart(pd.Series(preprocess_dist, name="count"))
        else:
            st.info("Folder `dataset/praprosesing` belum ditemukan / masih kosong.")

    st.markdown("**Dataset Split (train/testing/validation)**")
    split_df = build_split_table(split_dist)
    if not split_df.empty:
        st.dataframe(split_df, use_container_width=True)
        st.bar_chart(split_df.drop(columns=["total"], errors="ignore"))
    else:
        st.info("Folder `dataset/split` belum ditemukan / masih kosong.")

    if original_dist and preprocess_dist:
        ratio_rows = []
        for class_name in sorted(set(list(original_dist.keys()) + list(preprocess_dist.keys()))):
            original_count = int(original_dist.get(class_name, 0))
            preprocess_count = int(preprocess_dist.get(class_name, 0))
            multiplier = (float(preprocess_count) / float(original_count)) if original_count else np.nan
            ratio_rows.append(
                {
                    "class": class_name,
                    "original": original_count,
                    "praprosesing": preprocess_count,
                    "multiplier": multiplier,
                }
            )
        st.markdown("**Rasio Praprosesing terhadap Original**")
        st.dataframe(pd.DataFrame(ratio_rows), use_container_width=True)


def render_report_tab() -> None:
    st.subheader("Report Training")
    latest_df = summarize_reports_latest()
    if latest_df.empty:
        st.info("Belum ada report di folder `report/`.")
        return

    st.markdown("**Ringkasan Run Terbaru per Model**")
    st.dataframe(latest_df, use_container_width=True)

    model_options = sorted([p.name for p in REPORT_ROOT.iterdir() if p.is_dir()])
    selected_model = st.selectbox("Pilih model", model_options, index=0)
    model_report_dir = REPORT_ROOT / selected_model

    run_options = get_model_runs(model_report_dir)
    if not run_options:
        st.warning("Model ini belum memiliki run report.")
        return

    default_run = read_latest_run_id(model_report_dir)
    default_index = run_options.index(default_run) if default_run in run_options else 0
    selected_run = st.selectbox("Pilih run report", run_options, index=default_index)
    run_report_dir = model_report_dir / selected_run

    metrics = load_json(run_report_dir / "evaluation_metrics.json") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "{:.4f}".format(float(metrics.get("accuracy", 0.0))))
    c2.metric("F1-score", "{:.4f}".format(float(metrics.get("f1_score", 0.0))))
    roc_auc = metrics.get("roc_auc")
    if roc_auc is None or (isinstance(roc_auc, float) and np.isnan(roc_auc)):
        c3.metric("ROC-AUC", "NaN")
    else:
        c3.metric("ROC-AUC", "{:.4f}".format(float(roc_auc)))

    summary_path = run_report_dir / "summary.txt"
    if summary_path.exists():
        st.markdown("**Summary**")
        st.code(summary_path.read_text(encoding="utf-8"))

    history_df = read_report_csv(run_report_dir / "training_history.csv")
    if not history_df.empty:
        st.markdown("**Kurva Training (CSV)**")
        cols_to_show = [c for c in ["loss", "val_loss", "accuracy", "val_accuracy"] if c in history_df.columns]
        if cols_to_show:
            st.line_chart(history_df[cols_to_show])
        st.dataframe(history_df, use_container_width=True)

    classification_df = read_report_csv(run_report_dir / "classification_report.csv")
    if not classification_df.empty:
        st.markdown("**Classification Report**")
        st.dataframe(classification_df, use_container_width=True)

    split_df = read_report_csv(run_report_dir / "split_distribution.csv")
    if not split_df.empty:
        st.markdown("**Split Distribution (dari report run ini)**")
        st.dataframe(split_df, use_container_width=True)

    image_files = [
        run_report_dir / "training_curves.png",
        run_report_dir / "confusion_matrix.png",
        run_report_dir / "roc_curve.png",
        run_report_dir / "split_distribution.png",
        run_report_dir / "evaluation_table.png",
    ]
    existing_images = [img for img in image_files if img.exists()]
    if existing_images:
        st.markdown("**Visualisasi Report**")
        for img_path in existing_images:
            st.image(str(img_path), caption=img_path.name, use_column_width=True)


def render_prediction_tab() -> None:
    st.subheader("Prediksi Gambar dan Perbandingan Antar Model")

    model_options = sorted([p.name for p in TRAINED_MODELS_ROOT.iterdir() if p.is_dir()]) if TRAINED_MODELS_ROOT.exists() else []
    if not model_options:
        st.info("Belum ada model tersimpan di folder `trained_models/`.")
        return

    selected_models = st.multiselect(
        "Pilih model untuk dibandingkan",
        options=model_options,
        default=model_options,
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
        page_icon="🧠",
        layout="wide",
    )
    st.title("Parkinson Classification Dashboard")
    st.caption("Monitoring dataset, report training, dan prediksi gambar lintas model.")

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
