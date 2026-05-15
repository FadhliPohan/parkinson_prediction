from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.registry import DatasetRegistry
from src.datasets.validator import discover_class_directories, list_image_files
from src.inference.model_loader import load_model_bundle
from src.inference.predictor import predict_from_bundle
from src.models.registry import ModelRegistry
from src.reporting.report_reader import (
    build_latest_summary_table,
    list_datasets,
    list_models,
    list_runs,
    read_latest_run_id,
    safe_load_json,
)
from src.reporting.schemas import VISUAL_FILES
from src.utils.paths import REPORT_ROOT, TRAINED_MODELS_ROOT


def _count_split_distribution(split_root: Path) -> Dict[str, Dict[str, int]]:
    distribution: Dict[str, Dict[str, int]] = {}
    for split_name in ["train", "testing", "validation"]:
        class_map: Dict[str, int] = {}
        split_dir = split_root / split_name
        if split_dir.exists() and split_dir.is_dir():
            for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
                class_map[class_dir.name] = len(list_image_files(class_dir, [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]))
        distribution[split_name] = class_map
    return distribution


def _build_split_table(split_distribution: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    split_df = pd.DataFrame(split_distribution).T
    if split_df.empty:
        return split_df
    split_df["total"] = split_df.sum(axis=1)
    return split_df


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def render_dataset_tab(dataset_registry: DatasetRegistry) -> None:
    st.subheader("Ringkasan Dataset")

    dataset_ids = dataset_registry.list_dataset_ids()
    default_index = dataset_ids.index(dataset_registry.default_dataset)
    selected_dataset = st.selectbox("Pilih dataset", dataset_ids, index=default_index, key="dataset_tab_dataset")
    dataset_cfg = dataset_registry.get(selected_dataset)

    st.caption(f"Original: `{dataset_cfg.original_path}`")
    st.caption(f"Split: `{dataset_cfg.split_path}`")

    try:
        classes = discover_class_directories(
            dataset_root=dataset_cfg.original_path,
            class_mode=dataset_cfg.class_mode,
            extensions=dataset_cfg.valid_extensions,
        )
        original_df = pd.DataFrame(
            {
                "class_name": [entry.class_name for entry in classes],
                "source_rel_path": [entry.relative_path for entry in classes],
                "count": [entry.image_count for entry in classes],
            }
        )
    except Exception as exc:
        st.error(f"Gagal baca dataset original: {exc}")
        original_df = pd.DataFrame()

    st.markdown("**Dataset Original**")
    if not original_df.empty:
        st.dataframe(original_df, use_container_width=True)
        st.bar_chart(original_df.set_index("class_name")["count"])
    else:
        st.info("Dataset original belum siap / belum terbaca.")

    st.markdown("**Dataset Split (train/testing/validation)**")
    split_distribution = _count_split_distribution(dataset_cfg.split_path)
    split_df = _build_split_table(split_distribution)
    if not split_df.empty:
        st.dataframe(split_df, use_container_width=True)
        st.bar_chart(split_df.drop(columns=["total"], errors="ignore"))
    else:
        st.info("Dataset split belum tersedia.")

    st.info("Augmentasi training berjalan on-the-fly pada split `train`.")


def render_report_tab() -> None:
    st.subheader("Report Training")

    rows = build_latest_summary_table(REPORT_ROOT)
    latest_df = pd.DataFrame(rows)
    if latest_df.empty:
        st.info("Belum ada report pada struktur `report/<dataset>/<model>/<run_id>/`.")
        return

    st.markdown("**Ringkasan Run Terbaru per Dataset/Model**")
    st.dataframe(latest_df, use_container_width=True)

    dataset_options = list_datasets(REPORT_ROOT)
    selected_dataset = st.selectbox("Pilih dataset report", dataset_options, index=0, key="report_dataset")

    model_options = list_models(REPORT_ROOT, selected_dataset)
    if not model_options:
        st.warning("Belum ada model report untuk dataset ini.")
        return

    selected_model = st.selectbox("Pilih model", model_options, index=0, key="report_model")
    model_root = REPORT_ROOT / selected_dataset / selected_model

    run_options = list_runs(REPORT_ROOT, selected_dataset, selected_model)
    if not run_options:
        st.warning("Belum ada run report untuk model ini.")
        return

    latest_run = read_latest_run_id(model_root)
    default_idx = run_options.index(latest_run) if latest_run in run_options else 0
    selected_run = st.selectbox("Pilih run", run_options, index=default_idx, key="report_run")

    run_dir = model_root / selected_run
    manifest = safe_load_json(run_dir / "run_manifest.json") or {}
    metrics = safe_load_json(run_dir / "evaluation_metrics.json") or {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "{:.4f}".format(float(metrics.get("accuracy", 0.0))))
    c2.metric("F1-score", "{:.4f}".format(float(metrics.get("f1_score", 0.0))))
    roc_auc = metrics.get("roc_auc")
    if roc_auc is None or (isinstance(roc_auc, float) and np.isnan(roc_auc)):
        c3.metric("ROC-AUC", "NaN")
    else:
        c3.metric("ROC-AUC", "{:.4f}".format(float(roc_auc)))

    if manifest:
        st.markdown("**Run Manifest**")
        st.json(manifest)

    summary_path = run_dir / "summary.txt"
    if summary_path.exists():
        st.markdown("**Summary**")
        st.code(summary_path.read_text(encoding="utf-8"))

    history_df = _read_csv(run_dir / "training_history.csv")
    if not history_df.empty:
        st.markdown("**Training History**")
        cols = [c for c in ["loss", "val_loss", "accuracy", "val_accuracy"] if c in history_df.columns]
        if cols:
            st.line_chart(history_df[cols])
        st.dataframe(history_df, use_container_width=True)

    cls_df = _read_csv(run_dir / "classification_report.csv")
    if not cls_df.empty:
        st.markdown("**Classification Report**")
        st.dataframe(cls_df, use_container_width=True)

    cm_df = _read_csv(run_dir / "confusion_matrix.csv")
    if not cm_df.empty:
        st.markdown("**Confusion Matrix (CSV)**")
        st.dataframe(cm_df, use_container_width=True)

    split_df = _read_csv(run_dir / "split_distribution.csv")
    if not split_df.empty:
        st.markdown("**Split Distribution (dari run ini)**")
        st.dataframe(split_df, use_container_width=True)

    visuals = [run_dir / name for name in VISUAL_FILES]
    available_visuals = [path for path in visuals if path.exists()]
    if available_visuals:
        st.markdown("**Visualisasi Report**")
        for image_path in available_visuals:
            st.image(str(image_path), caption=image_path.name, use_container_width=True)


def _list_trained_models_for_dataset(dataset_name: str) -> List[str]:
    dataset_root = TRAINED_MODELS_ROOT / dataset_name
    if not dataset_root.exists() or not dataset_root.is_dir():
        return []
    return sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])


@st.cache_resource(show_spinner=False)
def _cached_load_model(run_dir_key: str):
    return load_model_bundle(Path(run_dir_key))


def render_prediction_tab(model_registry: ModelRegistry) -> None:
    st.subheader("Prediksi Gambar")

    dataset_options = list_datasets(TRAINED_MODELS_ROOT)
    if not dataset_options:
        st.info("Belum ada trained model di `trained_models/<dataset>/<model>/<run_id>/`.")
        return

    selected_dataset = st.selectbox("Pilih dataset model", dataset_options, index=0, key="predict_dataset")
    model_options = _list_trained_models_for_dataset(selected_dataset)
    if not model_options:
        st.info("Belum ada model untuk dataset yang dipilih.")
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
    for model_name in selected_models:
        model_root = TRAINED_MODELS_ROOT / selected_dataset / model_name
        runs = sorted([p.name for p in model_root.iterdir() if p.is_dir()], reverse=True) if model_root.exists() else []
        if not runs:
            continue
        latest_run = read_latest_run_id(model_root)
        default_idx = runs.index(latest_run) if latest_run in runs else 0
        selected_runs[model_name] = st.selectbox(
            f"Run untuk {model_name}",
            runs,
            index=default_idx,
            key=f"predict_run_{model_name}",
        )

    uploaded = st.file_uploader("Upload gambar (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload gambar untuk mulai prediksi.")
        return

    image_bytes = uploaded.getvalue()
    preview = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.image(preview, caption="Input", width=320)

    if st.button("Jalankan Prediksi"):
        summary_rows = []
        prob_rows = []
        errors = []

        for model_name in selected_models:
            run_id = selected_runs.get(model_name)
            if not run_id:
                errors.append(f"{model_name}: run tidak tersedia")
                continue

            run_dir = TRAINED_MODELS_ROOT / selected_dataset / model_name / run_id
            try:
                model_bundle = _cached_load_model(str(run_dir))
                preprocess_key = model_registry.get(model_name).preprocess_key
                prediction = predict_from_bundle(
                    model_bundle=model_bundle,
                    image_bytes=image_bytes,
                    preprocess_key=preprocess_key,
                )
                summary_rows.append(
                    {
                        "dataset": selected_dataset,
                        "model": model_name,
                        "run_id": run_id,
                        "predicted_label": prediction["predicted_label"],
                        "confidence": round(float(prediction["confidence"]) * 100.0, 2),
                    }
                )
                for class_name, prob in zip(prediction["class_names"], prediction["probabilities"]):
                    prob_rows.append(
                        {
                            "model": model_name,
                            "class": class_name,
                            "probability": float(prob),
                        }
                    )
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).sort_values(by="confidence", ascending=False)
            st.markdown("**Hasil Prediksi per Model**")
            st.dataframe(summary_df, use_container_width=True)

        if prob_rows:
            prob_df = pd.DataFrame(prob_rows)
            st.markdown("**Perbandingan Probabilitas**")
            pivot_df = prob_df.pivot(index="class", columns="model", values="probability").fillna(0.0)
            st.bar_chart(pivot_df)
            st.dataframe(prob_df, use_container_width=True)

        if errors:
            st.error("Sebagian model gagal dipakai:\n- " + "\n- ".join(errors))


def main() -> None:
    dataset_registry = DatasetRegistry()
    model_registry = ModelRegistry()

    st.set_page_config(
        page_title="Parkinson Classification Dashboard",
        page_icon="🧠",
        layout="wide",
    )

    st.title("Parkinson Classification Dashboard")
    st.caption("Visualisasi dataset, report training, dan prediksi model lintas run.")

    tab_dataset, tab_report, tab_predict = st.tabs([
        "Dataset",
        "Training Report",
        "Prediksi",
    ])

    with tab_dataset:
        render_dataset_tab(dataset_registry)
    with tab_report:
        render_report_tab()
    with tab_predict:
        render_prediction_tab(model_registry)


if __name__ == "__main__":
    main()
