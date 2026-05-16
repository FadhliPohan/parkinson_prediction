from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
    build_experiment_index,
    build_latest_summary_table,
    safe_load_json,
)
from src.reporting.schemas import VISUAL_FILES
from src.utils.paths import REPORT_ROOT


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def _count_split_distribution(split_root: Path) -> Dict[str, Dict[str, int]]:
    distribution: Dict[str, Dict[str, int]] = {}
    for split_name in ["train", "testing", "validation"]:
        class_map: Dict[str, int] = {}
        split_dir = split_root / split_name
        if split_dir.exists() and split_dir.is_dir():
            for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
                class_map[class_dir.name] = len(list_image_files(class_dir, IMAGE_EXTENSIONS))
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


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _metric_text(value: Optional[float]) -> str:
    if value is None or np.isnan(value):
        return "NaN"
    return "{:.4f}".format(value)


def _build_record_dataframe(records: List[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for item in records:
        rows.append(
            {
                "experiment_id": item.get("experiment_id"),
                "dataset": item.get("dataset"),
                "augmentation": item.get("augmentation"),
                "method": item.get("method"),
                "model": item.get("model"),
                "run_id": item.get("run_id"),
                "run_started_at": item.get("run_started_at"),
                "run_finished_at": item.get("run_finished_at"),
                "train_accuracy": _safe_float(item.get("train_accuracy")),
                "val_accuracy": _safe_float(item.get("val_accuracy")),
                "train_loss": _safe_float(item.get("train_loss")),
                "val_loss": _safe_float(item.get("val_loss")),
                "test_accuracy": _safe_float(item.get("accuracy")),
                "f1_score": _safe_float(item.get("f1_score")),
                "training_time_seconds": _safe_float(item.get("training_time_seconds")),
                "report_dir": item.get("report_dir") or item.get("run_dir"),
                "model_path": item.get("final_model_path"),
                "model_dir": item.get("model_dir"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "run_started_at" in df.columns:
        df = df.sort_values(by="run_started_at", ascending=False, na_position="last")
    return df


def _latest_per_key(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    sorted_df = df.sort_values(by="run_started_at", ascending=False, na_position="last")
    return sorted_df.drop_duplicates(subset=subset, keep="first")


def render_dataset_tab(dataset_registry: DatasetRegistry) -> None:
    st.subheader("Ringkasan Dataset")

    dataset_ids = dataset_registry.list_dataset_ids()
    default_index = dataset_ids.index(dataset_registry.default_dataset)
    selected_dataset = st.selectbox("Pilih dataset", dataset_ids, index=default_index, key="dataset_tab_dataset")
    dataset_cfg = dataset_registry.get(selected_dataset)

    st.caption(f"Original: `{dataset_cfg.original_path}`")
    st.caption(f"Split: `{dataset_cfg.split_path}`")
    if dataset_cfg.augmentation_options:
        st.caption("Augmentasi yang diizinkan: `{}`".format(", ".join(dataset_cfg.augmentation_options)))

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


def _render_run_detail(record: Dict[str, object]) -> None:
    dataset = str(record.get("dataset"))
    augmentation = str(record.get("augmentation"))
    method = str(record.get("method"))
    model = str(record.get("model"))
    run_id = str(record.get("run_id"))

    run_dir = Path(str(record.get("run_dir")))
    manifest = safe_load_json(run_dir / "run_manifest.json") or {}

    st.markdown("**Identitas Eksperimen**")
    st.write(
        {
            "experiment_id": record.get("experiment_id"),
            "dataset": dataset,
            "augmentation": augmentation,
            "method": method,
            "model": model,
            "run_id": run_id,
        }
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Accuracy", _metric_text(_safe_float(record.get("train_accuracy"))))
    c2.metric("Validation Accuracy", _metric_text(_safe_float(record.get("val_accuracy"))))
    c3.metric("Train Loss", _metric_text(_safe_float(record.get("train_loss"))))
    c4.metric("Validation Loss", _metric_text(_safe_float(record.get("val_loss"))))

    c5, c6, c7 = st.columns(3)
    c5.metric("Test Accuracy", _metric_text(_safe_float(record.get("accuracy"))))
    c6.metric("F1 Score", _metric_text(_safe_float(record.get("f1_score"))))
    duration = _safe_float(record.get("training_time_seconds"))
    c7.metric("Training Time (s)", "{:.2f}".format(duration) if duration is not None else "NaN")

    st.markdown("**Path Artifact**")
    st.code(
        "\n".join(
            [
                "Report Path : {}".format(record.get("report_dir") or run_dir),
                "Model Path  : {}".format(record.get("final_model_path") or "-"),
                "Best Model  : {}".format(record.get("best_model_path") or "-"),
            ]
        )
    )

    with st.expander("Parameter Penting"):
        training = manifest.get("training", {}) if isinstance(manifest, dict) else {}
        st.json(training.get("parameters", {}))

    with st.expander("Run Manifest Lengkap"):
        st.json(manifest)

    history_df = _read_csv(run_dir / "training_history.csv")
    if not history_df.empty:
        st.markdown("**Grafik Training**")

        acc_cols = [
            col
            for col in [
                "accuracy",
                "val_accuracy",
                "train/accuracy",
                "val/accuracy_top1",
                "metrics/accuracy_top1",
            ]
            if col in history_df.columns
        ]
        loss_cols = [
            col
            for col in ["loss", "val_loss", "train/loss", "val/loss", "train/cls_loss", "val/cls_loss"]
            if col in history_df.columns
        ]

        if acc_cols:
            st.caption("Akurasi")
            st.line_chart(history_df[acc_cols])
        if loss_cols:
            st.caption("Loss")
            st.line_chart(history_df[loss_cols])

        with st.expander("Training History (Tabel)"):
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
        st.markdown("**Distribusi Split pada Run Ini**")
        st.dataframe(split_df, use_container_width=True)

    visuals = [run_dir / name for name in VISUAL_FILES]
    available_visuals = [path for path in visuals if path.exists()]
    if available_visuals:
        st.markdown("**Visualisasi**")
        for image_path in available_visuals:
            st.image(str(image_path), caption=image_path.name, use_container_width=True)


def _render_comparison_section(records: List[Dict[str, object]]) -> None:
    df = _build_record_dataframe(records)
    if df.empty:
        st.info("Belum ada data untuk perbandingan.")
        return

    st.markdown("**Perbandingan Eksperimen**")

    dataset_options = sorted(df["dataset"].dropna().unique().tolist())
    selected_dataset = st.selectbox("Dataset untuk perbandingan", dataset_options, index=0, key="cmp_dataset")

    subset_df = df[df["dataset"] == selected_dataset].copy()
    if subset_df.empty:
        st.info("Belum ada run untuk dataset ini.")
        return

    tab_method, tab_model, tab_aug, tab_rank = st.tabs(
        [
            "Perbandingan Method",
            "Perbandingan Model",
            "Perbandingan Augmentasi",
            "Ranking Best Model",
        ]
    )

    with tab_method:
        by_method = _latest_per_key(subset_df, ["dataset", "augmentation", "method", "model"])
        method_summary = (
            by_method.groupby("method", dropna=False)
            .agg(
                runs=("run_id", "count"),
                avg_val_accuracy=("val_accuracy", "mean"),
                avg_test_accuracy=("test_accuracy", "mean"),
                avg_f1=("f1_score", "mean"),
            )
            .reset_index()
            .sort_values(by="avg_val_accuracy", ascending=False, na_position="last")
        )
        st.dataframe(method_summary, use_container_width=True)

    with tab_model:
        method_options = sorted(subset_df["method"].dropna().unique().tolist())
        selected_method = st.selectbox("Pilih method", method_options, index=0, key="cmp_model_method")
        method_df = subset_df[subset_df["method"] == selected_method].copy()
        by_model = _latest_per_key(method_df, ["dataset", "augmentation", "method", "model"])
        model_summary = (
            by_model.groupby("model", dropna=False)
            .agg(
                runs=("run_id", "count"),
                avg_val_accuracy=("val_accuracy", "mean"),
                avg_test_accuracy=("test_accuracy", "mean"),
                avg_f1=("f1_score", "mean"),
            )
            .reset_index()
            .sort_values(by="avg_val_accuracy", ascending=False, na_position="last")
        )
        st.dataframe(model_summary, use_container_width=True)

    with tab_aug:
        by_aug = _latest_per_key(subset_df, ["dataset", "augmentation", "method", "model"])
        aug_summary = (
            by_aug.groupby("augmentation", dropna=False)
            .agg(
                runs=("run_id", "count"),
                avg_val_accuracy=("val_accuracy", "mean"),
                avg_test_accuracy=("test_accuracy", "mean"),
                avg_f1=("f1_score", "mean"),
            )
            .reset_index()
            .sort_values(by="avg_val_accuracy", ascending=False, na_position="last")
        )
        st.dataframe(aug_summary, use_container_width=True)

    with tab_rank:
        latest_runs = _latest_per_key(subset_df, ["dataset", "augmentation", "method", "model"])  # 1 run terbaru / kombinasi
        ranking = latest_runs.sort_values(by="val_accuracy", ascending=False, na_position="last").reset_index(drop=True)
        ranking.index = ranking.index + 1
        st.dataframe(
            ranking[
                [
                    "dataset",
                    "augmentation",
                    "method",
                    "model",
                    "run_id",
                    "val_accuracy",
                    "test_accuracy",
                    "f1_score",
                    "training_time_seconds",
                ]
            ],
            use_container_width=True,
        )


def render_report_tab() -> None:
    st.subheader("Report Training")

    records = build_experiment_index(REPORT_ROOT)
    if not records:
        st.info("Belum ada report training.")
        return

    latest_rows = build_latest_summary_table(REPORT_ROOT)
    latest_df = pd.DataFrame(latest_rows)
    if not latest_df.empty:
        st.markdown("**Ringkasan Run Terbaru per Kombinasi**")
        st.dataframe(latest_df, use_container_width=True)

    tab_explorer, tab_comparison = st.tabs(["Explorer", "Perbandingan"])

    with tab_explorer:
        record_df = _build_record_dataframe(records)
        if record_df.empty:
            st.info("Belum ada data report.")
            return

        dataset_options = sorted(record_df["dataset"].dropna().unique().tolist())
        selected_dataset = st.selectbox("1) Dataset", dataset_options, index=0, key="report_dataset")

        dataset_df = record_df[record_df["dataset"] == selected_dataset].copy()
        aug_options = sorted(dataset_df["augmentation"].dropna().unique().tolist())
        if not aug_options:
            st.warning("Belum ada augmentasi untuk dataset ini.")
            return
        selected_aug = st.selectbox("2) Augmentasi", aug_options, index=0, key="report_augmentation")

        aug_df = dataset_df[dataset_df["augmentation"] == selected_aug].copy()
        method_options = sorted(aug_df["method"].dropna().unique().tolist())
        if not method_options:
            st.warning("Belum ada method untuk kombinasi dataset + augmentasi ini.")
            return
        selected_method = st.selectbox("3) Method Training", method_options, index=0, key="report_method")

        method_df = aug_df[aug_df["method"] == selected_method].copy()
        model_options = sorted(method_df["model"].dropna().unique().tolist())
        if not model_options:
            st.warning("Belum ada model untuk kombinasi dataset + augmentasi + method ini.")
            return
        selected_model = st.selectbox("4) Model", model_options, index=0, key="report_model")

        model_df = method_df[method_df["model"] == selected_model].copy()
        model_df = model_df.sort_values(by="run_started_at", ascending=False, na_position="last")
        run_options = model_df["run_id"].dropna().astype(str).tolist()
        if not run_options:
            st.warning("Belum ada run untuk kombinasi ini.")
            return
        selected_run = st.selectbox("5) Hasil Training (Run)", run_options, index=0, key="report_run")

        target_df = model_df[model_df["run_id"] == selected_run]
        if target_df.empty:
            st.error("Run record tidak ditemukan.")
            return

        target_run_dir = Path(str(target_df.iloc[0]["run_dir"]))
        manifest = safe_load_json(target_run_dir / "run_manifest.json")
        if not isinstance(manifest, dict):
            st.error("Manifest run tidak valid.")
            return

        selected_record = None
        for item in records:
            if (
                item.get("dataset") == selected_dataset
                and item.get("augmentation") == selected_aug
                and item.get("method") == selected_method
                and item.get("model") == selected_model
                and str(item.get("run_id")) == str(selected_run)
            ):
                selected_record = item
                break

        if selected_record is None:
            st.error("Run record tidak ditemukan di indeks.")
            return

        _render_run_detail(selected_record)

    with tab_comparison:
        _render_comparison_section(records)


@st.cache_resource(show_spinner=False)
def _cached_load_model(run_dir_key: str):
    return load_model_bundle(Path(run_dir_key))


def render_prediction_tab(model_registry: ModelRegistry) -> None:
    st.subheader("Prediksi Gambar")

    records = build_experiment_index(REPORT_ROOT)
    record_df = _build_record_dataframe(records)
    if record_df.empty:
        st.info("Belum ada model/report untuk prediksi.")
        return

    dataset_options = sorted(record_df["dataset"].dropna().unique().tolist())
    selected_dataset = st.selectbox("Pilih dataset model", dataset_options, index=0, key="predict_dataset")

    dataset_df = record_df[record_df["dataset"] == selected_dataset].copy()
    aug_options = sorted(dataset_df["augmentation"].dropna().unique().tolist())
    selected_aug = st.selectbox("Pilih augmentasi", aug_options, index=0, key="predict_aug")

    aug_df = dataset_df[dataset_df["augmentation"] == selected_aug].copy()
    method_options = sorted(aug_df["method"].dropna().unique().tolist())
    selected_method = st.selectbox("Pilih method", method_options, index=0, key="predict_method")

    scope_df = aug_df[aug_df["method"] == selected_method].copy()
    model_options = sorted(scope_df["model"].dropna().unique().tolist())
    if not model_options:
        st.info("Belum ada model pada kombinasi ini.")
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
        model_df = scope_df[scope_df["model"] == model_name].copy()
        model_df = model_df.sort_values(by="run_started_at", ascending=False, na_position="last")
        run_choices = model_df["run_id"].dropna().astype(str).tolist()
        if not run_choices:
            continue
        selected_runs[model_name] = st.selectbox(
            f"Run untuk {model_name}",
            run_choices,
            index=0,
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

            target = scope_df[(scope_df["model"] == model_name) & (scope_df["run_id"] == run_id)]
            if target.empty:
                errors.append(f"{model_name}: run record tidak ditemukan")
                continue

            record = target.iloc[0].to_dict()
            model_dir = record.get("model_dir")
            if not model_dir:
                errors.append(f"{model_name}: path model tidak tersedia di manifest")
                continue

            run_dir = Path(str(model_dir))
            if not run_dir.exists():
                errors.append(f"{model_name}: folder model tidak ditemukan ({run_dir})")
                continue

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
                        "augmentation": selected_aug,
                        "method": selected_method,
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
        layout="wide",
    )

    st.title("Parkinson Classification Dashboard")
    st.caption("Visualisasi dataset, report training bertingkat, perbandingan eksperimen, dan prediksi model.")

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
