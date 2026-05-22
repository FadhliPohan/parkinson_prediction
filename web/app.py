from __future__ import annotations

import io
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.reporting.html_report import build_full_html_report
from src.reporting.schemas import VISUAL_FILES
from src.training.augmentations import TrainingAugmentationRegistry
from src.training.strategies import TrainingMethodRegistry
from src.utils.paths import REPORT_ROOT


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def _inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

        :root,
        html[data-theme="light"],
        html[data-theme="dark"],
        body[data-theme="light"],
        body[data-theme="dark"] {
            --pp-bg-a: #f7faf8;
            --pp-bg-b: #e9f4ef;
            --pp-card: #ffffff;
            --pp-line: #d7e7de;
            --pp-ink: #123726;
            --pp-ink-soft: #4a6a5b;
            --pp-accent: #1c8f5a;
            --pp-accent-soft: #e4f6ee;
            color-scheme: light !important;
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            font-family: "Manrope", "Avenir Next", "Segoe UI", sans-serif;
            color: var(--pp-ink) !important;
            background:
                radial-gradient(circle at 8% 0%, #ffffff 0%, var(--pp-bg-a) 42%, var(--pp-bg-b) 100%) !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f2faf6 0%, #ebf5f0 100%) !important;
            border-right: 1px solid var(--pp-line) !important;
        }

        h1, h2, h3, h4, h5, h6, p, label, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
            color: var(--pp-ink) !important;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            background: var(--pp-card);
            border: 1px solid var(--pp-line);
            border-radius: 12px;
            padding: 0.5rem 0.7rem;
            box-shadow: 0 8px 26px rgba(18, 55, 38, 0.05);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--pp-ink-soft);
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: var(--pp-ink);
            font-weight: 800;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea,
        input {
            background: #ffffff !important;
            color: var(--pp-ink) !important;
            border: 1px solid var(--pp-line) !important;
            border-radius: 10px;
        }

        div[data-baseweb="select"] svg,
        div[data-baseweb="input"] svg {
            fill: var(--pp-ink-soft) !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: #ffffff !important;
            border: 1px solid var(--pp-line) !important;
            border-radius: 10px;
        }

        button[kind="secondary"] {
            border-radius: 10px;
            border-color: var(--pp-line);
        }

        button[kind="primary"] {
            border-radius: 10px;
            background: var(--pp-accent);
            border-color: var(--pp-accent);
        }

        div[role="tablist"] button {
            border-radius: 10px 10px 0 0;
        }

        .pp-note {
            background: var(--pp-accent-soft);
            border: 1px solid #b7e8d1;
            color: var(--pp-ink);
            border-radius: 12px;
            padding: 0.6rem 0.85rem;
            font-size: 0.93rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    split_df = split_df.fillna(0).astype(int)
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


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_datetime_text(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    try:
        dt = datetime.fromisoformat(text)
        return dt.strftime("%d %b %Y %H:%M")
    except Exception:
        return text


def _value_to_path(value: object) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return Path(text)


def _resolve_run_dir(record: Dict[str, object]) -> Optional[Path]:
    direct_path = _value_to_path(record.get("run_dir"))
    if direct_path is not None:
        return direct_path

    report_path = _value_to_path(record.get("report_dir"))
    if report_path is not None:
        return report_path

    dataset = str(record.get("dataset") or "").strip()
    augmentation = str(record.get("augmentation") or "").strip()
    method = str(record.get("method") or "").strip()
    model = str(record.get("model") or "").strip()
    run_id = str(record.get("run_id") or "").strip()
    if dataset and augmentation and method and model and run_id:
        return REPORT_ROOT / dataset / augmentation / method / model / run_id

    return None


def _metric_text(value: Optional[float]) -> str:
    if value is None or np.isnan(value):
        return "NaN"
    return "{:.4f}".format(value)


def _duration_text(value: Optional[float]) -> str:
    if value is None or np.isnan(value):
        return "NaN"
    return "{:.2f}".format(value)


def _build_run_label(row: pd.Series) -> str:
    run_id = str(row.get("run_id") or "-")
    started = _safe_datetime_text(row.get("run_started_at"))

    metrics: List[str] = []
    val_acc = _safe_float(row.get("val_accuracy"))
    test_acc = _safe_float(row.get("test_accuracy"))
    if val_acc is not None and not np.isnan(val_acc):
        metrics.append(f"val={_metric_text(val_acc)}")
    if test_acc is not None and not np.isnan(test_acc):
        metrics.append(f"test={_metric_text(test_acc)}")

    label_parts = [run_id]
    if started != "-":
        label_parts.append(started)
    if metrics:
        label_parts.append(" | ".join(metrics))
    return " | ".join(label_parts)


def _counts_by_column(df: pd.DataFrame, column: str) -> Dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    values = df[column].dropna().astype(str)
    if values.empty:
        return {}
    return {str(key): int(val) for key, val in values.value_counts().to_dict().items()}


def _format_option_with_run_count(option: str, count_map: Dict[str, int]) -> str:
    return f"{option} ({count_map.get(str(option), 0)} run)"


def _sanitize_widget_key(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_")
    return cleaned or "col"


def _render_filterable_dataframe(
    df: pd.DataFrame,
    key_prefix: str,
    *,
    use_container_width: bool = True,
    filters_expanded: bool = False,
    empty_message: str = "Tidak ada data untuk ditampilkan.",
) -> pd.DataFrame:
    if df.empty:
        st.info(empty_message)
        return df

    filtered_df = df.copy()

    with st.expander("Search / Filter Tabel", expanded=filters_expanded):
        c_search_1, c_search_2 = st.columns((2, 1))
        search_term = c_search_1.text_input(
            "Cari keyword",
            key=f"{key_prefix}_search_term",
            placeholder="Cari teks/angka pada tabel...",
        ).strip()
        search_column_options = ["(Semua kolom)"] + [str(col) for col in filtered_df.columns]
        selected_search_column = c_search_2.selectbox(
            "Kolom pencarian",
            options=search_column_options,
            index=0,
            key=f"{key_prefix}_search_col",
        )

        filter_columns = st.multiselect(
            "Filter nilai kolom (opsional)",
            options=[str(col) for col in filtered_df.columns],
            default=[],
            key=f"{key_prefix}_filter_cols",
        )

        exact_filters: Dict[str, set[str]] = {}
        contains_filters: Dict[str, str] = {}

        for column_name in filter_columns:
            safe_col_key = _sanitize_widget_key(column_name)
            col_series = filtered_df[column_name]
            unique_values = sorted({str(item) for item in col_series.dropna().unique()})

            if unique_values and len(unique_values) <= 40:
                selected_values = st.multiselect(
                    f"Nilai untuk `{column_name}`",
                    options=unique_values,
                    default=[],
                    key=f"{key_prefix}_filter_vals_{safe_col_key}",
                )
                if selected_values:
                    exact_filters[column_name] = set(selected_values)
            else:
                contains_value = st.text_input(
                    f"Cari teks di `{column_name}`",
                    key=f"{key_prefix}_filter_contains_{safe_col_key}",
                    placeholder="contains...",
                ).strip()
                if contains_value:
                    contains_filters[column_name] = contains_value

        if search_term:
            if selected_search_column == "(Semua kolom)":
                search_mask = pd.Series(False, index=filtered_df.index)
                for col in filtered_df.columns:
                    col_mask = filtered_df[col].astype(str).str.contains(search_term, case=False, na=False, regex=False)
                    search_mask = search_mask | col_mask
                filtered_df = filtered_df[search_mask]
            elif selected_search_column in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df[selected_search_column].astype(str).str.contains(
                        search_term, case=False, na=False, regex=False
                    )
                ]

        for col, accepted in exact_filters.items():
            if col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(accepted)]

        for col, contains_text in contains_filters.items():
            if col in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df[col].astype(str).str.contains(
                        contains_text, case=False, na=False, regex=False
                    )
                ]

        st.caption(f"Hasil filter: {len(filtered_df)} / {len(df)} baris")

    st.dataframe(filtered_df, use_container_width=use_container_width)
    return filtered_df


def _build_record_dataframe(records: List[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for item in records:
        run_dir = item.get("run_dir") or item.get("report_dir")
        rows.append(
            {
                "experiment_id": item.get("experiment_id"),
                "dataset": item.get("dataset"),
                "augmentation": item.get("augmentation"),
                "method": item.get("method"),
                "model": item.get("model"),
                "run_id": str(item.get("run_id") or ""),
                "run_started_at": item.get("run_started_at"),
                "run_finished_at": item.get("run_finished_at"),
                "epochs": _safe_int(item.get("epochs")),
                "batch_size": _safe_int(item.get("batch_size")),
                "fine_tune_epochs": _safe_int(item.get("fine_tune_epochs")),
                "train_accuracy": _safe_float(item.get("train_accuracy")),
                "val_accuracy": _safe_float(item.get("val_accuracy")),
                "train_loss": _safe_float(item.get("train_loss")),
                "val_loss": _safe_float(item.get("val_loss")),
                "test_accuracy": _safe_float(item.get("accuracy") or item.get("test_accuracy")),
                "f1_score": _safe_float(item.get("f1_score")),
                "training_time_seconds": _safe_float(item.get("training_time_seconds")),
                "run_dir": run_dir,
                "report_dir": item.get("report_dir") or run_dir,
                "model_dir": item.get("model_dir"),
                "best_model_path": item.get("best_model_path"),
                "final_model_path": item.get("final_model_path"),
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
    if not dataset_ids:
        st.info("Belum ada dataset terdaftar di registry.")
        return

    default_index = dataset_ids.index(dataset_registry.default_dataset)
    selected_dataset = st.selectbox("Pilih dataset", dataset_ids, index=default_index, key="dataset_tab_dataset")
    dataset_cfg = dataset_registry.get(selected_dataset)

    st.markdown(
        "<div class='pp-note'>"
        f"<b>Original</b>: <code>{dataset_cfg.original_path}</code><br/>"
        f"<b>Split</b>: <code>{dataset_cfg.split_path}</code>"
        "</div>",
        unsafe_allow_html=True,
    )
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
        if not original_df.empty:
            original_df = original_df.sort_values(by="count", ascending=False).reset_index(drop=True)
    except Exception as exc:
        st.error(f"Gagal baca dataset original: {exc}")
        original_df = pd.DataFrame()

    split_distribution = _count_split_distribution(dataset_cfg.split_path)
    split_df = _build_split_table(split_distribution)

    total_original = int(original_df["count"].sum()) if not original_df.empty else 0
    class_count = int(len(original_df)) if not original_df.empty else 0
    train_total = int(split_df.loc["train", "total"]) if not split_df.empty and "train" in split_df.index else 0
    test_total = int(split_df.loc["testing", "total"]) if not split_df.empty and "testing" in split_df.index else 0
    val_total = int(split_df.loc["validation", "total"]) if not split_df.empty and "validation" in split_df.index else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Jumlah Kelas", class_count)
    m2.metric("Total Gambar Original", total_original)
    m3.metric("Split Train", train_total)
    m4.metric("Split Testing", test_total)
    m5.metric("Split Validation", val_total)

    c1, c2 = st.columns((1, 1))

    with c1:
        st.markdown("**Dataset Original**")
        if not original_df.empty:
            _render_filterable_dataframe(
                original_df,
                key_prefix="dataset_original_table",
                use_container_width=True,
                empty_message="Dataset original belum siap atau belum terbaca.",
            )
            chart_df = original_df.set_index("class_name")[["count"]]
            st.bar_chart(chart_df)
        else:
            st.info("Dataset original belum siap atau belum terbaca.")

    with c2:
        st.markdown("**Dataset Split (train/testing/validation)**")
        if not split_df.empty:
            _render_filterable_dataframe(
                split_df,
                key_prefix="dataset_split_table",
                use_container_width=True,
                empty_message="Dataset split belum tersedia.",
            )
            chart_df = split_df.drop(columns=["total"], errors="ignore")
            if not chart_df.empty:
                st.bar_chart(chart_df)
        else:
            st.info("Dataset split belum tersedia.")


def _render_run_detail(record: Dict[str, object]) -> None:
    dataset = str(record.get("dataset") or "-")
    augmentation = str(record.get("augmentation") or "-")
    method = str(record.get("method") or "-")
    model = str(record.get("model") or "-")
    run_id = str(record.get("run_id") or "-")

    run_dir = _resolve_run_dir(record)
    if run_dir is None:
        st.error("Path run report tidak tersedia pada record ini.")
        return

    manifest = safe_load_json(run_dir / "run_manifest.json") or {}
    if not isinstance(manifest, dict):
        manifest = {}

    st.markdown("**Identitas Eksperimen**")
    st.write(
        {
            "experiment_id": record.get("experiment_id"),
            "dataset": dataset,
            "augmentation": augmentation,
            "method": method,
            "model": model,
            "run_id": run_id,
            "run_started_at": record.get("run_started_at"),
            "run_finished_at": record.get("run_finished_at"),
        }
    )

    training = manifest.get("training", {}) if isinstance(manifest, dict) else {}
    run_params = training.get("parameters", {}) if isinstance(training, dict) else {}
    if not isinstance(run_params, dict):
        run_params = {}

    epochs_value = _safe_int(run_params.get("epochs") or record.get("epochs"))
    batch_size_value = _safe_int(run_params.get("batch_size") or record.get("batch_size"))
    fine_tune_epochs_value = _safe_int(run_params.get("fine_tune_epochs") or record.get("fine_tune_epochs"))

    c_cfg_1, c_cfg_2, c_cfg_3 = st.columns(3)
    c_cfg_1.metric("Epoch Stage-1", str(epochs_value) if epochs_value is not None else "-")
    c_cfg_2.metric("Batch Size", str(batch_size_value) if batch_size_value is not None else "-")
    c_cfg_3.metric("Fine-tune Epochs", str(fine_tune_epochs_value) if fine_tune_epochs_value is not None else "-")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Accuracy", _metric_text(_safe_float(record.get("train_accuracy"))))
    c2.metric("Validation Accuracy", _metric_text(_safe_float(record.get("val_accuracy"))))
    c3.metric("Train Loss", _metric_text(_safe_float(record.get("train_loss"))))
    c4.metric("Validation Loss", _metric_text(_safe_float(record.get("val_loss"))))

    c5, c6, c7 = st.columns(3)
    test_accuracy = _safe_float(record.get("accuracy") or record.get("test_accuracy"))
    c5.metric("Test Accuracy", _metric_text(test_accuracy))
    c6.metric("F1 Score", _metric_text(_safe_float(record.get("f1_score"))))
    duration = _safe_float(record.get("training_time_seconds"))
    c7.metric("Training Time (s)", _duration_text(duration))

    report_path = _value_to_path(record.get("report_dir")) or run_dir
    model_dir = _value_to_path(record.get("model_dir"))

    st.markdown("**Path Artifact**")
    st.code(
        "\n".join(
            [
                f"Run Report : {run_dir}",
                f"Report Dir : {report_path}",
                f"Model Dir  : {model_dir if model_dir else '-'}",
                "Final Model: {}".format(record.get("final_model_path") or "-"),
                "Best Model : {}".format(record.get("best_model_path") or "-"),
            ]
        )
    )

    tab_chart, tab_tables, tab_manifest = st.tabs(["Grafik", "Tabel", "Manifest"])

    with tab_chart:
        history_df = _read_csv(run_dir / "training_history.csv")
        if not history_df.empty:
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
                _render_filterable_dataframe(
                    history_df,
                    key_prefix=f"run_history_{run_id}",
                    use_container_width=True,
                    empty_message="Training history tidak tersedia.",
                )
        else:
            st.info("`training_history.csv` tidak tersedia pada run ini.")

        visuals = [run_dir / name for name in VISUAL_FILES]
        available_visuals = [path for path in visuals if path.exists()]
        if available_visuals:
            st.markdown("**Visualisasi File PNG**")
            for image_path in available_visuals:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)

    with tab_tables:
        cls_df = _read_csv(run_dir / "classification_report.csv")
        if not cls_df.empty:
            st.markdown("**Classification Report**")
            _render_filterable_dataframe(
                cls_df,
                key_prefix=f"run_cls_{run_id}",
                use_container_width=True,
                empty_message="Classification report tidak tersedia.",
            )

        cm_df = _read_csv(run_dir / "confusion_matrix.csv")
        if not cm_df.empty:
            st.markdown("**Confusion Matrix (CSV)**")
            _render_filterable_dataframe(
                cm_df,
                key_prefix=f"run_cm_{run_id}",
                use_container_width=True,
                empty_message="Confusion matrix tidak tersedia.",
            )

        split_df = _read_csv(run_dir / "split_distribution.csv")
        if not split_df.empty:
            st.markdown("**Distribusi Split pada Run Ini**")
            _render_filterable_dataframe(
                split_df,
                key_prefix=f"run_split_{run_id}",
                use_container_width=True,
                empty_message="Distribusi split tidak tersedia.",
            )

        if cls_df.empty and cm_df.empty and split_df.empty:
            st.info("Belum ada tabel CSV detail untuk run ini.")

    with tab_manifest:
        with st.expander("Parameter Training", expanded=True):
            st.json(run_params)

        st.markdown("**Run Manifest Lengkap**")
        if manifest:
            st.json(manifest)
        else:
            st.warning("`run_manifest.json` tidak tersedia atau tidak valid.")


def _render_comparison_section(
    records: List[Dict[str, object]],
    dataset_registry: DatasetRegistry,
    method_registry: TrainingMethodRegistry,
    augmentation_registry: TrainingAugmentationRegistry,
    model_registry: ModelRegistry,
) -> None:
    df = _build_record_dataframe(records)
    if df.empty:
        st.info("Belum ada data untuk perbandingan.")
        return

    st.markdown("**Perbandingan Eksperimen**")

    dataset_options = dataset_registry.list_dataset_ids()
    if not dataset_options:
        st.info("Belum ada dataset terdaftar.")
        return

    dataset_count_map = _counts_by_column(df, "dataset")
    selected_dataset = st.selectbox(
        "Dataset untuk perbandingan",
        dataset_options,
        index=0,
        key="cmp_dataset",
        format_func=lambda opt: _format_option_with_run_count(str(opt), dataset_count_map),
    )

    subset_df = df[df["dataset"] == selected_dataset].copy()

    tab_method, tab_model, tab_aug, tab_rank = st.tabs(
        [
            "Perbandingan Method",
            "Perbandingan Model",
            "Perbandingan Augmentasi",
            "Ranking Best Model",
        ]
    )

    with tab_method:
        by_method = _latest_per_key(subset_df, ["dataset", "augmentation", "method", "model"]) if not subset_df.empty else pd.DataFrame()
        method_summary = pd.DataFrame(
            columns=["method", "runs", "avg_val_accuracy", "avg_test_accuracy", "avg_f1"]
        )
        if not by_method.empty:
            method_summary = (
                by_method.groupby("method", dropna=False)
                .agg(
                    runs=("run_id", "count"),
                    avg_val_accuracy=("val_accuracy", "mean"),
                    avg_test_accuracy=("test_accuracy", "mean"),
                    avg_f1=("f1_score", "mean"),
                )
                .reset_index()
            )

        method_order = method_registry.list_method_ids()
        method_summary = (
            method_summary.set_index("method")
            .reindex(method_order)
            .reset_index()
            .rename(columns={"index": "method"})
        )
        if "runs" in method_summary.columns:
            method_summary["runs"] = method_summary["runs"].fillna(0).astype(int)
        method_summary = method_summary.sort_values(by="avg_val_accuracy", ascending=False, na_position="last")
        _render_filterable_dataframe(
            method_summary,
            key_prefix=f"cmp_method_{selected_dataset}",
            use_container_width=True,
            empty_message="Data perbandingan method belum tersedia.",
        )

    with tab_model:
        method_options = method_registry.list_method_ids()
        method_count_map = _counts_by_column(subset_df, "method")
        selected_method = st.selectbox(
            "Pilih method",
            method_options,
            index=0,
            key="cmp_model_method",
            format_func=lambda opt: _format_option_with_run_count(str(opt), method_count_map),
        )
        method_df = subset_df[subset_df["method"] == selected_method].copy()
        by_model = _latest_per_key(method_df, ["dataset", "augmentation", "method", "model"]) if not method_df.empty else pd.DataFrame()
        model_summary = pd.DataFrame(
            columns=["model", "runs", "avg_val_accuracy", "avg_test_accuracy", "avg_f1"]
        )
        if not by_model.empty:
            model_summary = (
                by_model.groupby("model", dropna=False)
                .agg(
                    runs=("run_id", "count"),
                    avg_val_accuracy=("val_accuracy", "mean"),
                    avg_test_accuracy=("test_accuracy", "mean"),
                    avg_f1=("f1_score", "mean"),
                )
                .reset_index()
            )

        model_order = model_registry.list_model_ids(enabled_only=True)
        model_summary = (
            model_summary.set_index("model")
            .reindex(model_order)
            .reset_index()
            .rename(columns={"index": "model"})
        )
        if "runs" in model_summary.columns:
            model_summary["runs"] = model_summary["runs"].fillna(0).astype(int)
        model_summary = model_summary.sort_values(by="avg_val_accuracy", ascending=False, na_position="last")
        _render_filterable_dataframe(
            model_summary,
            key_prefix=f"cmp_model_{selected_dataset}_{selected_method}",
            use_container_width=True,
            empty_message="Data perbandingan model belum tersedia.",
        )

    with tab_aug:
        by_aug = _latest_per_key(subset_df, ["dataset", "augmentation", "method", "model"]) if not subset_df.empty else pd.DataFrame()
        aug_summary = pd.DataFrame(
            columns=["augmentation", "runs", "avg_val_accuracy", "avg_test_accuracy", "avg_f1"]
        )
        if not by_aug.empty:
            aug_summary = (
                by_aug.groupby("augmentation", dropna=False)
                .agg(
                    runs=("run_id", "count"),
                    avg_val_accuracy=("val_accuracy", "mean"),
                    avg_test_accuracy=("test_accuracy", "mean"),
                    avg_f1=("f1_score", "mean"),
                )
                .reset_index()
            )

        dataset_cfg = dataset_registry.get(selected_dataset)
        aug_order = dataset_cfg.augmentation_options or augmentation_registry.list_augmentation_ids()
        aug_summary = (
            aug_summary.set_index("augmentation")
            .reindex(aug_order)
            .reset_index()
            .rename(columns={"index": "augmentation"})
        )
        if "runs" in aug_summary.columns:
            aug_summary["runs"] = aug_summary["runs"].fillna(0).astype(int)
        aug_summary = aug_summary.sort_values(by="avg_val_accuracy", ascending=False, na_position="last")
        _render_filterable_dataframe(
            aug_summary,
            key_prefix=f"cmp_aug_{selected_dataset}",
            use_container_width=True,
            empty_message="Data perbandingan augmentasi belum tersedia.",
        )

    with tab_rank:
        if subset_df.empty:
            st.info("Belum ada run untuk dataset ini.")
            return
        latest_runs = _latest_per_key(subset_df, ["dataset", "augmentation", "method", "model"])
        ranking = latest_runs.sort_values(by="val_accuracy", ascending=False, na_position="last").reset_index(drop=True)
        ranking.index = ranking.index + 1
        ranking_df = ranking[
            [
                "dataset",
                "augmentation",
                "method",
                "model",
                "run_id",
                "epochs",
                "batch_size",
                "fine_tune_epochs",
                "val_accuracy",
                "test_accuracy",
                "f1_score",
                "training_time_seconds",
            ]
        ]
        _render_filterable_dataframe(
            ranking_df,
            key_prefix=f"cmp_rank_{selected_dataset}",
            use_container_width=True,
            empty_message="Ranking model belum tersedia.",
        )


def _render_full_report_export_section(
    dataset_registry: DatasetRegistry,
    model_registry: ModelRegistry,
    augmentation_registry: TrainingAugmentationRegistry,
    method_registry: TrainingMethodRegistry,
) -> None:
    st.markdown("**Generate Laporan HTML (Auto Print A4)**")
    st.caption(
        "Tersedia 2 mode: `Lengkap` (embed semua gambar artifact run, file besar) "
        "dan `Ringkas` (tanpa embed gambar artifact run, file lebih ringan)."
    )

    c_btn_1, c_btn_2 = st.columns(2)
    do_generate_full = c_btn_1.button("Generate Report Lengkap", type="primary", key="generate_full_html_report_btn")
    do_generate_compact = c_btn_2.button("Generate Report Ringkas", type="secondary", key="generate_compact_html_report_btn")

    if do_generate_full:
        with st.spinner("Menyusun laporan lengkap. Proses bisa memakan waktu karena semua run akan dirangkum..."):
            _, output_path = build_full_html_report(
                dataset_registry=dataset_registry,
                model_registry=model_registry,
                augmentation_registry=augmentation_registry,
                method_registry=method_registry,
                report_root=REPORT_ROOT,
                report_mode="full",
                return_html=False,
            )
        st.session_state["html_report_last_path"] = str(output_path)
        st.session_state["html_report_last_mode"] = "Lengkap"
        st.success(f"Laporan lengkap berhasil dibuat: {output_path}")

    if do_generate_compact:
        with st.spinner("Menyusun laporan ringkas. Proses lebih cepat karena gambar run tidak di-embed..."):
            _, output_path = build_full_html_report(
                dataset_registry=dataset_registry,
                model_registry=model_registry,
                augmentation_registry=augmentation_registry,
                method_registry=method_registry,
                report_root=REPORT_ROOT,
                report_mode="compact",
                return_html=False,
            )
        st.session_state["html_report_last_path"] = str(output_path)
        st.session_state["html_report_last_mode"] = "Ringkas"
        st.success(f"Laporan ringkas berhasil dibuat: {output_path}")

    output_path_state = st.session_state.get("html_report_last_path")
    output_mode_state = st.session_state.get("html_report_last_mode", "-")
    if not output_path_state:
        return

    output_path = Path(str(output_path_state))
    if not output_path.exists():
        st.warning("File laporan terakhir tidak ditemukan. Silakan generate ulang.")
        return

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    st.markdown("**Output Laporan Terakhir**")
    st.code(
        f"{output_path}\nMode: {output_mode_state}\nUkuran file: {file_size_mb:.2f} MB"
    )

    st.download_button(
        "Download HTML Report",
        data=output_path.read_bytes(),
        file_name=output_path.name,
        mime="text/html",
        key="download_full_html_report_btn",
    )

    try:
        st.link_button(
            "Buka Laporan (Auto Print)",
            url=output_path.resolve().as_uri(),
            key="open_full_html_report_link_btn",
            type="secondary",
        )
    except Exception:
        pass

    st.info(
        "Saat file HTML dibuka di browser, window print akan terpanggil otomatis dan layout menyesuaikan kertas A4."
    )


def render_report_tab(
    dataset_registry: DatasetRegistry,
    model_registry: ModelRegistry,
    augmentation_registry: TrainingAugmentationRegistry,
    method_registry: TrainingMethodRegistry,
) -> None:
    st.subheader("Report Training")

    _render_full_report_export_section(
        dataset_registry=dataset_registry,
        model_registry=model_registry,
        augmentation_registry=augmentation_registry,
        method_registry=method_registry,
    )
    st.divider()

    records = build_experiment_index(REPORT_ROOT)
    if not records:
        st.info("Belum ada report training.")
        return

    latest_rows = build_latest_summary_table(REPORT_ROOT)
    latest_df = pd.DataFrame(latest_rows)
    if not latest_df.empty:
        st.markdown("**Ringkasan Run Terbaru per Kombinasi**")
        _render_filterable_dataframe(
            latest_df,
            key_prefix="report_latest_summary",
            use_container_width=True,
            empty_message="Ringkasan run terbaru belum tersedia.",
        )

    tab_explorer, tab_comparison = st.tabs(["Explorer", "Perbandingan"])

    with tab_explorer:
        record_df = _build_record_dataframe(records)
        if record_df.empty:
            st.info("Belum ada data report.")
            return

        st.markdown("<div class='pp-note'>Filter eksperimen dari kiri ke kanan, lalu pilih run yang ingin dieksplor.</div>", unsafe_allow_html=True)

        filter_c1, filter_c2, filter_c3, filter_c4 = st.columns(4)

        dataset_options = dataset_registry.list_dataset_ids()
        if not dataset_options:
            st.warning("Belum ada dataset terdaftar di registry.")
            return
        dataset_count_map = _counts_by_column(record_df, "dataset")
        selected_dataset = filter_c1.selectbox(
            "1) Dataset",
            dataset_options,
            index=0,
            key="report_dataset",
            format_func=lambda opt: _format_option_with_run_count(str(opt), dataset_count_map),
        )

        dataset_df = record_df[record_df["dataset"] == selected_dataset].copy()
        dataset_cfg = dataset_registry.get(selected_dataset)
        aug_options = dataset_cfg.augmentation_options or augmentation_registry.list_augmentation_ids()
        if not aug_options:
            st.warning("Belum ada augmentasi terdaftar.")
            return
        aug_count_map = _counts_by_column(dataset_df, "augmentation")
        selected_aug = filter_c2.selectbox(
            "2) Augmentasi",
            aug_options,
            index=0,
            key="report_augmentation",
            format_func=lambda opt: _format_option_with_run_count(str(opt), aug_count_map),
        )

        aug_df = dataset_df[dataset_df["augmentation"] == selected_aug].copy()
        method_options = method_registry.list_method_ids()
        if not method_options:
            st.warning("Belum ada method training terdaftar.")
            return
        method_count_map = _counts_by_column(aug_df, "method")
        selected_method = filter_c3.selectbox(
            "3) Method Training",
            method_options,
            index=0,
            key="report_method",
            format_func=lambda opt: _format_option_with_run_count(str(opt), method_count_map),
        )

        method_df = aug_df[aug_df["method"] == selected_method].copy()
        model_options = model_registry.list_model_ids(enabled_only=True)
        if not model_options:
            st.warning("Belum ada model aktif di registry.")
            return
        model_count_map = _counts_by_column(method_df, "model")
        selected_model = filter_c4.selectbox(
            "4) Model",
            model_options,
            index=0,
            key="report_model",
            format_func=lambda opt: _format_option_with_run_count(str(opt), model_count_map),
        )

        model_df = method_df[method_df["model"] == selected_model].copy()
        model_df = model_df.sort_values(by="run_started_at", ascending=False, na_position="last").reset_index(drop=True)
        if model_df.empty:
            st.warning(
                "Belum ada run untuk kombinasi terpilih: "
                f"`{selected_dataset}` / `{selected_aug}` / `{selected_method}` / `{selected_model}`."
            )
            return

        run_label_map = { _build_run_label(row): str(row.get("run_id") or "") for _, row in model_df.iterrows() }
        run_labels = list(run_label_map.keys())
        selected_run_label = st.selectbox("5) Hasil Training (Run)", run_labels, index=0, key="report_run")
        selected_run = run_label_map.get(selected_run_label, "")

        target_df = model_df[model_df["run_id"].astype(str) == str(selected_run)]
        if target_df.empty:
            st.error("Run record tidak ditemukan.")
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
            selected_record = target_df.iloc[0].to_dict()
            selected_record["accuracy"] = selected_record.get("test_accuracy")

        _render_run_detail(selected_record)

    with tab_comparison:
        _render_comparison_section(
            records=records,
            dataset_registry=dataset_registry,
            method_registry=method_registry,
            augmentation_registry=augmentation_registry,
            model_registry=model_registry,
        )


@st.cache_resource(show_spinner=False)
def _cached_load_model(run_dir_key: str):
    return load_model_bundle(Path(run_dir_key))


def render_prediction_tab(
    dataset_registry: DatasetRegistry,
    model_registry: ModelRegistry,
    augmentation_registry: TrainingAugmentationRegistry,
    method_registry: TrainingMethodRegistry,
) -> None:
    st.subheader("Prediksi Gambar")

    records = build_experiment_index(REPORT_ROOT)
    record_df = _build_record_dataframe(records)
    if record_df.empty:
        st.info("Belum ada model/report untuk prediksi.")
        return

    st.markdown("<div class='pp-note'>Pilih kombinasi dataset, augmentasi, method, lalu bandingkan prediksi beberapa model dalam satu input gambar.</div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)

    dataset_options = dataset_registry.list_dataset_ids()
    if not dataset_options:
        st.info("Belum ada dataset terdaftar.")
        return
    dataset_count_map = _counts_by_column(record_df, "dataset")
    selected_dataset = f1.selectbox(
        "Dataset model",
        dataset_options,
        index=0,
        key="predict_dataset",
        format_func=lambda opt: _format_option_with_run_count(str(opt), dataset_count_map),
    )

    dataset_df = record_df[record_df["dataset"] == selected_dataset].copy()
    dataset_cfg = dataset_registry.get(selected_dataset)
    aug_options = dataset_cfg.augmentation_options or augmentation_registry.list_augmentation_ids()
    if not aug_options:
        st.warning("Belum ada augmentasi terdaftar.")
        return
    aug_count_map = _counts_by_column(dataset_df, "augmentation")
    selected_aug = f2.selectbox(
        "Augmentasi",
        aug_options,
        index=0,
        key="predict_aug",
        format_func=lambda opt: _format_option_with_run_count(str(opt), aug_count_map),
    )

    aug_df = dataset_df[dataset_df["augmentation"] == selected_aug].copy()
    method_options = method_registry.list_method_ids()
    if not method_options:
        st.warning("Belum ada method training terdaftar.")
        return
    method_count_map = _counts_by_column(aug_df, "method")
    selected_method = f3.selectbox(
        "Method",
        method_options,
        index=0,
        key="predict_method",
        format_func=lambda opt: _format_option_with_run_count(str(opt), method_count_map),
    )

    scope_df = aug_df[aug_df["method"] == selected_method].copy()
    model_options = model_registry.list_model_ids(enabled_only=True)
    if not model_options:
        st.info("Belum ada model aktif di registry.")
        return

    model_count_map = _counts_by_column(scope_df, "model")
    available_models = [model_name for model_name in model_options if model_count_map.get(model_name, 0) > 0]
    st.caption(
        "Model terdaftar: {} | Model dengan run pada kombinasi ini: {}".format(
            len(model_options), len(available_models)
        )
    )
    selected_models = st.multiselect(
        "Pilih model untuk dibandingkan",
        options=model_options,
        default=available_models,
        format_func=lambda opt: _format_option_with_run_count(str(opt), model_count_map),
    )
    if not selected_models:
        st.warning("Pilih minimal 1 model.")
        return

    selected_runs: Dict[str, str] = {}
    with st.expander("Pilih run per model", expanded=True):
        for model_name in selected_models:
            model_df = scope_df[scope_df["model"] == model_name].copy()
            model_df = model_df.sort_values(by="run_started_at", ascending=False, na_position="last")
            if model_df.empty:
                st.info(f"{model_name}: belum ada run untuk kombinasi filter saat ini.")
                continue

            run_label_map = {_build_run_label(row): str(row.get("run_id") or "") for _, row in model_df.iterrows()}
            run_labels = list(run_label_map.keys())
            selected_label = st.selectbox(
                f"Run untuk {model_name}",
                run_labels,
                index=0,
                key=f"predict_run_{model_name}",
            )
            selected_runs[model_name] = run_label_map.get(selected_label, "")

    uploaded = st.file_uploader("Upload gambar (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload gambar untuk mulai prediksi.")
        return

    image_bytes = uploaded.getvalue()
    preview = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    c_left, c_right = st.columns((1, 2))
    with c_left:
        st.image(preview, caption="Input", use_container_width=True)
    with c_right:
        st.markdown("**Model aktif untuk prediksi**")
        model_run_rows = []
        for model_name in selected_models:
            model_run_rows.append(
                {
                    "model": model_name,
                    "run_id": selected_runs.get(model_name) or "-",
                }
            )
        _render_filterable_dataframe(
            pd.DataFrame(model_run_rows),
            key_prefix=f"predict_active_models_{selected_dataset}_{selected_aug}_{selected_method}",
            use_container_width=True,
            empty_message="Belum ada model aktif untuk ditampilkan.",
        )

    if st.button("Jalankan Prediksi", type="primary"):
        summary_rows = []
        prob_rows = []
        errors = []

        with st.spinner("Memuat model dan menjalankan inferensi..."):
            for model_name in selected_models:
                run_id = selected_runs.get(model_name)
                if not run_id:
                    errors.append(f"{model_name}: run tidak tersedia")
                    continue

                target = scope_df[
                    (scope_df["model"] == model_name)
                    & (scope_df["run_id"].astype(str) == str(run_id))
                ]
                if target.empty:
                    errors.append(f"{model_name}: run record tidak ditemukan")
                    continue

                record = target.iloc[0].to_dict()
                model_dir = _value_to_path(record.get("model_dir"))
                if model_dir is None:
                    errors.append(f"{model_name}: path model tidak tersedia di manifest")
                    continue

                if not model_dir.exists():
                    errors.append(f"{model_name}: folder model tidak ditemukan ({model_dir})")
                    continue

                try:
                    model_bundle = _cached_load_model(str(model_dir))
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
            summary_df = pd.DataFrame(summary_rows).sort_values(by="confidence", ascending=False).reset_index(drop=True)
            st.markdown("**Hasil Prediksi per Model**")
            _render_filterable_dataframe(
                summary_df,
                key_prefix=f"predict_summary_{selected_dataset}_{selected_aug}_{selected_method}",
                use_container_width=True,
                empty_message="Hasil prediksi belum tersedia.",
            )

        if prob_rows:
            prob_df = pd.DataFrame(prob_rows)
            st.markdown("**Perbandingan Probabilitas**")
            pivot_df = prob_df.pivot(index="class", columns="model", values="probability").fillna(0.0)
            st.bar_chart(pivot_df)
            _render_filterable_dataframe(
                prob_df,
                key_prefix=f"predict_prob_{selected_dataset}_{selected_aug}_{selected_method}",
                use_container_width=True,
                empty_message="Probabilitas prediksi belum tersedia.",
            )

        if errors:
            st.error("Sebagian model gagal dipakai:\n- " + "\n- ".join(errors))


def main() -> None:
    st.set_page_config(
        page_title="Parkinson Classification Dashboard",
        page_icon=":material/neurology:",
        layout="wide",
    )

    _inject_dashboard_styles()

    dataset_registry = DatasetRegistry()
    model_registry = ModelRegistry()
    augmentation_registry = TrainingAugmentationRegistry()
    method_registry = TrainingMethodRegistry()

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
        render_report_tab(
            dataset_registry=dataset_registry,
            model_registry=model_registry,
            augmentation_registry=augmentation_registry,
            method_registry=method_registry,
        )
    with tab_predict:
        render_prediction_tab(
            dataset_registry=dataset_registry,
            model_registry=model_registry,
            augmentation_registry=augmentation_registry,
            method_registry=method_registry,
        )


if __name__ == "__main__":
    main()
