from __future__ import annotations

import base64
import html
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.datasets.registry import DatasetRegistry
from src.datasets.validator import discover_class_directories, list_image_files
from src.models.registry import ModelRegistry
from src.reporting.report_reader import (
    build_experiment_index,
    build_latest_summary_table,
    safe_load_json,
)
from src.reporting.schemas import VISUAL_FILES
from src.training.augmentations import TrainingAugmentationRegistry
from src.training.strategies import TrainingMethodRegistry
from src.utils.paths import REPORT_ROOT

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_datetime_text(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    try:
        return datetime.fromisoformat(text).strftime("%d %b %Y %H:%M:%S")
    except Exception:
        return text


def _metric_text(value: Optional[float]) -> str:
    if value is None:
        return "NaN"
    try:
        if pd.isna(value):
            return "NaN"
    except Exception:
        pass
    return "{:.4f}".format(value)


def _duration_text(value: Optional[float]) -> str:
    if value is None:
        return "NaN"
    try:
        if pd.isna(value):
            return "NaN"
    except Exception:
        pass
    return "{:.2f}".format(value)


def _sanitize_token(value: str, fallback: str = "item") -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value).strip())
    token = token.strip("-")
    return token or fallback


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


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


def _build_record_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in records:
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
                "report_dir": item.get("report_dir") or item.get("run_dir"),
                "run_dir": item.get("run_dir") or item.get("report_dir"),
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


def _img_file_to_data_uri(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None

    suffix = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    mime = mime_map.get(suffix, "application/octet-stream")

    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def _fig_to_data_uri(fig: Any) -> Optional[str]:
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def _plot_series_bar(series: pd.Series, title: str) -> Optional[str]:
    if series.empty:
        return None
    try:
        fig, ax = plt.subplots(figsize=(7.8, 3.2))
        series.plot(kind="bar", ax=ax, color="#1c8f5a", edgecolor="#145e3c")
        ax.set_title(title)
        ax.set_xlabel("Kelas")
        ax.set_ylabel("Jumlah Gambar")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        return _fig_to_data_uri(fig)
    except Exception:
        return None


def _plot_split_grouped(split_df: pd.DataFrame, title: str) -> Optional[str]:
    if split_df.empty:
        return None
    plot_df = split_df.drop(columns=["total"], errors="ignore")
    if plot_df.empty:
        return None
    try:
        fig, ax = plt.subplots(figsize=(7.8, 3.8))
        plot_df.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Split")
        ax.set_ylabel("Jumlah Gambar")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        return _fig_to_data_uri(fig)
    except Exception:
        return None


def _df_to_html(df: pd.DataFrame, empty_text: str = "Tidak ada data") -> str:
    if df.empty:
        return f"<p class='empty'>{html.escape(empty_text)}</p>"

    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}".rstrip("0").rstrip("."))

    try:
        return display_df.to_html(index=False, classes="table", border=0, justify="left", escape=True)
    except Exception:
        return "<p class='empty'>Gagal merender tabel.</p>"


def _kv_table_html(items: List[Tuple[str, str]]) -> str:
    rows = []
    for key, value in items:
        rows.append(
            "<tr><th>{}</th><td>{}</td></tr>".format(
                html.escape(str(key)),
                html.escape(str(value)),
            )
        )
    return "<table class='table kv'><tbody>{}</tbody></table>".format("".join(rows))


def _build_styles() -> str:
    return """
<style>
:root {
  --ink: #103325;
  --ink-soft: #416657;
  --line: #d5e5dd;
  --muted-bg: #f1f8f4;
  --card: #ffffff;
  --accent: #1c8f5a;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  color: var(--ink);
  font-family: "Segoe UI", "Calibri", Arial, sans-serif;
  background: #ffffff;
}
body {
  margin: 14px auto;
  max-width: 1120px;
  line-height: 1.45;
  padding: 0 12px 28px;
}
h1, h2, h3, h4 { margin: 0.2rem 0 0.5rem; }
h1 { font-size: 1.85rem; color: #0f4f31; }
h2 { font-size: 1.35rem; color: #0f4f31; border-bottom: 2px solid var(--line); padding-bottom: 5px; margin-top: 24px; }
h3 { font-size: 1.08rem; color: #155f3d; margin-top: 18px; }
h4 { font-size: 0.98rem; color: #1c6f47; margin-top: 14px; }
.small { font-size: 0.88rem; color: var(--ink-soft); }
.note {
  background: #eaf6f0;
  border: 1px solid #c9e7d8;
  border-radius: 10px;
  padding: 10px 12px;
  margin: 8px 0 14px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 10px;
}
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin: 8px 0 12px;
}
.kpi {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px;
  background: #fbfefd;
}
.kpi .label { font-size: 0.78rem; color: var(--ink-soft); margin-bottom: 3px; }
.kpi .value { font-size: 1.02rem; font-weight: 700; color: var(--ink); }
.table {
  width: 100%;
  border-collapse: collapse;
  margin: 6px 0 12px;
  font-size: 0.86rem;
}
.table th, .table td {
  border: 1px solid var(--line);
  padding: 6px 8px;
  text-align: left;
  vertical-align: top;
}
.table th {
  background: #edf8f2;
  color: #154c33;
  font-weight: 700;
}
.table tr:nth-child(even) td { background: #fcfffd; }
.table.kv th { width: 220px; }
.empty {
  color: #7d8f86;
  font-style: italic;
  margin: 6px 0 12px;
}
.section {
  border: 1px solid var(--line);
  border-radius: 10px;
  margin: 10px 0;
  padding: 10px;
  background: #ffffff;
}
.run-box {
  border: 1px solid #bfdcca;
  border-radius: 10px;
  padding: 10px;
  margin: 14px 0;
  background: #fcfffd;
  break-inside: avoid-page;
}
.figure {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 6px;
  background: #fff;
  margin: 8px 0 12px;
}
.figure img {
  width: 100%;
  height: auto;
  display: block;
}
.figure figcaption {
  font-size: 0.8rem;
  color: var(--ink-soft);
  margin-top: 5px;
}
.code {
  font-family: Consolas, "Courier New", monospace;
  font-size: 0.82rem;
  color: #183a2a;
  background: #f4fbf7;
  border: 1px solid #d6eadf;
  border-radius: 6px;
  padding: 6px;
  white-space: pre-wrap;
  word-break: break-word;
}
ul { margin-top: 6px; }
li { margin-bottom: 3px; }
hr.sep { border: 0; border-top: 1px solid var(--line); margin: 18px 0; }
.print-button-wrap { margin: 8px 0 10px; }
.print-button {
  padding: 7px 12px;
  border: 1px solid #1a6f47;
  background: #1c8f5a;
  color: #fff;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.9rem;
}
@page {
  size: A4 portrait;
  margin: 11mm;
}
@media print {
  body { max-width: none; margin: 0; padding: 0; }
  .print-button-wrap { display: none !important; }
  h1, h2, h3 { break-after: avoid-page; }
  .section, .run-box, .card, .figure { break-inside: avoid-page; }
  table { page-break-inside: auto; }
  tr, td, th { page-break-inside: avoid; }
  thead { display: table-header-group; }
}
</style>
"""


def _build_glossary_html(
    dataset_registry: DatasetRegistry,
    augmentation_registry: TrainingAugmentationRegistry,
    method_registry: TrainingMethodRegistry,
    model_registry: ModelRegistry,
) -> str:
    dataset_rows = []
    for dataset_id in dataset_registry.list_dataset_ids():
        cfg = dataset_registry.get(dataset_id)
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "deskripsi": cfg.description,
                "class_mode": cfg.class_mode,
                "split": "train={} / testing={} / validation={}".format(
                    cfg.split.get("train"), cfg.split.get("testing"), cfg.split.get("validation")
                ),
                "opsi_augmentasi": ", ".join(cfg.augmentation_options or []),
            }
        )

    aug_rows = []
    for aug_id in augmentation_registry.list_augmentation_ids():
        aug = augmentation_registry.get(aug_id)
        aug_rows.append(
            {
                "augmentation_id": aug_id,
                "label": aug.label,
                "deskripsi": aug.description,
                "disable_augmentation": aug.disable_augmentation,
            }
        )

    method_rows = []
    for method_id in method_registry.list_method_ids():
        method = method_registry.get(method_id)
        overrides_text = ", ".join(f"{k}={v}" for k, v in method.arg_overrides.items())
        method_rows.append(
            {
                "method_id": method_id,
                "deskripsi": method.description,
                "arg_overrides": overrides_text,
            }
        )

    model_rows = []
    for model_id in model_registry.list_model_ids(enabled_only=False):
        model = model_registry.get(model_id)
        model_rows.append(
            {
                "model_id": model_id,
                "display_name": model.display_name,
                "framework": model.framework,
                "preprocess_key": model.preprocess_key,
                "enabled": model.enabled,
            }
        )

    dictionary_rows = [
        {"istilah": "run_id", "arti_non_teknis": "Kode unik untuk satu sesi training. Setiap training baru akan menghasilkan run_id baru."},
        {"istilah": "experiment_id", "arti_non_teknis": "ID eksperimen gabungan (dataset + augmentasi + method + model)."},
        {"istilah": "train_accuracy", "arti_non_teknis": "Akurasi model pada data latih (data yang dipakai belajar)."},
        {"istilah": "val_accuracy", "arti_non_teknis": "Akurasi model pada data validasi (data cek selama training)."},
        {"istilah": "test_accuracy", "arti_non_teknis": "Akurasi akhir pada data uji yang tidak dipakai belajar."},
        {"istilah": "f1_score", "arti_non_teknis": "Skor seimbang antara ketepatan prediksi positif dan kemampuan menemukan kasus positif."},
        {"istilah": "train_loss / val_loss", "arti_non_teknis": "Besarnya kesalahan model. Semakin kecil umumnya semakin baik."},
        {"istilah": "baseline", "arti_non_teknis": "Metode dasar tanpa transfer bobot model pra-latih."},
        {"istilah": "transfer_learning", "arti_non_teknis": "Melatih model dengan memanfaatkan pengetahuan awal dari model yang sudah dilatih di data besar."},
        {"istilah": "transfer_learning_mixed_precision", "arti_non_teknis": "Transfer learning dengan teknik komputasi campuran (lebih cepat/hemat memori pada GPU kompatibel)."},
        {"istilah": "full_fine_tuning", "arti_non_teknis": "Transfer learning dengan penyesuaian parameter model lebih dalam/lebih lama."},
        {"istilah": "no_augment", "arti_non_teknis": "Training tanpa augmentasi tambahan pada gambar train."},
        {"istilah": "augment_on_the_fly", "arti_non_teknis": "Augmentasi gambar dilakukan saat proses training (mis. rotasi/transformasi ringan) untuk menambah variasi data."},
        {"istilah": "fine_tune_epochs", "arti_non_teknis": "Jumlah epoch tahap fine-tuning (penyempurnaan model setelah tahap awal)."},
        {"istilah": "batch_size", "arti_non_teknis": "Jumlah gambar yang diproses sekaligus pada satu langkah training."},
    ]

    process_notes = """
<ul>
  <li><b>Alur ringkas pipeline:</b> cek dataset -> split train/testing/validation -> training per kombinasi -> evaluasi -> simpan report + model.</li>
  <li><b>Balancing kelas:</b> saat split, sistem dapat menambah kelas minoritas agar distribusi kelas lebih seimbang.</li>
  <li><b>Perbandingan eksperimen:</b> laporan ini menyajikan seluruh run yang pernah tersimpan di folder report, tanpa filter.</li>
  <li><b>Cara membaca angka:</b> fokus utama biasanya pada <code>val_accuracy</code>, <code>test_accuracy</code>, dan <code>f1_score</code>; loss dipakai untuk melihat stabilitas proses belajar.</li>
</ul>
"""

    return """
<h2>Kamus Data dan Catatan Non-IT</h2>
<div class='note'>Bagian ini menjelaskan istilah teknis agar pembaca non-IT dapat memahami isi laporan eksperimen.</div>
<h3>Istilah Utama</h3>
{dictionary_table}
<h3>Definisi Dataset</h3>
{dataset_table}
<h3>Definisi Opsi Augmentasi</h3>
{aug_table}
<h3>Definisi Method Training</h3>
{method_table}
<h3>Definisi Model</h3>
{model_table}
<h3>Catatan Proses</h3>
{process_notes}
""".format(
        dictionary_table=_df_to_html(pd.DataFrame(dictionary_rows), "Kamus data tidak tersedia."),
        dataset_table=_df_to_html(pd.DataFrame(dataset_rows), "Tidak ada dataset terdaftar."),
        aug_table=_df_to_html(pd.DataFrame(aug_rows), "Tidak ada opsi augmentasi terdaftar."),
        method_table=_df_to_html(pd.DataFrame(method_rows), "Tidak ada method training terdaftar."),
        model_table=_df_to_html(pd.DataFrame(model_rows), "Tidak ada model terdaftar."),
        process_notes=process_notes,
    )


def _build_dataset_section(dataset_registry: DatasetRegistry) -> str:
    sections: List[str] = ["<h2>Ringkasan Dataset (Semua Dataset Terdaftar)</h2>"]

    for dataset_id in dataset_registry.list_dataset_ids():
        cfg = dataset_registry.get(dataset_id)

        try:
            classes = discover_class_directories(
                dataset_root=cfg.original_path,
                class_mode=cfg.class_mode,
                extensions=cfg.valid_extensions,
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
        except Exception:
            original_df = pd.DataFrame()

        split_distribution = _count_split_distribution(cfg.split_path)
        split_df = _build_split_table(split_distribution)

        original_chart_uri = None
        if not original_df.empty and "class_name" in original_df.columns:
            original_chart_uri = _plot_series_bar(
                original_df.set_index("class_name")["count"],
                f"Distribusi Dataset Original - {dataset_id}",
            )

        split_chart_uri = _plot_split_grouped(split_df, f"Distribusi Split - {dataset_id}") if not split_df.empty else None

        summary_items = [
            ("Dataset ID", dataset_id),
            ("Deskripsi", cfg.description or "-"),
            ("Class Mode", cfg.class_mode),
            ("Original Dir", str(cfg.original_path)),
            ("Split Dir", str(cfg.split_path)),
            ("Augmentasi yang Diizinkan", ", ".join(cfg.augmentation_options or [])),
            (
                "Split Ratio",
                "train={} | testing={} | validation={}".format(
                    cfg.split.get("train"), cfg.split.get("testing"), cfg.split.get("validation")
                ),
            ),
        ]

        total_original = int(original_df["count"].sum()) if not original_df.empty else 0
        class_count = int(len(original_df)) if not original_df.empty else 0
        train_total = int(split_df.loc["train", "total"]) if not split_df.empty and "train" in split_df.index else 0
        test_total = int(split_df.loc["testing", "total"]) if not split_df.empty and "testing" in split_df.index else 0
        val_total = int(split_df.loc["validation", "total"]) if not split_df.empty and "validation" in split_df.index else 0

        sections.append(
            """
<div class='section'>
  <h3>Dataset: {dataset_id}</h3>
  {summary_table}
  <div class='kpi-grid'>
    <div class='kpi'><div class='label'>Jumlah Kelas</div><div class='value'>{class_count}</div></div>
    <div class='kpi'><div class='label'>Total Gambar Original</div><div class='value'>{total_original}</div></div>
    <div class='kpi'><div class='label'>Split Train</div><div class='value'>{train_total}</div></div>
    <div class='kpi'><div class='label'>Split Testing</div><div class='value'>{test_total}</div></div>
    <div class='kpi'><div class='label'>Split Validation</div><div class='value'>{val_total}</div></div>
  </div>
  <h4>Tabel Distribusi Original</h4>
  {original_table}
  {original_chart}
  <h4>Tabel Distribusi Split</h4>
  {split_table}
  {split_chart}
</div>
""".format(
                dataset_id=html.escape(dataset_id),
                summary_table=_kv_table_html(summary_items),
                class_count=class_count,
                total_original=total_original,
                train_total=train_total,
                test_total=test_total,
                val_total=val_total,
                original_table=_df_to_html(original_df, "Distribusi dataset original tidak tersedia."),
                original_chart=(
                    "<figure class='figure'><img src='{}' alt='chart-original'><figcaption>Grafik distribusi dataset original.</figcaption></figure>".format(original_chart_uri)
                    if original_chart_uri
                    else ""
                ),
                split_table=_df_to_html(split_df, "Distribusi split belum tersedia."),
                split_chart=(
                    "<figure class='figure'><img src='{}' alt='chart-split'><figcaption>Grafik distribusi train/testing/validation.</figcaption></figure>".format(split_chart_uri)
                    if split_chart_uri
                    else ""
                ),
            )
        )

    return "\n".join(sections)


def _build_comparison_sections(
    record_df: pd.DataFrame,
    dataset_registry: DatasetRegistry,
    method_registry: TrainingMethodRegistry,
    augmentation_registry: TrainingAugmentationRegistry,
    model_registry: ModelRegistry,
) -> str:
    sections = ["<h2>Perbandingan Eksperimen (Semua Data)</h2>"]

    if record_df.empty:
        sections.append("<p class='empty'>Belum ada data report training.</p>")
        return "\n".join(sections)

    global_summary = _latest_per_key(record_df, ["dataset", "augmentation", "method", "model"])
    sections.append("<h3>Ringkasan Global Run Terbaru per Kombinasi</h3>")
    sections.append(_df_to_html(global_summary, "Ringkasan global belum tersedia."))

    for dataset_id in dataset_registry.list_dataset_ids():
        subset_df = record_df[record_df["dataset"] == dataset_id].copy()
        if subset_df.empty:
            sections.append(f"<h3>Dataset: {html.escape(dataset_id)}</h3><p class='empty'>Belum ada run untuk dataset ini.</p>")
            continue

        sections.append(f"<h3>Dataset: {html.escape(dataset_id)}</h3>")

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
        )
        method_summary = (
            method_summary.set_index("method")
            .reindex(method_registry.list_method_ids())
            .reset_index()
            .rename(columns={"index": "method"})
        )
        if "runs" in method_summary.columns:
            method_summary["runs"] = method_summary["runs"].fillna(0).astype(int)
        method_summary = method_summary.sort_values(by="avg_val_accuracy", ascending=False, na_position="last")

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
        )
        dataset_cfg = dataset_registry.get(dataset_id)
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

        by_model = _latest_per_key(subset_df, ["dataset", "augmentation", "method", "model"])
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
        model_summary = (
            model_summary.set_index("model")
            .reindex(model_registry.list_model_ids(enabled_only=True))
            .reset_index()
            .rename(columns={"index": "model"})
        )
        if "runs" in model_summary.columns:
            model_summary["runs"] = model_summary["runs"].fillna(0).astype(int)
        model_summary = model_summary.sort_values(by="avg_val_accuracy", ascending=False, na_position="last")

        ranking = by_model.sort_values(by="val_accuracy", ascending=False, na_position="last").reset_index(drop=True)
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

        sections.append("<div class='section'>")
        sections.append("<h4>Perbandingan Method</h4>")
        sections.append(_df_to_html(method_summary, "Belum ada data method."))
        sections.append("<h4>Perbandingan Model</h4>")
        sections.append(_df_to_html(model_summary, "Belum ada data model."))
        sections.append("<h4>Perbandingan Augmentasi</h4>")
        sections.append(_df_to_html(aug_summary, "Belum ada data augmentasi."))
        sections.append("<h4>Ranking Run Terbaik (berdasarkan val_accuracy)</h4>")
        sections.append(_df_to_html(ranking_df, "Belum ada ranking run."))
        sections.append("</div>")

    return "\n".join(sections)


def _build_prediction_section(record_df: pd.DataFrame) -> str:
    parts = ["<h2>Ringkasan Kesiapan Prediksi</h2>"]
    if record_df.empty:
        parts.append("<p class='empty'>Belum ada model/report untuk prediksi.</p>")
        return "\n".join(parts)

    pred_df = record_df[["dataset", "augmentation", "method", "model", "run_id", "model_dir", "run_started_at"]].copy()
    pred_df = pred_df.sort_values(by=["dataset", "augmentation", "method", "model", "run_started_at"], ascending=[True, True, True, True, False])

    latest_pred = _latest_per_key(pred_df, ["dataset", "augmentation", "method", "model"])
    latest_pred = latest_pred[["dataset", "augmentation", "method", "model", "run_id", "model_dir", "run_started_at"]]

    parts.append("<h3>Model Terbaru per Kombinasi untuk Prediksi</h3>")
    parts.append(_df_to_html(latest_pred, "Belum ada model prediksi."))

    parts.append("<h3>Daftar Semua Kandidat Run Prediksi</h3>")
    parts.append(_df_to_html(pred_df, "Belum ada kandidat run prediksi."))
    return "\n".join(parts)


def _build_run_detail_section(records: List[Dict[str, Any]], include_embedded_run_images: bool) -> str:
    parts: List[str] = ["<h2>Detail Seluruh Run Training</h2>"]
    if not records:
        parts.append("<p class='empty'>Belum ada run training.</p>")
        return "\n".join(parts)

    for idx, record in enumerate(records, start=1):
        dataset = str(record.get("dataset") or "-")
        augmentation = str(record.get("augmentation") or "-")
        method = str(record.get("method") or "-")
        model = str(record.get("model") or "-")
        run_id = str(record.get("run_id") or "-")
        run_dir = Path(str(record.get("run_dir") or record.get("report_dir") or ""))

        manifest = safe_load_json(run_dir / "run_manifest.json") or {}
        training = manifest.get("training", {}) if isinstance(manifest, dict) else {}
        params = training.get("parameters", {}) if isinstance(training, dict) else {}
        if not isinstance(params, dict):
            params = {}

        epochs_value = _safe_int(params.get("epochs") or record.get("epochs"))
        batch_size_value = _safe_int(params.get("batch_size") or record.get("batch_size"))
        fine_tune_epochs_value = _safe_int(params.get("fine_tune_epochs") or record.get("fine_tune_epochs"))

        identity_table = _kv_table_html(
            [
                ("No.", str(idx)),
                ("Experiment ID", str(record.get("experiment_id") or "-")),
                ("Dataset", dataset),
                ("Augmentasi", augmentation),
                ("Method", method),
                ("Model", model),
                ("Run ID", run_id),
                ("Run Started", _safe_datetime_text(record.get("run_started_at"))),
                ("Run Finished", _safe_datetime_text(record.get("run_finished_at"))),
                ("Epoch Stage-1", str(epochs_value) if epochs_value is not None else "-"),
                ("Batch Size", str(batch_size_value) if batch_size_value is not None else "-"),
                ("Fine-tune Epochs", str(fine_tune_epochs_value) if fine_tune_epochs_value is not None else "-"),
                ("Train Accuracy", _metric_text(_safe_float(record.get("train_accuracy")))),
                ("Validation Accuracy", _metric_text(_safe_float(record.get("val_accuracy")))),
                ("Train Loss", _metric_text(_safe_float(record.get("train_loss")))),
                ("Validation Loss", _metric_text(_safe_float(record.get("val_loss")))),
                ("Test Accuracy", _metric_text(_safe_float(record.get("accuracy") or record.get("test_accuracy")))),
                ("F1 Score", _metric_text(_safe_float(record.get("f1_score")))),
                ("Training Time (s)", _duration_text(_safe_float(record.get("training_time_seconds")))),
                ("Run Report Dir", str(run_dir)),
                ("Model Dir", str(record.get("model_dir") or "-")),
                ("Final Model", str(record.get("final_model_path") or "-")),
                ("Best Model", str(record.get("best_model_path") or "-")),
            ]
        )

        history_df = _safe_read_csv(run_dir / "training_history.csv")
        cls_df = _safe_read_csv(run_dir / "classification_report.csv")
        cm_df = _safe_read_csv(run_dir / "confusion_matrix.csv")
        split_df = _safe_read_csv(run_dir / "split_distribution.csv")
        eval_df = _safe_read_csv(run_dir / "evaluation_metrics.csv")
        summary_txt = _safe_read_text(run_dir / "summary.txt")

        visuals_html_parts: List[str] = []
        available_visual_names: List[str] = []
        for visual_name in VISUAL_FILES:
            img_path = run_dir / visual_name
            if not img_path.exists():
                continue
            available_visual_names.append(visual_name)

            if include_embedded_run_images:
                img_uri = _img_file_to_data_uri(img_path)
                if img_uri:
                    visuals_html_parts.append(
                        "<figure class='figure'><img src='{}' alt='{}'><figcaption>{}</figcaption></figure>".format(
                            img_uri,
                            html.escape(visual_name),
                            html.escape(visual_name),
                        )
                    )

        params_html = "<pre class='code'>{}</pre>".format(html.escape(str(params))) if params else "<p class='empty'>Parameter tidak tersedia.</p>"
        summary_html = "<pre class='code'>{}</pre>".format(html.escape(summary_txt)) if summary_txt.strip() else "<p class='empty'>summary.txt tidak tersedia.</p>"

        parts.append("<div class='run-box'>")
        parts.append(f"<h3>Run #{idx}: {html.escape(dataset)} / {html.escape(augmentation)} / {html.escape(method)} / {html.escape(model)} / {html.escape(run_id)}</h3>")
        parts.append(identity_table)
        parts.append("<h4>Parameter Training</h4>")
        parts.append(params_html)
        parts.append("<h4>Ringkasan Teks (summary.txt)</h4>")
        parts.append(summary_html)
        parts.append("<h4>Tabel Evaluation Metrics</h4>")
        parts.append(_df_to_html(eval_df, "evaluation_metrics.csv tidak tersedia."))
        parts.append("<h4>Tabel Training History</h4>")
        parts.append(_df_to_html(history_df, "training_history.csv tidak tersedia."))
        parts.append("<h4>Tabel Classification Report</h4>")
        parts.append(_df_to_html(cls_df, "classification_report.csv tidak tersedia."))
        parts.append("<h4>Tabel Confusion Matrix</h4>")
        parts.append(_df_to_html(cm_df, "confusion_matrix.csv tidak tersedia."))
        parts.append("<h4>Tabel Split Distribution</h4>")
        parts.append(_df_to_html(split_df, "split_distribution.csv tidak tersedia."))
        parts.append("<h4>Visualisasi Gambar dari Artifact Run</h4>")
        if include_embedded_run_images:
            if visuals_html_parts:
                parts.extend(visuals_html_parts)
            else:
                parts.append("<p class='empty'>Belum ada visualisasi PNG pada run ini.</p>")
        else:
            if available_visual_names:
                visual_rows: List[Dict[str, str]] = []
                for visual_name in available_visual_names:
                    visual_rows.append(
                        {
                            "visual_file": visual_name,
                            "lokasi_file": str(run_dir / visual_name),
                            "catatan": "Mode ringkas: gambar tidak di-embed agar ukuran file lebih ringan.",
                        }
                    )
                parts.append(_df_to_html(pd.DataFrame(visual_rows), "Daftar visual tidak tersedia."))
            else:
                parts.append("<p class='empty'>Belum ada visualisasi PNG pada run ini.</p>")
        parts.append("</div>")

    return "\n".join(parts)


def build_full_html_report(
    dataset_registry: DatasetRegistry,
    model_registry: ModelRegistry,
    augmentation_registry: TrainingAugmentationRegistry,
    method_registry: TrainingMethodRegistry,
    report_root: Path = REPORT_ROOT,
    report_mode: str = "full",
    return_html: bool = True,
) -> Tuple[str, Path]:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized_mode = str(report_mode).strip().lower()
    if normalized_mode not in {"full", "compact"}:
        normalized_mode = "full"
    mode_label = "Lengkap" if normalized_mode == "full" else "Ringkas"
    include_embedded_run_images = normalized_mode == "full"

    records = build_experiment_index(report_root)
    latest_rows = build_latest_summary_table(report_root)
    latest_df = pd.DataFrame(latest_rows)
    record_df = _build_record_dataframe(records)

    total_runs = len(records)
    total_datasets = len(dataset_registry.list_dataset_ids())
    total_methods = len(method_registry.list_method_ids())
    total_models_enabled = len(model_registry.list_model_ids(enabled_only=True))

    header_summary = """
<div class='note'>
  <b>Tujuan laporan:</b> dokumen komprehensif semua hasil training yang tersimpan pada aplikasi. Laporan ini tidak memakai filter, sehingga seluruh item ditampilkan.
  <br><b>Mode laporan:</b> {mode_label} ({mode_note})
</div>
<div class='kpi-grid'>
  <div class='kpi'><div class='label'>Waktu Generate</div><div class='value'>{generated_at}</div></div>
  <div class='kpi'><div class='label'>Total Run Training</div><div class='value'>{total_runs}</div></div>
  <div class='kpi'><div class='label'>Dataset Terdaftar</div><div class='value'>{total_datasets}</div></div>
  <div class='kpi'><div class='label'>Method Terdaftar</div><div class='value'>{total_methods}</div></div>
  <div class='kpi'><div class='label'>Model Aktif</div><div class='value'>{total_models_enabled}</div></div>
</div>
""".format(
        generated_at=html.escape(generated_at),
        total_runs=total_runs,
        total_datasets=total_datasets,
        total_methods=total_methods,
        total_models_enabled=total_models_enabled,
        mode_label=html.escape(mode_label),
        mode_note=(
            "semua gambar artifact run di-embed ke HTML"
            if include_embedded_run_images
            else "gambar artifact run tidak di-embed (lebih ringan)"
        ),
    )

    latest_summary_html = """
<h2>Ringkasan Run Terbaru per Kombinasi</h2>
{latest_table}
""".format(latest_table=_df_to_html(latest_df, "Belum ada ringkasan run terbaru."))

    glossary_html = _build_glossary_html(
        dataset_registry=dataset_registry,
        augmentation_registry=augmentation_registry,
        method_registry=method_registry,
        model_registry=model_registry,
    )
    dataset_html = _build_dataset_section(dataset_registry)
    comparison_html = _build_comparison_sections(
        record_df=record_df,
        dataset_registry=dataset_registry,
        method_registry=method_registry,
        augmentation_registry=augmentation_registry,
        model_registry=model_registry,
    )
    prediction_html = _build_prediction_section(record_df)
    run_detail_html = _build_run_detail_section(
        records,
        include_embedded_run_images=include_embedded_run_images,
    )

    generated_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = report_root / "_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = "full_training_report" if normalized_mode == "full" else "compact_training_report"
    export_path = export_dir / f"{filename_prefix}_{generated_ts}.html"

    html_doc = """
<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Laporan {mode_label} Training Parkinson - {generated_at}</title>
  {styles}
  <script>
    function startPrint() {{
      try {{
        window.focus();
        window.print();
      }} catch (e) {{
        console.log(e);
      }}
    }}
    window.addEventListener('load', function() {{
      setTimeout(function() {{
        startPrint();
      }}, 450);
    }});
  </script>
</head>
<body>
  <div class="print-button-wrap">
    <button class="print-button" onclick="startPrint()">Print Sekarang</button>
  </div>
  <h1>Laporan {mode_label} Hasil Training Parkinson</h1>
  <p class="small">Dokumen ini digenerate otomatis dari artifact aplikasi pada {generated_at}. Semua item ditampilkan tanpa filter.</p>
  {header_summary}
  {glossary_html}
  {dataset_html}
  {latest_summary_html}
  {comparison_html}
  {prediction_html}
  {run_detail_html}
  <hr class="sep">
  <p class="small">Selesai. Dokumen ini dibuat otomatis oleh modul report generator aplikasi.</p>
</body>
</html>
""".format(
        generated_at=html.escape(generated_at),
        mode_label=html.escape(mode_label),
        styles=_build_styles(),
        header_summary=header_summary,
        glossary_html=glossary_html,
        dataset_html=dataset_html,
        latest_summary_html=latest_summary_html,
        comparison_html=comparison_html,
        prediction_html=prediction_html,
        run_detail_html=run_detail_html,
    )

    export_path.write_text(html_doc, encoding="utf-8")
    if return_html:
        return html_doc, export_path
    return "", export_path
