# Dokumentasi Fitur Baru

> Update: 2026-06-03. Dokumen ini melengkapi `architecture.md` dan `pipeline.md`
> untuk fitur yang baru ditambahkan: modul optimizer, split dinamis, dan
> penyempurnaan dashboard Streamlit (laporan, perbandingan overlay, download
> berfilter, serta training via web).

---

## 1. Modul Optimizer (`src/optimizer/`)

### Tujuan
Menyediakan pemilihan optimizer yang seragam dan dapat dipanggil dari config,
CLI, maupun UI Streamlit.

### Struktur
```
src/optimizer/
├─ __init__.py        # get_optimizer(), list_optimizers(), peta YOLO
├─ adam.py            # build_adam() -> tf.keras.optimizers.Adam
└─ no_optimize.py     # build_no_optimize() -> SGD plain (baseline)
```

### Antarmuka
```python
from src.optimizer import get_optimizer, list_optimizers

list_optimizers()                       # ['adam', 'no_optimize']
opt = get_optimizer("adam", learning_rate=1e-3)
model.compile(optimizer=opt, loss=..., metrics=...)
```

Catatan desain (asumsi eksplisit): pipeline ini berbasis Keras (define-and-compile),
sehingga optimizer dibangun dari `learning_rate` + konfigurasi, bukan dari
`model.parameters()` ala PyTorch. Karena itu signature-nya
`get_optimizer(name, learning_rate, **cfg)`.

### Opsi optimizer
| Nama | Implementasi | Kegunaan |
|---|---|---|
| `adam` | `tf.keras.optimizers.Adam` | Default adaptif, rekomendasi. |
| `no_optimize` | `tf.keras.optimizers.SGD` (momentum=0, nesterov=False) | Baseline polos tanpa tuning sebagai pembanding. |

### Cara memilih
- **Config**: `configs/default_training.yaml` -> `optimizer: adam`.
- **CLI**: `python training/train.py ... --optimizer no_optimize`.
- **Worker langsung**: `--optimizer adam` (TF) atau YOLO (dipetakan ke `Adam`/`SGD`).
- **Web**: dropdown "Optimizer" pada tab Training.

### YOLOv8
Untuk YOLO, optimizer project dipetakan ke optimizer Ultralytics:
`adam -> Adam`, `no_optimize -> SGD` (lihat `get_yolo_optimizer_name`).

---

## 2. Split Data Dinamis

### Preset
Terdaftar di `src/datasets/splitter.py` (`SPLIT_PRESETS`):
- `80-10-10` (train/testing/validation)
- `70-15-15`

### Aturan validasi (`validate_split_ratios`)
1. Setiap rasio > 0.
2. Total = 100% (toleransi kecil).
3. `testing == validation` (kebijakan split simetris project).

### Cara pakai
```bash
# Lewat script split
python training/2.split_data_testing.py --dataset parkinson_merder --split-preset 70-15-15 --on-existing-split resplit

# Rasio manual (harus testing == validation, total = 1.0)
python training/2.split_data_testing.py --dataset parkinson_merder --train-ratio 0.7 --test-ratio 0.15 --val-ratio 0.15

# Lewat orchestrator saat split-first
python training/train.py --dataset parkinson_merder --models mobilenetv2 --split-first --split-preset 80-10-10 --on-existing-split resplit
```
Seed tetap dikunci (default 42) agar hasil reproducible.

---

## 3. Dashboard Streamlit — Tab Baru & Penyempurnaan

Dashboard kini memiliki 4 tab: **Dataset**, **Training**, **Training Report**, **Prediksi**.

### 3.1 Tab Training (baru, T7)
Menjalankan training langsung dari web. Memanggil `training/train.py` di belakang
layar dan menampilkan log real-time.

Pilihan yang tersedia:
- Dataset, Model (multi), Method, Optimizer, Augmentasi.
- Hyperparameter: epoch, batch size, learning rate, fine-tune epoch, seed.
- Split: opsi `--split-first` + preset (80-10-10 / 70-15-15).

Catatan: karena web bersifat non-interaktif, mode `--on-existing retrain`
dipakai agar training benar-benar berjalan (run lama tetap tersimpan sebagai histori).
Hasil ringkas muncul di akhir; detail metrik/grafik dilihat di tab Training Report.

### 3.2 Tab Training Report (disempurnakan)
- **Metrik lengkap (T3)**: selain akurasi & F1, kini menampilkan **ROC-AUC**,
  **Precision (macro)**, dan **Recall (macro)** dari `classification_report.csv`.
- **Kurva training** (loss/accuracy per epoch), confusion matrix, classification
  report, dan distribusi split.
- **Ringkasan konfigurasi** run (epoch, batch, fine-tune) dari manifest.
- **Download berfilter (T5)**: panel "Download Report Berfilter" memungkinkan
  filter dataset/method/augmentasi/**model spesifik** + rentang tanggal, lalu
  unduh **CSV** atau **HTML printable (A4 -> Save as PDF)**.

### 3.3 Tab Perbandingan — Overlay Model (baru, T4)
Sub-tab "Overlay Model" memungkinkan memilih ≥2 run lintas model/method/augmentasi,
lalu menampilkan:
- Tabel metrik berdampingan (val/test accuracy, F1, ROC-AUC, loss, waktu training).
- Grafik **overlay** kurva training (pilih metrik: val_accuracy/accuracy/val_loss/loss).

---

## 4. Penyempurnaan Pipeline Training (rekomendasi diterapkan)

Lihat detail di `analisis_training_rekomendasi.md`. Ringkas:
- Default epoch & patience dinaikkan; LR fine-tune diperbesar (config YAML).
- Callback diselaraskan ke `val_accuracy` dan dibuat baru per stage.
- `class_weight` 'balanced' diterapkan pada `model.fit`.
- Output layer `dtype="float32"` (stabil saat mixed precision).
- YOLO: cosine LR + warmup + label smoothing.

### Ditunda (sesuai keputusan)
- **Transformer** (ViT/Swin/DeiT): perbaikan strategi pretrained / resep khusus
  (REC-09) **ditunda**. Placeholder/TODO terdokumentasi agar mudah dilanjutkan.

---

## 5. Reproducibility & Cara Menjalankan

- Seed global dikunci (default 42) di seluruh worker.
- Dependency tidak berubah (tetap `requirements.txt`); fitur baru hanya memakai
  pustaka yang sudah ada (TensorFlow, Streamlit, pandas, PIL).

```bash
# 1) Aktifkan environment (lihat documentation/requirements.md)
# 2) Split (opsional, dengan preset dinamis)
python training/2.split_data_testing.py --dataset parkinson_merder --split-preset 80-10-10 --on-existing-split resplit

# 3) Training via CLI (contoh optimizer no_optimize sebagai baseline)
python training/train.py --dataset parkinson_merder --models mobilenetv2 --method transfer_learning --optimizer adam --split-first --split-preset 80-10-10 --on-existing retrain

# 4) Dashboard (Dataset / Training / Report / Prediksi)
streamlit run web/app.py
```
