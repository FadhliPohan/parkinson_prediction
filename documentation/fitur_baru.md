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

---

## 6. Preset Split Multi (incl. "keduanya"), Optimizer di Terminal, & Streamlit Non-Blocking

> Update: 2026-06-09.

### 6.1 Preset split dinamis dengan folder & report terpisah
- Preset rasio split: `80-10-10`, `70-15-15`, `config` (rasio default config dataset),
  atau `both` (jalankan KEDUA preset).
- Setiap preset eksplisit ditulis ke **folder terpisah** agar tidak saling menimpa:
  - `config`  → `dataset/split/<id>` (folder dasar, kompatibel lama)
  - `80-10-10` → `dataset/split/<id>__80-10-10`
  - `70-15-15` → `dataset/split/<id>__70-15-15`
- `split_manifest.json` kini menyimpan field `split_preset` (mis. `"80-10-10"`).
- Argumen baru:
  - `training/2.split_data_testing.py --split-presets both|config|80-10-10,70-15-15`
  - `training/train.py --split-presets both|config|<preset>[,<preset>]`
  - `--split-preset` (tunggal, lama) tetap ada; bila dipakai eksplisit kini juga
    menulis ke folder bersuffix preset.

```bash
# Split kedua preset sekaligus (dua folder terpisah)
python training/2.split_data_testing.py --dataset parkinson_merder --split-presets both --on-existing-split resplit

# Training pada KEDUA preset (report terpisah, bisa dibandingkan)
python training/train.py --dataset parkinson_merder --models all --method transfer_learning \
  --split-first --split-presets both --on-existing retrain
```

### 6.2 Visibilitas split saat training
- `train.py` mencetak **preset split aktif + folder** sebelum setiap blok kombinasi,
  dan menambahkan token `split=<preset>` pada baris `[RUN] ...`.
- Worker model (`training_common.py`) mencetak `=== Split Dataset Dipakai ===`
  (preset + rasio yang dibaca dari `split_manifest.json`).
- `run_manifest.json` menyimpan `dataset.split_preset` & `dataset.split_ratio`.
- Report/Streamlit menambahkan kolom **`split_preset`** dan **`optimizer`** (ringkasan
  terbaru, ranking, download, overlay, serta detail run di Explorer yang juga menampilkan
  **rasio split**), dan deteksi "kombinasi existing" kini memperhitungkan preset
  sehingga 80-10-10 vs 70-15-15 tidak saling menimpa pointer model.
- `report_reader` men-surface `optimizer` dari `run_manifest.json`
  (`training.parameters.optimizer`).

### 6.3 Pemilihan optimizer di terminal (`main.py`)
- Semua menu training (5/6/7/8/10/11/12) kini menanyakan **optimizer**: `adam` atau
  `no_optimize`. Pilihan diteruskan sebagai `--optimizer` ke `train.py`.
- Semua menu training juga menanyakan **preset split** (termasuk `both`); menu Split
  (no.3) juga menanyakan preset (incl. `both`).

### 6.4 Streamlit tahan-lama (tidak timeout) saat training
- Training di tab **Training** kini dijalankan sebagai **proses terpisah** yang menulis
  ke file log (`report/_web_runs/train_*.log`).
- Dashboard memantau log secara berkala (rerun pendek tiap ~2 detik) sehingga koneksi
  websocket tetap hidup dan **tidak terlihat hang/mati** walau training lama. Tersedia
  tombol **Hentikan Training** dan reset monitor.

### 6.5 Default & clue hyperparameter per model/method (`src/training/recommendations.py`)
- Saat model/method dipilih, hyperparameter (epoch, batch, fine-tune, LR) **otomatis
  terisi** dengan rekomendasi yang sesuai karakter model:
  - CNN pretrained ringan vs berat (batch lebih kecil), ResNeXt50 & Transformer
    (from-scratch → epoch lebih banyak, fine-tune 0), YOLO (fine-tune tidak dipakai).
  - Method `baseline` → from-scratch (fine-tune 0, epoch dinaikkan); `full_fine_tuning`
    → fine-tune lebih panjang; `*_mixed_precision` → boleh batch lebih besar.
- Dashboard menampilkan **clue/peringatan** bila setelan kurang pas (mis. fine-tune
  diaktifkan pada model from-scratch, batch terlalu besar → risiko OOM, LR terlalu besar,
  epoch terlalu kecil).

### 6.6 Tab Training Streamlit: semua pilihan multi-select (jalankan semua kombinasi)
- **Dataset, Model, Method, Optimizer, Augmentasi, dan Preset split** di tab Training kini
  semuanya **multi-select**. Satu kali klik "Mulai Training" menjalankan **semua kombinasi**
  dataset × model × method × optimizer × augmentasi × preset split dalam **satu proses**.
- Dashboard menampilkan estimasi **total kombinasi** sebelum dijalankan.
- Fan-out optimizer baru di `train.py` via argumen `--optimizers` (mis. `all` atau
  `adam,no_optimize`); optimizer kini menjadi dimensi kombinasi (masuk `experiment_id`,
  deteksi run existing, dan summary). `--optimizer` (tunggal) tetap didukung.
- Opsi augmentasi adalah **gabungan** dari semua dataset terpilih.
- Rekomendasi/clue hyperparameter mengikuti **model & method pertama** yang dipilih
  (karena epoch/batch/LR berlaku global untuk seluruh kombinasi pada satu run web).

### 6.7 Konfirmasi retrain via UI + fix training web menggantung (2026-06-20)
- **Masalah lama**: saat menjalankan training dari web, train.py menampilkan prompt
  terminal `Lanjutkan training ulang? [y/N]` untuk kombinasi yang sudah pernah dilatih.
  Karena Streamlit tidak punya stdin yang bisa diketik, training **menggantung total**.
- **Akar masalah**: subprocess mewarisi tty milik proses `streamlit run`, sehingga
  `sys.stdin.isatty()` bernilai True dan train.py mengira sesi interaktif.
- **Perbaikan**:
  - `web/app.py` menjalankan subprocess dengan `stdin=DEVNULL` (tidak mewarisi tty).
  - `train.py` punya flag `--non-interactive` (dipakai web) + helper `_session_is_interactive()`;
    semua `input()` dibungkus `try/except EOFError` → tidak pernah menggantung.
  - **Konfirmasi dipindah ke UI**: sebelum "Mulai Training", dashboard mendeteksi berapa
    kombinasi terpilih yang sudah pernah ditraining lalu menampilkan pilihan
    **Latih ulang (retrain)** vs **Lewati yang sudah ada (skip)**. Pilihan dikirim ke
    train.py via `--on-existing`. Tidak perlu lagi mengetik y/N di terminal.
