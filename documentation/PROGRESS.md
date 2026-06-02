# PROGRESS — Tracking Pekerjaan

> Update: 2026-06-03. Checklist seluruh tugas. `- [x]` selesai, `- [ ]` belum,
> `- [~]` sebagian/ditunda dengan catatan.

## T1 — Terapkan rekomendasi dokumen
- [x] Naikkan default `epochs` (8 → 25) di `configs/default_training.yaml` (REC-01)
- [x] Naikkan `early_stopping_patience` (4 → 8) (REC-02)
- [x] Naikkan `fine_tune_learning_rate` (1e-5 → 5e-5) (REC-05)
- [x] Tambah override epoch/patience per method di `configs/training_methods.yaml`
- [x] Selaraskan monitoring callback ke `val_accuracy` (REC-03)
- [x] Buat callback baru per stage (state tidak terbawa) (REC-04)
- [x] Terapkan `class_weight` 'balanced' pada `model.fit` (REC-10)
- [x] Output layer `dtype="float32"` untuk mixed precision (REC-11)
- [x] YOLO: cosine LR + warmup + label smoothing (REC-08)
- [~] REC-06 (cosine scheduler TF) — ditunda (low impact, butuh refactor)
- [~] REC-09 (Transformer pretrained/resep khusus) — DITUNDA sesuai batasan (TODO terdokumentasi)

## T2 — Folder optimizer (`src/optimizer/`)
- [x] `src/optimizer/adam.py` (wrapper Adam)
- [x] `src/optimizer/no_optimize.py` (SGD plain baseline)
- [x] `src/optimizer/__init__.py` — `get_optimizer(name, learning_rate, **cfg)`, `list_optimizers()`
- [x] Wiring ke `build_model` + recompile stage 2 (`training_common.py`)
- [x] Argumen `--optimizer` di worker TF (`build_common_arg_parser`)
- [x] Argumen `--optimizer` di YOLO + pemetaan Adam/SGD
- [x] `optimizer: adam` di `configs/default_training.yaml`
- [x] Argumen `--optimizer` + override di `training/train.py`

## T3 — Laporan Streamlit lengkap
- [x] Tambah metrik ROC-AUC, Precision (macro), Recall (macro) pada detail run
- [x] Kurva training (loss/accuracy) — sudah ada, dipertahankan
- [x] Confusion matrix + classification report (CSV) — sudah ada, dipertahankan
- [x] Ringkasan konfigurasi run (epoch/batch/fine-tune) — sudah ada

## T4 — Perbandingan antar-model (overlay)
- [x] Sub-tab "Overlay Model" (pilih ≥2 run)
- [x] Tabel metrik berdampingan
- [x] Grafik overlay kurva training (pilih metrik)

## T5 — Download report berfilter
- [x] Filter dataset/method/augmentasi/**model spesifik** + rentang tanggal
- [x] Download CSV
- [x] Download HTML printable (A4 → Save as PDF) tanpa dependency baru

## T6 — Update dokumentasi
- [x] `architecture.md` (folder `src/optimizer`, optimizer, split dinamis, penyempurnaan)
- [x] `pipeline.md` (tab dashboard baru)
- [x] `documentation/fitur_baru.md` (dokumen komprehensif fitur baru)
- [x] `documentation/analisis_training_rekomendasi.md` (sumber rekomendasi, sudah ada)

## T7 — Training via web (Streamlit)
- [x] Tab "Training" baru
- [x] Pilihan model, optimizer, method, augmentasi, dataset
- [x] Hyperparameter: epoch, batch size, learning rate, fine-tune epoch, seed
- [x] Pilihan split preset + `--split-first`
- [x] Eksekusi `training/train.py` via subprocess + log real-time
- [x] Tampilkan ringkasan hasil setelah selesai

## T8 — Dynamic data split
- [x] Preset `80-10-10` dan `70-15-15` (`SPLIT_PRESETS`)
- [x] `validate_split_ratios` (total=100%, testing==validation)
- [x] Override rasio manual (`--train-ratio/--test-ratio/--val-ratio`)
- [x] Seed dikunci (default 42)
- [x] Terintegrasi di `2.split_data_testing.py`, `train.py`, dan tab Training web

## T9 — Rapikan struktur file & folder
- [~] Pendekatan **aditif & aman** (keputusan user): nama folder lama dipertahankan
      (`dataset/`, `web/`, `report/`, `documentation/`) agar tidak ada path yang patah.
      Hanya menambah `src/optimizer/`. Reorg rename penuh tidak dilakukan.

## T10 — File tracking
- [x] `documentation/PROGRESS.md` (file ini)

## Verifikasi
- [x] `py_compile` semua file yang diubah → OK
- [x] Uji fungsi optimizer registry & resolusi/validasi split → OK
- [ ] Uji training end-to-end nyata (butuh dataset + GPU/CPU; dijalankan user)
