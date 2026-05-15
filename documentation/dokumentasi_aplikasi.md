# Dokumentasi Aplikasi Prediksi Parkinson (Versi Dinamis)

## 1. Ringkasan
Aplikasi ini adalah pipeline machine learning klasifikasi gambar (multi-class) berbasis terminal dan Streamlit.

Fitur utama saat ini:
1. Dataset dinamis via registry/config.
2. Model dinamis via registry/config.
3. Method training dinamis via registry/config.
4. Orkestrasi training via CLI (`train.py`) dan menu terminal (`main.py`).
5. Report terstandar untuk dashboard Streamlit.

## 2. Komponen Utama
1. `configs/`
   - `datasets.yaml`
   - `models.yaml`
   - `training_methods.yaml`
   - `default_training.yaml`
2. `src/`
   - dataset management (`src/datasets/*`)
   - model registry (`src/models/registry.py`)
   - training orchestration (`src/training/*`)
   - reporting schema/reader (`src/reporting/*`)
   - inference loader/predictor (`src/inference/*`)
3. Entry points:
   - `train.py` (CLI)
   - `main.py` (menu terminal)
   - `web/app.py` (dashboard)

## 3. Menjalankan Aplikasi

### A. Menu Terminal
```bash
python3 main.py
```

### B. CLI Training Langsung
Contoh satu model:
```bash
python3 train.py --dataset parkinson_multiclass --models mobilenetv2 --method baseline
```

Contoh semua model + satu method:
```bash
python3 train.py --dataset parkinson_multiclass --models all --method transfer_learning
```

Contoh semua model + semua method:
```bash
python3 train.py --dataset parkinson_multiclass --models all --all-methods
```

Contoh pipeline penuh:
```bash
python3 train.py --dataset parkinson_multiclass --models all --all-methods --check-first --split-first --augment-info
```

### C. Dashboard Streamlit
```bash
streamlit run web/app.py
```

## 4. Struktur Artifact
### Report
```
report/<dataset_name>/<model_name>/<run_id>/
```
Isi utama:
- `run_manifest.json`
- `evaluation_metrics.json/csv`
- `training_history.csv`
- `classification_report.csv`
- `confusion_matrix.csv`
- `split_distribution.csv`
- `summary.txt`
- visual `.png`

### Trained Model
```
trained_models/<dataset_name>/<model_name>/<run_id>/
```
Isi utama:
- `best_model.keras` / `best_model.pt`
- `final_model.keras` / `final_model.pt`
- `class_names.json`

## 5. Menambah Dataset
1. Tambahkan dataset baru ke `configs/datasets.yaml`.
2. Jalankan validasi dan split:
```bash
python3 1.check_dataset.py --dataset <dataset_id>
python3 2.split_data_testing.py --dataset <dataset_id>
```

## 6. Menambah Model
1. Buat script model baru di `model/`.
2. Tambahkan entry di `configs/models.yaml`.
3. Model otomatis tampil di menu/CLI/dashboard.

## 7. Menambah Method Training
1. Tambahkan entry method di `configs/training_methods.yaml`.
2. Isi `arg_overrides` sesuai strategi.
3. Method otomatis tersedia di `main.py` dan `train.py`.

## 8. Catatan
1. Prioritas utama training saat ini tetap dari terminal/CLI.
2. Trigger training dari Streamlit masih ditunda (pending) sesuai keputusan proyek.
3. Untuk multi-class, pipeline sudah disesuaikan agar stabil (metric training tidak lagi memaksa AUC multi-label pada sparse label).
