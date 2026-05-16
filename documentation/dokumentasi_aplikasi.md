# Dokumentasi Aplikasi Prediksi Parkinson

## 1. Ringkasan
Aplikasi ini adalah pipeline machine learning klasifikasi gambar berbasis terminal + dashboard Streamlit.

Fitur inti saat ini:
1. Pemilihan dataset dinamis lewat config.
2. Pemilihan model dinamis lewat registry.
3. Pemilihan method training dinamis.
4. Split dataset otomatis dengan balancing kelas sebelum split.
5. Report training terstandar untuk dibaca dashboard.

## 2. Dataset yang Tersedia
Berdasarkan `configs/datasets.yaml`:
1. `parkinson_merder`
2. `parkinson_mixing`
3. `all` (opsi CLI untuk menjalankan semua dataset berurutan: Merder lalu Mixing)

## 3. Method Training yang Tersedia
Berdasarkan `configs/training_methods.yaml`:
1. `baseline`
2. `transfer_learning` (default)
3. `transfer_learning_mixed_precision`
4. `full_fine_tuning`

## 4. Model yang Tersedia
Berdasarkan `configs/models.yaml`:
1. `mobilenetv2`
2. `resnet50`
3. `vgg19`
4. `resnet152`
5. `inception_googlenet`
6. `efficientnet`
7. `densenet121`
8. `vit`
9. `swintransformer`
10. `deit`
11. `yolov8`

## 5. Mekanisme Balancing Dataset
Balancing dilakukan di tahap split (`src/datasets/splitter.py`):
1. Sistem cek jumlah gambar per kelas pada dataset original.
2. Jika kelas tidak seimbang, hanya kelas minoritas yang ditambah.
3. Penambahan data dilakukan dengan augmentasi rotasi acak kecil (`-20` sampai `+20` derajat).
4. Target jumlah tiap kelas disamakan ke jumlah kelas mayoritas.
5. Setelah itu baru dilakukan split train/testing/validation.

Catatan:
- Balancing ini membuat file hasil augmentasi di folder split (bukan mengubah dataset original).
- Detail balancing tercatat di `_metadata/split_manifest.json`.

## 6. Menjalankan Aplikasi

### A. Menu Terminal
```bash
python3 main.py
```

### B. Check Dataset
```bash
python3 training/1.check_dataset.py --dataset parkinson_merder
```

### C. Split Dataset
```bash
python3 training/2.split_data_testing.py --dataset parkinson_merder
```

### D. Training dari CLI
Contoh satu model satu method:
```bash
python3 training/train.py --dataset parkinson_merder --models mobilenetv2 --method transfer_learning --preprocessing-mode augment
```

Contoh semua model semua method + split dulu untuk semua dataset:
```bash
python3 training/train.py --dataset all --models all --all-methods --preprocessing-mode both --split-first
```

### E. Dashboard Streamlit
```bash
streamlit run web/app.py
```

## 7. Struktur Output

### Report
```text
report/<dataset_name>/<model_name>/<run_id>/
```
File utama:
- `run_manifest.json`
- `evaluation_metrics.json`
- `evaluation_metrics.csv`
- `training_history.csv`
- `classification_report.csv`
- `confusion_matrix.csv`
- `split_distribution.csv`
- `summary.txt`
- visual `.png`

### Model
```text
trained_models/<dataset_name>/<model_name>/<run_id>/
```
File utama:
- `best_model.keras` / `best_model.pt`
- `final_model.keras` / `final_model.pt`
- `class_names.json`

## 8. Menambah Komponen Baru

### A. Menambah Dataset
1. Tambah entry di `configs/datasets.yaml`.
2. Jalankan check dan split.

### B. Menambah Model
1. Buat script model worker.
2. Tambahkan ke `configs/models.yaml`.
3. Pastikan `enabled: true` jika ingin muncul di menu/CLI.

### C. Menambah Method Training
1. Tambah method di `configs/training_methods.yaml`.
2. Isi `description` dan `arg_overrides`.

## 9. Catatan Operasional
1. Orkestrasi training utama ada di `training/train.py`.
2. Dashboard Streamlit difokuskan sebagai pembaca report/model, bukan pusat logic training.
3. Jika split dijalankan ulang, isi folder split lama akan ditimpa dengan hasil split terbaru.
