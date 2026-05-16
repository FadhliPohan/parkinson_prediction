# Dokumentasi Aplikasi Prediksi Parkinson

## 1. Ringkasan
Aplikasi ini adalah pipeline machine learning klasifikasi gambar berbasis terminal + dashboard Streamlit.

Fitur inti saat ini:
1. Pemilihan dataset dinamis lewat config.
2. Pemilihan augmentasi dinamis lewat registry.
3. Pemilihan model dinamis lewat registry.
4. Pemilihan method training dinamis.
5. Split dataset otomatis dengan balancing kelas sebelum split.
6. Eksekusi training per kombinasi eksperimen secara independen.
7. Report training terstandar untuk dibaca dashboard.

## 2. Dataset yang Tersedia
Berdasarkan `configs/datasets.yaml`:
1. `parkinson_merder`
2. `parkinson_mixing`
3. `all` (opsi CLI untuk menjalankan semua dataset berurutan)

## 3. Augmentasi Training yang Tersedia
Berdasarkan `configs/augmentations.yaml`:
1. `no_augment`
2. `augment_on_the_fly`

Catatan:
- Setiap dataset bisa membatasi augmentasi yang boleh dipakai melalui `augmentation_options` di `configs/datasets.yaml`.

## 4. Method Training yang Tersedia
Berdasarkan `configs/training_methods.yaml`:
1. `baseline`
2. `transfer_learning` (default)
3. `transfer_learning_mixed_precision`
4. `full_fine_tuning`

## 5. Model yang Tersedia
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

## 6. Mekanisme Balancing Dataset
Balancing dilakukan di tahap split (`src/datasets/splitter.py`):
1. Sistem cek jumlah gambar per kelas pada dataset original.
2. Jika kelas tidak seimbang, hanya kelas minoritas yang ditambah.
3. Penambahan data dilakukan dengan augmentasi rotasi acak kecil (`-20` sampai `+20` derajat).
4. Target jumlah tiap kelas disamakan ke jumlah kelas mayoritas.
5. Setelah itu baru dilakukan split train/testing/validation.

Catatan:
- Balancing membuat file hasil augmentasi di area split (bukan mengubah dataset original).
- Detail balancing tercatat di `_metadata/split_manifest.json`.

## 7. Menjalankan Aplikasi

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
Contoh satu kombinasi:
```bash
python3 training/train.py \
  --dataset parkinson_merder \
  --augmentations augment_on_the_fly \
  --models mobilenetv2 \
  --method transfer_learning \
  --split-first \
  --on-existing-split ask \
  --on-existing ask
```

Contoh banyak kombinasi:
```bash
python3 training/train.py \
  --dataset all \
  --augmentations all \
  --models all \
  --method all \
  --method-overrides-json '{"baseline":{"epochs":10,"batch_size":16,"fine_tune_epochs":0},"transfer_learning":{"epochs":8,"batch_size":12,"fine_tune_epochs":2}}' \
  --split-first \
  --on-existing-split ask \
  --on-existing ask
```

Catatan kompatibilitas:
- Opsi lama `--preprocessing-mode augment|no_augment|both` masih didukung.
- Opsi `--on-existing`:
  - `ask`: jika kombinasi sudah pernah training, user ditanya perlu retrain atau tidak.
  - `retrain`: langsung training ulang dan membuat run/model baru dengan timestamp training baru.
  - `skip`: kombinasi lama dilewati.
- Opsi `--on-existing-split` (saat `--split-first` aktif):
  - `ask`: jika folder split sudah ada, user ditanya perlu split ulang atau tidak.
  - `resplit`: langsung split ulang.
  - `skip`: pakai split lama.
- Opsi `--method-overrides-json`:
  - mengatur runtime config per method (`epochs`, `batch_size`, `fine_tune_epochs`) dalam satu command.
  - contoh JSON: `{"baseline":{"epochs":10,"batch_size":16,"fine_tune_epochs":0}}`.
- Jika menjalankan dari menu `main.py`, user akan ditanya di awal untuk setiap method:
  - `Epoch stage-1`,
  - `Batch size`,
  - `Fine-tune epochs`.

### E. Dashboard Streamlit
```bash
streamlit run web/app.py
```

## 8. Struktur Output

### Report
```text
report/<dataset>/<augmentasi>/<method>/<model>/<run_id>/
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

Ringkasan eksekusi workflow:
```text
report/_workflow_runs/workflow_summary_<timestamp>.json
report/_workflow_runs/workflow_summary_<timestamp>.csv
```
Ringkasan ini juga memuat konfigurasi runtime utama (`epochs`, `batch_size`, `fine_tune_epochs`) per kombinasi.

### Model
```text
trained_models/<dataset>/<augmentasi>/<method>/<model>/<run_id>/
```
File utama:
- `best_model.keras` / `best_model.pt`
- `final_model.keras` / `final_model.pt`
- `class_names.json`

## 9. Tampilan Dashboard
1. Tab `Dataset`:
   - ringkasan dataset original,
   - ringkasan split train/testing/validation.
2. Tab `Training Report`:
   - `Explorer`: telusur bertingkat dataset -> augmentasi -> method -> model -> run,
   - `Perbandingan`: antar method, antar model, antar augmentasi, dan ranking val accuracy.
   - detail run menampilkan runtime config yang dipakai (`epoch`, `batch`, `fine-tune`).
3. Tab `Prediksi`:
   - pilih kombinasi dataset/augmentasi/method,
   - bandingkan prediksi beberapa model lintas run.

## 10. Menambah Komponen Baru

### A. Menambah Dataset
1. Tambah entry di `configs/datasets.yaml`.
2. Tambahkan `augmentation_options` bila perlu.
3. Jalankan check dan split.

### B. Menambah Model
1. Buat script model worker.
2. Tambahkan ke `configs/models.yaml`.
3. Pastikan `enabled: true` jika ingin muncul di menu/CLI.

### C. Menambah Method Training
1. Tambah method di `configs/training_methods.yaml`.
2. Isi `description` dan `arg_overrides`.

### D. Menambah Opsi Augmentasi
1. Tambah entry di `configs/augmentations.yaml`.
2. Aktifkan di `augmentation_options` dataset terkait.

## 11. Catatan Operasional
1. Orkestrasi training utama ada di `training/train.py`.
2. Dashboard Streamlit difokuskan sebagai pembaca report/model, bukan pusat logic training.
3. Jika split dijalankan ulang, isi folder split lama akan ditimpa hasil split terbaru.
4. Jika kombinasi eksperimen sangat banyak, jalankan batch bertahap agar mudah dianalisis.
5. Sebelum training berjalan, pipeline memvalidasi split balance (after_counts + train/testing/validation) agar kelas tetap seimbang.
