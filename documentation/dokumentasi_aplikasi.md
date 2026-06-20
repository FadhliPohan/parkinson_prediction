# Dokumentasi Aplikasi Prediksi Parkinson

## 1. Ringkasan
Aplikasi ini adalah pipeline machine learning klasifikasi gambar berbasis terminal + dashboard Streamlit.

Fitur inti saat ini:
1. Pemilihan dataset dinamis lewat folder `dataset/original` + override opsional lewat config.
2. Pemilihan augmentasi dinamis lewat registry.
3. Pemilihan model dinamis lewat registry.
4. Pemilihan method training dinamis.
5. Split dataset otomatis dengan balancing kelas sebelum split.
6. Eksekusi training per kombinasi eksperimen secara independen.
7. Report training terstandar untuk dibaca dashboard.

## 2. Dataset yang Tersedia
Berdasarkan registry dinamis (`src/datasets/registry.py`):
1. Dataset yang didefinisikan di `configs/datasets.yaml` (jika ada).
2. Dataset hasil auto-discovery dari setiap subfolder pada `dataset/original`.
3. `all` (opsi CLI untuk menjalankan semua dataset berurutan).

Catatan:
- Untuk cek daftar ID dataset aktif saat runtime:
  ```bash
  python3 -c "from src.datasets.registry import DatasetRegistry as R; r=R(); print('Default:', r.default_dataset); print('Datasets:', ', '.join(r.list_dataset_ids()))"
  ```

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
1. Family `cnn`:
   - `mobilenetv2`
   - `resnet50`
   - `vgg19`
   - `resnet152`
   - `inception_googlenet`
   - `efficientnet`
   - `densenet121`
2. Family `transformer`:
   - `vit`
   - `swintransformer`
   - `deit`
3. Family `yolo`:
   - `yolov8`

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
- Kebijakan split saat ini:
  - rasio default `80:10:10` (train:testing:validation),
  - `testing` dan `validation` harus sama per kelas,
  - sisa pembulatan otomatis dimasukkan ke `train` (contoh 101 data -> 81/10/10).

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

Contoh pipeline khusus CNN:
```bash
python3 training/train.py \
  --dataset parkinson_merder \
  --models all \
  --model-families cnn \
  --all-methods \
  --split-first \
  --on-existing ask
```

Contoh pipeline khusus Transformer:
```bash
python3 training/train.py \
  --dataset parkinson_merder \
  --models all \
  --model-families transformer \
  --all-methods \
  --split-first \
  --on-existing ask
```

Catatan kompatibilitas:
- Opsi lama `--preprocessing-mode augment|no_augment|both` masih didukung.
- Opsi `--model-families`:
  - memfilter model berdasarkan family (`cnn`, `transformer`, `yolo`, atau `all`).
- Opsi `--on-existing`:
  - `ask`: jika ada kombinasi existing, user ditanya sekali di awal apakah semua kombinasi existing perlu diretrain.
  - `retrain`: langsung training ulang dan membuat run/model baru dengan timestamp training baru.
  - `skip`: kombinasi lama dilewati.
- Opsi `--on-existing-split` (saat `--split-first` aktif):
  - `ask`: jika folder split sudah ada, user ditanya perlu split ulang atau tidak.
  - `resplit`: langsung split ulang.
  - `skip`: pakai split lama.
- Opsi `--method-overrides-json`:
  - mengatur runtime config per method (`epochs`, `batch_size`, `fine_tune_epochs`) dalam satu command.
  - contoh JSON: `{"baseline":{"epochs":10,"batch_size":16,"fine_tune_epochs":0}}`.
- Opsi `--shutdown-on-finish`:
  - setelah workflow training selesai, perangkat otomatis menjalankan perintah shutdown OS.
- Jika menjalankan dari menu `main.py`, user akan ditanya di awal untuk setiap method:
  - `Epoch stage-1`,
  - `Batch size`,
  - `Fine-tune epochs`.
  - opsi auto-shutdown setelah training selesai.

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
   - `Ringkasan Run Terbaru per Kombinasi`: tabel run terbaru per kombinasi yang
     dilengkapi kolom **Hapus** (checkbox). Centang run yang ingin dibuang lalu
     konfirmasi untuk menghapus permanen folder report (`report/...`) sekaligus
     folder model (`trained_models/...`) milik run tersebut. Marker `latest_run.txt`
     otomatis diperbarui dan folder kombinasi yang menjadi kosong ikut dibersihkan.
     Tujuannya: menyisakan hanya model yang benar-benar baik dan menghemat storage.
   - `Explorer`: telusur bertingkat dataset -> augmentasi -> method -> model -> run,
   - `Perbandingan`: antar method, antar model, antar augmentasi, dan ranking val accuracy.
   - detail run menampilkan runtime config yang dipakai (`epoch`, `batch`, `fine-tune`).
3. Tab `Prediksi`:
   - pilih kombinasi dataset/augmentasi/method,
   - bandingkan prediksi beberapa model lintas run.
4. Tab `Dokumentasi`:
   - penjelasan akademis & teknis aplikasi untuk pengguna: cara menjalankan, alur sistem
     end-to-end, proses split data (anti-leakage), preprocessing, augmentasi, arsitektur model,
     proses training dua tahap, serta evaluasi & metrik.
   - daftar model/method/optimizer/preset split diambil dinamis dari konfigurasi aktif
     (selalu sinkron dengan sistem).

## 10. Menambah Komponen Baru

### A. Menambah Dataset
1. Tambahkan folder dataset baru di `dataset/original/<nama_folder_dataset>`.
2. Jalankan check/split; dataset baru akan otomatis muncul di registry.
3. (Opsional) Tambahkan entry di `configs/datasets.yaml` jika butuh ID khusus, split dir khusus, atau override parameter dataset.

### B. Menambah Model
1. Buat script model worker.
2. Tambahkan ke `configs/models.yaml`.
3. Isi `family` agar bisa dipakai pipeline khusus family model (`cnn`, `transformer`, `yolo`, atau custom).
4. Pastikan `enabled: true` jika ingin muncul di menu/CLI.

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
6. Untuk job panjang semalaman, gunakan `--shutdown-on-finish` jika ingin perangkat mati otomatis saat workflow selesai.
