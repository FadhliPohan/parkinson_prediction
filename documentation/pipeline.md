# Dokumentasi Lengkap Pipeline Eksekusi

## 1. Tujuan Dokumen
Dokumen ini menjelaskan pipeline menu terminal `main.py`, alur eksekusi backend, dan output yang dihasilkan.

Fokus dokumen:
1. Menjelaskan pipeline `1` sampai `10`.
2. Menjelaskan hubungan menu terminal dengan `training/train.py`.
3. Menjelaskan eksekusi kombinasi eksperimen `dataset -> augmentasi -> method -> model`.

## 2. Konteks Sistem Saat Ini
1. Dataset aktif:
   - `parkinson_merder`
   - `parkinson_mixing`
2. Augmentasi aktif:
   - `no_augment`
   - `augment_on_the_fly`
3. Method training aktif:
   - `baseline`
   - `transfer_learning`
   - `transfer_learning_mixed_precision`
   - `full_fine_tuning`
4. Model aktif:
   - `mobilenetv2`
   - `resnet50`
   - `vgg19`
   - `resnet152`
   - `inception_googlenet`
   - `efficientnet`
   - `densenet121`
   - `vit`
   - `swintransformer`
   - `deit`
   - `yolov8`

## 3. Entry Point Eksekusi
1. Menu interaktif:
```bash
python3 main.py
```
2. CLI orchestrator langsung:
```bash
python3 training/train.py ...
```

## 4. Peta Pipeline Menu
1. Instalasi dependency + virtual environment.
2. Check distribusi dataset.
3. Split dataset.
4. Info kebijakan augmentasi train.
5. Training satu model.
6. Training semua model (satu method).
7. Training semua model + semua method.
8. Pipeline penuh (check -> split -> augment -> train all models).
9. Jalankan dashboard Streamlit.
10. Workflow training fleksibel (multi dataset/augmentasi/method/model).

## 5. Detail Tiap Pipeline

### Pipeline 1 - Instalasi dependency + virtual environment
Aktivitas:
1. Cek `.venv`.
2. Buat venv jika belum ada.
3. Upgrade `pip`.
4. Install dependency dari `requirements.txt`.

### Pipeline 2 - Check distribusi dataset
Aktivitas:
1. Memanggil `training/1.check_dataset.py`.
2. Menampilkan distribusi kelas dataset original.
3. Menyimpan grafik distribusi.

### Pipeline 3 - Split dataset
Aktivitas:
1. Memanggil `training/2.split_data_testing.py`.
2. Menjalankan balancing kelas minoritas bila perlu.
3. Split data ke train/testing/validation.
4. Menulis manifest split.

### Pipeline 4 - Info kebijakan augmentasi train
Aktivitas:
1. Memanggil `training/3.augmentasi.py`.
2. Menampilkan distribusi split train.
3. Menampilkan kebijakan augmentasi on-the-fly.

### Pipeline 5 - Training satu model
Aktivitas:
1. Pilih dataset.
2. Pilih model.
3. Pilih method.
4. Pilih preprocessing mode (`augment`, `no_augment`, `both`).
5. Isi override opsional.
6. `main.py` membangun command ke `training/train.py`.

Output:
1. Report per kombinasi di `report/<dataset>/<augmentasi>/<method>/<model>/<run_id>/`.
2. Model per kombinasi di `trained_models/<dataset>/<augmentasi>/<method>/<model>/<run_id>/`.

### Pipeline 6 - Training semua model (satu method)
Aktivitas:
1. Pilih dataset.
2. Pilih satu method.
3. Jalankan semua model aktif.
4. Preprocessing mode sesuai pilihan user.

### Pipeline 7 - Training semua model + semua method
Aktivitas:
1. Pilih dataset.
2. Jalankan semua model.
3. Jalankan semua method.
4. Preprocessing mode sesuai pilihan user.

### Pipeline 8 - Pipeline penuh
Aktivitas:
1. Check dataset.
2. Split dataset.
3. Tampilkan info augmentasi.
4. Training semua model dengan method sesuai pilihan.

### Pipeline 9 - Jalankan dashboard Streamlit
Aktivitas:
1. Menjalankan `streamlit run web/app.py`.
2. Dashboard membaca report dan model artifact.

### Pipeline 10 - Workflow training fleksibel (multi kombinasi)
Aktivitas:
1. User memilih dataset (single/multi/all).
2. User memilih augmentasi (single/multi/all).
3. User memilih method (single/multi/all).
4. User memilih model (single/multi/all).
5. Orchestrator mengeksekusi kombinasi secara independen.
6. Setiap kombinasi punya `experiment_id` unik.
7. Jika kombinasi sudah pernah ditraining, sistem akan menangani sesuai mode `on-existing`:
   - `ask`: tanya user apakah perlu training ulang,
   - `retrain`: langsung training ulang,
   - `skip`: lewati kombinasi lama.

Output tambahan:
1. Ringkasan batch workflow di:
   - `report/_workflow_runs/workflow_summary_<timestamp>.json`
   - `report/_workflow_runs/workflow_summary_<timestamp>.csv`

## 6. Flow Runtime `training/train.py`
Setelah command dieksekusi:
1. Resolve dataset target (`single`, `multi`, atau `all`).
2. Resolve augmentasi target:
   - prioritas `--augmentations` jika diberikan,
   - fallback `--preprocessing-mode` untuk kompatibilitas lama.
3. Resolve method target (`--method` multi/all atau `--all-methods`).
4. Resolve model target (`--models` single/multi/all).
5. Merge parameter training:
   - default training,
   - override method,
   - override user.
6. Eksekusi loop kombinasi:
   - dataset loop,
   - augmentasi loop,
   - method loop,
   - model loop.
7. Cek apakah kombinasi sudah punya run sebelumnya.
8. Jika run sudah ada:
   - mode `ask` akan meminta konfirmasi retrain,
   - mode `retrain` akan membuat run/model baru dengan timestamp training baru,
   - mode `skip` akan melewati kombinasi lama.
9. Simpan hasil run per kombinasi.

## 7. Contoh Command

### A. Satu kombinasi
```bash
python3 training/train.py \
  --dataset parkinson_merder \
  --augmentations augment_on_the_fly \
  --method transfer_learning \
  --models mobilenetv2
```

### B. Banyak kombinasi
```bash
python3 training/train.py \
  --dataset all \
  --augmentations all \
  --method all \
  --models resnet50,mobilenetv2,yolov8 \
  --split-first \
  --on-existing ask
```

### C. Kompatibilitas mode lama
```bash
python3 training/train.py \
  --dataset parkinson_merder \
  --models all \
  --all-methods \
  --preprocessing-mode both
```

### D. Mode existing run
```bash
# Tanyakan dulu jika kombinasi sudah pernah training
python3 training/train.py ... --on-existing ask

# Langsung retrain semua kombinasi yang sudah ada
python3 training/train.py ... --on-existing retrain

# Lewati kombinasi yang sudah ada
python3 training/train.py ... --on-existing skip
```

## 8. Struktur Output

### Report per run
```text
report/<dataset>/<augmentasi>/<method>/<model>/<run_id>/
```

File penting:
1. `run_manifest.json`
2. `evaluation_metrics.json` dan `evaluation_metrics.csv`
3. `training_history.csv`
4. `classification_report.csv`
5. `confusion_matrix.csv`
6. `split_distribution.csv`
7. visual `.png`
8. `summary.txt`

### Model per run
```text
trained_models/<dataset>/<augmentasi>/<method>/<model>/<run_id>/
```

## 9. Mapping Menu ke Backend
1. Pipeline 1 -> install dependency internal.
2. Pipeline 2 -> `training/1.check_dataset.py`.
3. Pipeline 3 -> `training/2.split_data_testing.py`.
4. Pipeline 4 -> `training/3.augmentasi.py`.
5. Pipeline 5/6/7/8/10 -> `training/train.py`.
6. Pipeline 9 -> `streamlit run web/app.py`.

## 10. Alur Dashboard Report
1. Dashboard membaca indeks eksperimen dari seluruh `run_manifest.json`.
2. Report ditampilkan bertingkat pada tab `Explorer`:
   - dataset -> augmentasi -> method -> model -> run.
3. Perbandingan disediakan terpisah pada tab `Perbandingan`:
   - antar method,
   - antar model,
   - antar augmentasi,
   - ranking model berdasarkan validation accuracy.

## 11. Troubleshooting Singkat
1. Error dependency:
   - Jalankan Pipeline 1 ulang.
2. Dataset tidak ditemukan:
   - Cek `configs/datasets.yaml`.
3. Augmentasi tidak valid:
   - Cek `configs/augmentations.yaml` dan `augmentation_options` dataset.
4. Dashboard kosong:
   - Pastikan minimal satu run sukses dan artifact report terbentuk.
5. Training terlalu berat:
   - Kurangi kombinasi (dataset/method/model/augmentasi) atau jalankan batch bertahap.
6. Kombinasi lama ikut tertimpa analisis:
   - Gunakan `--on-existing ask` atau `--on-existing skip` agar run lama tidak tercampur tanpa sengaja.

## 12. Catatan Penting
1. Setiap kombinasi eksperimen dijalankan terpisah agar analisis tidak tercampur.
2. Pipeline menu dan CLI saling melengkapi.
3. Untuk automation/batch besar, gunakan `training/train.py` langsung.
4. Untuk eksplorasi cepat, gunakan menu `main.py`.
