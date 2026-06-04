# Dokumentasi Lengkap Pipeline Eksekusi

## 1. Tujuan Dokumen
Dokumen ini menjelaskan pipeline menu terminal `main.py`, alur eksekusi backend, dan output yang dihasilkan.

Fokus dokumen:
1. Menjelaskan pipeline `1` sampai `12`.
2. Menjelaskan hubungan menu terminal dengan `training/train.py`.
3. Menjelaskan eksekusi kombinasi eksperimen `dataset -> augmentasi -> method -> model`.

## 2. Konteks Sistem Saat Ini
1. Dataset aktif:
   - Berasal dari registry dinamis:
     - hasil auto-discovery semua subfolder `dataset/original/*`,
     - ditambah/di-override entry opsional dari `configs/datasets.yaml`.
2. Augmentasi aktif:
   - `no_augment`
   - `augment_on_the_fly`
3. Method training aktif:
   - `baseline`
   - `transfer_learning`
   - `transfer_learning_mixed_precision`
   - `full_fine_tuning`
4. Model aktif:
   - Family `cnn`:
     - `mobilenetv2`
     - `resnet50`
     - `vgg16` *(baru)*
     - `vgg19`
     - `resnext50` *(baru — ResNeXt-50-32x4d, grouped convolutions)*
     - `resnet152`
     - `inception_googlenet`
     - `efficientnet`
     - `densenet121`
   - Family `transformer`:
     - `vit`
     - `swintransformer`
     - `deit`
   - Family `yolo`:
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
11. Pipeline penuh model CNN.
12. Pipeline penuh model Transformer.

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
5. Isi konfigurasi runtime method di awal:
   - `epoch stage-1`,
   - `batch size`,
   - `fine-tune epochs`.
6. Konfigurasi runtime method direkam ke report run (`run_manifest`) dan ringkasan workflow.
7. `main.py` membangun command ke `training/train.py`.

Output:
1. Report per kombinasi di `report/<dataset>/<augmentasi>/<method>/<model>/<run_id>/`.
2. Model per kombinasi di `trained_models/<dataset>/<augmentasi>/<method>/<model>/<run_id>/`.

### Pipeline 6 - Training semua model (satu method)
Aktivitas:
1. Pilih dataset.
2. Pilih satu method.
3. Set konfigurasi runtime method di awal (`epoch`, `batch`, `fine-tune`).
4. Jalankan semua model aktif.
5. Preprocessing mode sesuai pilihan user.

### Pipeline 7 - Training semua model + semua method
Aktivitas:
1. Pilih dataset.
2. Jalankan semua model.
3. Jalankan semua method.
4. Untuk tiap method, set konfigurasi runtime di awal (`epoch`, `batch`, `fine-tune`).
5. Preprocessing mode sesuai pilihan user.

### Pipeline 8 - Pipeline penuh
Aktivitas:
1. Check dataset.
2. Split dataset.
3. Tampilkan info augmentasi.
4. Set konfigurasi runtime per method di awal (`epoch`, `batch`, `fine-tune`).
5. Training semua model dengan method sesuai pilihan.

### Pipeline 9 - Jalankan dashboard Streamlit
Aktivitas:
1. Menjalankan `streamlit run web/app.py`.
2. Dashboard membaca report dan model artifact.
3. Dashboard kini memiliki 4 tab: **Dataset**, **Training** (training via web),
   **Training Report** (metrik lengkap + ROC/AUC + download berfilter + overlay
   perbandingan), dan **Prediksi**. Detail di `documentation/fitur_baru.md`.

### Pipeline 10 - Workflow training fleksibel (multi kombinasi)
Aktivitas:
1. User memilih dataset (single/multi/all).
2. User memilih augmentasi (single/multi/all).
3. User memilih method (single/multi/all).
4. User memilih model (single/multi/all).
5. User mengisi konfigurasi runtime untuk tiap method terpilih:
   - `epoch stage-1`,
   - `batch size`,
   - `fine-tune epochs`.
6. Orchestrator mengeksekusi kombinasi secara independen.
7. Setiap kombinasi punya `experiment_id` unik.
8. Jika kombinasi sudah pernah ditraining, sistem akan menangani sesuai mode `on-existing`:
   - `ask`: tanya user sekali di awal apakah semua kombinasi existing perlu training ulang,
   - `retrain`: langsung training ulang,
   - `skip`: lewati kombinasi lama.
9. User bisa memfilter family model lewat opsi `model-families` (contoh: `cnn`, `transformer`).
10. User bisa mengaktifkan auto-shutdown setelah training selesai.

Output tambahan:
1. Ringkasan batch workflow di:
   - `report/_workflow_runs/workflow_summary_<timestamp>.json`
   - `report/_workflow_runs/workflow_summary_<timestamp>.csv`

### Pipeline 11 - Pipeline penuh model CNN
Aktivitas:
1. Pilih dataset.
2. Pilih method (single atau all).
3. Jalankan check + split + info augmentasi.
4. Training seluruh model pada family `cnn`.
5. Opsi auto-shutdown tersedia setelah workflow selesai.

### Pipeline 12 - Pipeline penuh model Transformer
Aktivitas:
1. Pilih dataset.
2. Pilih method (single atau all).
3. Jalankan check + split + info augmentasi.
4. Training seluruh model pada family `transformer`.
5. Opsi auto-shutdown tersedia setelah workflow selesai.

## 6. Flow Runtime `training/train.py`
Setelah command dieksekusi:
1. Resolve dataset target (`single`, `multi`, atau `all`).
2. Resolve augmentasi target:
   - prioritas `--augmentations` jika diberikan,
   - fallback `--preprocessing-mode` untuk kompatibilitas lama.
3. Resolve method target (`--method` multi/all atau `--all-methods`).
4. Resolve model target (`--models` single/multi/all).
5. (Opsional) filter model berdasarkan family (`--model-families`).
6. Merge parameter training:
   - default training,
   - override method,
   - override user global,
   - override runtime per method (jika diisi).
7. Jika `--split-first` aktif dan folder split sudah ada:
   - `--on-existing-split ask`: tanya user split ulang atau pakai split lama,
   - `--on-existing-split resplit`: langsung split ulang,
   - `--on-existing-split skip`: pakai split lama.
8. Validasi split balance dilakukan sebelum training:
   - cek balance total per class sesudah balancing,
   - cek balance per split `train/testing/validation`.
9. Kebijakan pembagian jumlah data split:
   - default rasio `80:10:10`,
   - `testing` dan `validation` harus sama,
   - sisa pembulatan masuk ke `train`.
10. Eksekusi loop kombinasi:
   - dataset loop,
   - augmentasi loop,
   - method loop,
   - model loop.
11. Cek apakah kombinasi sudah punya run sebelumnya.
12. Jika run sudah ada:
   - mode `ask` akan meminta konfirmasi retrain sekali di awal workflow,
   - mode `retrain` akan membuat run/model baru dengan timestamp training baru,
   - mode `skip` akan melewati kombinasi lama.
13. Simpan hasil run per kombinasi.
14. Ringkasan workflow juga menyimpan `epochs`, `batch_size`, dan `fine_tune_epochs` per kombinasi.
15. Jika `--shutdown-on-finish` aktif, sistem menjalankan perintah shutdown OS setelah workflow selesai.

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
  --method-overrides-json '{"baseline":{"epochs":10,"batch_size":16,"fine_tune_epochs":0},"full_fine_tuning":{"epochs":12,"batch_size":8,"fine_tune_epochs":8}}' \
  --split-first \
  --on-existing-split ask \
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

### D. Pipeline khusus family CNN
```bash
python3 training/train.py \
  --dataset parkinson_merder \
  --models all \
  --model-families cnn \
  --all-methods \
  --split-first \
  --on-existing ask
```

### E. Pipeline khusus family Transformer
```bash
python3 training/train.py \
  --dataset parkinson_merder \
  --models all \
  --model-families transformer \
  --all-methods \
  --split-first \
  --on-existing ask
```

### F. Auto-shutdown setelah training
```bash
python3 training/train.py \
  --dataset parkinson_merder \
  --models mobilenetv2 \
  --method transfer_learning \
  --shutdown-on-finish
```

### G. Mode existing run
```bash
# Tanyakan dulu jika kombinasi sudah pernah training
python3 training/train.py ... --on-existing ask

# Langsung retrain semua kombinasi yang sudah ada
python3 training/train.py ... --on-existing retrain

# Lewati kombinasi yang sudah ada
python3 training/train.py ... --on-existing skip
```

### H. Mode existing split
```bash
# Tanyakan dulu jika folder split sudah ada
python3 training/train.py ... --split-first --on-existing-split ask

# Paksa split ulang
python3 training/train.py ... --split-first --on-existing-split resplit

# Pakai split lama tanpa split ulang
python3 training/train.py ... --split-first --on-existing-split skip
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
9. Konfigurasi runtime training (`epochs`, `batch_size`, `fine_tune_epochs`) terekam di `run_manifest.json` dan terbaca di dashboard.

### Model per run
```text
trained_models/<dataset>/<augmentasi>/<method>/<model>/<run_id>/
```

## 9. Mapping Menu ke Backend
1. Pipeline 1 -> install dependency internal.
2. Pipeline 2 -> `training/1.check_dataset.py`.
3. Pipeline 3 -> `training/2.split_data_testing.py`.
4. Pipeline 4 -> `training/3.augmentasi.py`.
5. Pipeline 5/6/7/8/10/11/12 -> `training/train.py`.
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
4. Detail run menampilkan konfigurasi runtime (`epoch`, `batch`, `fine-tune`) agar setiap eksperimen mudah diaudit.

## 11. Troubleshooting Singkat
1. Error dependency:
   - Jalankan Pipeline 1 ulang.
2. Dataset tidak ditemukan:
   - Cek apakah folder dataset ada di `dataset/original`.
   - Jika pakai ID custom, cek mapping di `configs/datasets.yaml`.
3. Augmentasi tidak valid:
   - Cek `configs/augmentations.yaml` dan `augmentation_options` dataset.
4. Dashboard kosong:
   - Pastikan minimal satu run sukses dan artifact report terbentuk.
5. Training terlalu berat:
   - Kurangi kombinasi (dataset/method/model/augmentasi) atau jalankan batch bertahap.
6. Kombinasi lama ikut tertimpa analisis:
   - Gunakan `--on-existing ask` atau `--on-existing skip` agar run lama tidak tercampur tanpa sengaja.
7. Auto-shutdown tidak berjalan:
   - Cek hak akses user OS untuk perintah shutdown.
   - Di Windows, pastikan command prompt dijalankan sebagai user yang punya izin shutdown.

## 12. Catatan Penting
1. Setiap kombinasi eksperimen dijalankan terpisah agar analisis tidak tercampur.
2. Pipeline menu dan CLI saling melengkapi.
3. Untuk automation/batch besar, gunakan `training/train.py` langsung.
4. Untuk eksplorasi cepat, gunakan menu `main.py`.
