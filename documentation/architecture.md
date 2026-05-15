# Arsitektur Baru Aplikasi Parkinson Prediction

## 1. Masalah pada Struktur Lama
1. Pemilihan model di `main.py` masih hard-code per menu dan per script.
2. Dataset belum benar-benar dinamis (path sumber/split statis, asumsi kelas tidak fleksibel untuk struktur folder bertingkat).
3. Strategi training belum diregistrasi; perubahan metode perlu edit kode langsung.
4. Path artifact report/model belum memisahkan dimensi dataset.
5. Dashboard Streamlit membaca report/model dengan asumsi struktur lama (`report/<model>/<run_id>`).
6. Metadata run belum standar dan belum ada schema manifest yang lengkap.
7. Duplikasi logic tinggi pada wrapper model dan beberapa bagian pipeline.

## 2. Tujuan Perbaikan
1. Dataset dinamis melalui registry/config.
2. Model dinamis melalui registry/config.
3. Training method/strategy dinamis melalui registry/config.
4. Orkestrasi training berbasis terminal/CLI sebagai jalur utama.
5. Report menjadi sumber utama dashboard, dengan schema standar.
6. Struktur modular agar mudah dikembangkan dan dipelihara.

## 3. Arsitektur Baru (Ringkas)
Arsitektur sekarang menggunakan kombinasi:
1. `configs/*.yaml` sebagai sumber konfigurasi (dataset, model, method, default training).
2. `src/*` sebagai lapisan backend modular (datasets, models, training, reporting, inference, utils).
3. `training/train.py` sebagai CLI orchestrator dinamis.
4. `main.py` sebagai terminal menu dinamis yang memanggil pipeline registry.
5. `web/app.py` sebagai dashboard visualisasi dan prediksi yang membaca artifact/report.

## 4. Struktur Folder Baru
```text
parkinson_prediction/
├─ configs/
│  ├─ datasets.yaml
│  ├─ models.yaml
│  ├─ training_methods.yaml
│  └─ default_training.yaml
│
├─ src/
│  ├─ datasets/
│  │  ├─ registry.py
│  │  ├─ splitter.py
│  │  ├─ validator.py
│  │  └─ transforms.py
│  ├─ models/
│  │  └─ registry.py
│  ├─ training/
│  │  ├─ strategies.py
│  │  ├─ trainer.py
│  │  ├─ tensorflow_trainer.py
│  │  ├─ pytorch_trainer.py
│  │  └─ yolo_trainer.py
│  ├─ reporting/
│  │  ├─ schemas.py
│  │  ├─ report_writer.py
│  │  └─ report_reader.py
│  ├─ inference/
│  │  ├─ model_loader.py
│  │  └─ predictor.py
│  └─ utils/
│     ├─ config.py
│     ├─ paths.py
│     └─ runtime.py
│
├─ dataset/
│  ├─ original/
│  └─ split/
│
├─ model/                      # script model legacy tetap dipakai sebagai worker
│  └─ legacy_or_wrappers/
├─ report/
│  └─ <dataset_name>/<model_name>/<run_id>/
├─ trained_models/
│  └─ <dataset_name>/<model_name>/<run_id>/
├─ training/
│  ├─ train.py                 # CLI dinamis utama
│  ├─ 1.check_dataset.py
│  ├─ 2.split_data_testing.py
│  └─ 3.augmentasi.py
├─ main.py                     # menu terminal dinamis
├─ web/app.py                  # dashboard
└─ documentation/
   ├─ architecture.md
   ├─ dokumentasi_aplikasi.md
   ├─ history.md
   └─ requirements.md
```

## 5. Alur Dataset
1. Definisi dataset ada di `configs/datasets.yaml`.
2. `src/datasets/registry.py` memuat konfigurasi dataset aktif.
3. `src/datasets/validator.py` mendukung deteksi kelas multi-class dengan mode:
   - `direct`
   - `recursive_leaf` (folder leaf berisi gambar dianggap kelas).
4. `src/datasets/splitter.py` melakukan split dinamis berdasar config:
   - rasio split,
   - seed,
   - resize per split,
   - output manifest split (`_metadata/split_manifest.json`).
5. `src/datasets/transforms.py` menjadi sumber tunggal kebijakan augmentasi on-the-fly.

## 6. Alur Model Registry
1. Daftar model ada di `configs/models.yaml`.
2. `src/models/registry.py` membaca model aktif dan script worker.
3. Menambah model baru cukup dengan:
   - tambah entry config model,
   - sediakan script worker model di `model/`.

## 7. Alur Method / Training Strategy
1. Method terdaftar di `configs/training_methods.yaml`.
2. Default hyperparameter di `configs/default_training.yaml`.
3. `src/training/strategies.py` merge:
   - default training,
   - override method,
   - override dari CLI user.
4. `src/training/trainer.py` melakukan dispatch ke runner framework:
   - `tensorflow_trainer.py`
   - `pytorch_trainer.py`
   - `yolo_trainer.py`
5. Mode preprocessing train tersedia secara dinamis:
   - `augment`
   - `no_augment`
   - `both` (menjalankan dua eksperimen per method/model).

## 8. Alur Training
1. `training/train.py` menerima input dataset, model(s), method(s), mode preprocessing, dan override parameter.
2. Opsional langkah awal:
   - check dataset,
   - split dataset,
   - tampilkan info augmentasi.
3. Untuk setiap model/method, orchestrator memanggil script worker model.
4. Script model menghasilkan report dan model artifact dengan struktur baru berbasis dataset.

## 9. Alur Report
1. Format output utama:
   - `report/<dataset>/<model>/<run_id>/...`
   - `trained_models/<dataset>/<model>/<run_id>/...`
2. File report standar:
   - `evaluation_metrics.json/csv`
   - `training_history.csv`
   - `classification_report.csv`
   - `confusion_matrix.csv`
   - `split_distribution.csv`
   - visual PNG (curves, confusion matrix, roc, split, evaluation table)
   - `summary.txt`
   - `run_manifest.json` (schema metadata run).
3. Marker `latest_run.txt` tetap dipertahankan per model-dataset.

## 10. Alur Dashboard Streamlit
1. Streamlit membaca report dinamis dari `report/<dataset>/<model>/<run_id>`.
2. Tab Dataset:
   - ringkasan kelas dataset original,
   - ringkasan split train/testing/validation.
3. Tab Report:
   - ringkasan latest run lintas dataset/model,
   - detail run (manifest, metrics, csv, visual).
4. Tab Prediksi:
   - baca model dari `trained_models/<dataset>/<model>/<run_id>`,
   - prediksi lintas model,
   - bandingkan confidence/probabilitas.

## 11. Format Artifact Training (Schema v2)
`run_manifest.json` minimal memuat:
1. `schema_version`, `run_id`, `run_started_at`, `run_finished_at`.
2. `dataset`:
   - `dataset_name`, `dataset_dir`, `class_names`, `class_count`, `split_counts`.
3. `model`:
   - `model_name`, `framework`, metadata model tambahan.
4. `training`:
   - `method`, `parameters`, `device_used`.
5. `artifacts`:
   - `report_dir`, `model_dir`, `best_model_path`, `final_model_path`.

## 12. Cara Menambah Dataset Baru
1. Tambah entry di `configs/datasets.yaml`.
2. Isi minimal:
   - `original_dir`
   - `split_dir`
   - `class_mode`
   - `split` ratio
   - `seed`
3. Jalankan:
```bash
python3 training/1.check_dataset.py --dataset <dataset_id>
python3 training/2.split_data_testing.py --dataset <dataset_id>
```

## 13. Cara Menambah Model Baru
1. Tambah script worker model di `model/<nama_model>.py`.
2. Tambah entry model di `configs/models.yaml`:
   - `display_name`
   - `script_path`
   - `framework`
   - `preprocess_key`
   - `enabled`
3. Model otomatis muncul di CLI/menu dan dashboard prediksi.

## 14. Cara Menambah Method Training Baru
1. Tambah entry method di `configs/training_methods.yaml`.
2. Isi:
   - `description`
   - `arg_overrides` (mis. epoch, mixed precision, no_pretrained, dll).
3. Method otomatis tersedia di `training/train.py` dan menu `main.py`.

## 15. Cara Menjalankan Training via Terminal
### Opsi A: Menu Dinamis
```bash
python3 main.py
```

### Opsi B: CLI Dinamis Langsung
Contoh satu model:
```bash
python3 training/train.py --dataset parkinson_merder --models mobilenetv2 --method baseline --preprocessing-mode augment
```

Contoh semua model + satu method:
```bash
python3 training/train.py --dataset parkinson_mixing --models all --method transfer_learning --preprocessing-mode no_augment
```

Contoh semua model + semua method:
```bash
python3 training/train.py --dataset parkinson_multiclass --models all --all-methods --preprocessing-mode both
```

Contoh pipeline penuh:
```bash
python3 training/train.py --dataset parkinson_multiclass --models all --all-methods --preprocessing-mode both --check-first --split-first --augment-info
```

## 16. Cara Menjalankan Dashboard
```bash
streamlit run web/app.py
```

## 17. Catatan Migrasi
1. Path artifact baru sekarang berbasis dataset:
   - dari: `report/<model>/<run_id>`
   - ke: `report/<dataset>/<model>/<run_id>`
2. Path trained model juga berubah:
   - dari: `trained_models/<model>/<run_id>`
   - ke: `trained_models/<dataset>/<model>/<run_id>`
3. Dashboard sudah mengikuti path baru.
4. Report lama boleh dihapus sesuai keputusan proyek saat ini.
