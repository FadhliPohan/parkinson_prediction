# Arsitektur Aplikasi Parkinson Prediction (Versi Dinamis)

## 1. Tujuan Arsitektur
Arsitektur ini dirancang agar pipeline machine learning lebih:
1. Dinamis untuk pemilihan dataset, model, dan method training.
2. Modular dan mudah dipelihara.
3. Konsisten untuk artifact training dan report dashboard.
4. Aman untuk eksperimen karena seluruh run terdokumentasi.

## 2. Gambaran Umum
Komponen utama sistem:
1. `configs/*.yaml` sebagai sumber konfigurasi.
2. `src/` sebagai backend modular (dataset, model, training, reporting, inference, utils).
3. `training/train.py` sebagai orchestrator CLI.
4. `main.py` sebagai menu terminal dinamis.
5. `web/app.py` sebagai dashboard pembaca report/model.

## 3. Struktur Folder
```text
parkinson_prediction/
├─ configs/
│  ├─ datasets.yaml
│  ├─ models.yaml
│  ├─ training_methods.yaml
│  └─ default_training.yaml
│
├─ dataset/
│  ├─ original/
│  ├─ processed/
│  └─ split/
│
├─ src/
│  ├─ datasets/
│  │  ├─ registry.py
│  │  ├─ splitter.py
│  │  ├─ validator.py
│  │  └─ transforms.py
│  ├─ models/
│  │  ├─ registry.py
│  │  ├─ tensorflow_models/
│  │  ├─ pytorch_models/
│  │  └─ yolo_models/
│  ├─ training/
│  │  ├─ cli_args.py
│  │  ├─ strategies.py
│  │  ├─ trainer.py
│  │  ├─ tensorflow_trainer.py
│  │  ├─ pytorch_trainer.py
│  │  └─ yolo_trainer.py
│  ├─ reporting/
│  │  ├─ schemas.py
│  │  ├─ report_writer.py
│  │  ├─ report_reader.py
│  │  └─ plots.py
│  ├─ inference/
│  │  ├─ model_loader.py
│  │  └─ predictor.py
│  └─ utils/
│     ├─ config.py
│     ├─ paths.py
│     ├─ logging.py
│     └─ runtime.py
│
├─ model/
│  └─ legacy_or_wrappers/
├─ report/
│  └─ <dataset_name>/<model_name>/<run_id>/
├─ trained_models/
│  └─ <dataset_name>/<model_name>/<run_id>/
├─ training/
│  ├─ train.py
│  ├─ 1.check_dataset.py
│  ├─ 2.split_data_testing.py
│  └─ 3.augmentasi.py
├─ web/
│  └─ app.py
├─ main.py
└─ documentation/
   ├─ architecture.md
   ├─ arsitectur.md
   ├─ dokumentasi_aplikasi.md
   ├─ history.md
   └─ requirements.md
```

## 4. Alur Dataset
1. Dataset dipilih dari `configs/datasets.yaml` melalui `DatasetRegistry`.
2. Validasi kelas dilakukan oleh `src/datasets/validator.py` (`direct` atau `recursive_leaf`).
3. `src/datasets/splitter.py` melakukan proses berikut sebelum split:
   - Cek distribusi kelas pada dataset original.
   - Jika tidak seimbang, kelas minoritas diaugmentasi rotasi acak kecil (`-20` sampai `+20` derajat).
   - Jumlah data kelas minoritas dinaikkan sampai sama dengan kelas mayoritas.
4. Setelah balancing (jika perlu), data di-split ke `train/testing/validation` sesuai rasio config.
5. Hasil split menyimpan metadata ke `dataset/split/<dataset_id>/_metadata/split_manifest.json`.

## 5. Split Manifest (Schema v1.1.0)
`split_manifest.json` berisi:
1. Informasi umum: `schema_version`, `seed`, `original_dir`, `split_dir`, `class_mode`.
2. Rasio split pada `split_ratio`.
3. Statistik split per kelas pada `split_stats`.
4. Statistik jumlah gambar augmentasi yang masuk ke tiap split pada `split_generated_stats`.
5. Ringkasan balancing pada `class_balancing`:
   - `applied`
   - `strategy`
   - `rotation_range_degrees`
   - `target_per_class`
   - `before_counts`
   - `after_counts`
   - `generated_per_class`
   - `total_generated`

## 6. Model Registry
1. Model terdaftar di `configs/models.yaml`.
2. `src/models/registry.py` memuat metadata model (`framework`, `script_path`, `preprocess_key`, `enabled`).
3. Menambah model baru cukup:
   - tambah entry config,
   - sediakan worker script model.

## 7. Method Registry dan Training Strategy
1. Method training terdaftar di `configs/training_methods.yaml`.
2. Parameter default terpusat di `configs/default_training.yaml`.
3. `src/training/strategies.py` melakukan merge bertingkat:
   - default training,
   - override dari method,
   - override dari argumen CLI.
4. Method yang tersedia saat ini:
   - `baseline`
   - `transfer_learning`
   - `transfer_learning_mixed_precision`
   - `full_fine_tuning`

## 8. Alur Training CLI
1. Entry utama: `python3 training/train.py`.
2. Opsi penting:
   - `--dataset` (`parkinson_merder`, `parkinson_mixing`, atau `all`)
   - `--models`
   - `--method` atau `--all-methods`
   - `--preprocessing-mode {augment,no_augment,both}`
   - `--check-first`
   - `--split-first`
   - `--augment-info`
3. Saat `--split-first` aktif, pipeline otomatis menjalankan split + balancing sebelum training.
4. Training didispatch ke runner sesuai framework melalui `src/training/trainer.py`.

## 9. Artifact Report dan Model
1. Report disimpan ke `report/<dataset>/<model>/<run_id>/`.
2. Model disimpan ke `trained_models/<dataset>/<model>/<run_id>/`.
3. File report utama (schema `2.0.0` dari `src/reporting/schemas.py`):
   - `run_manifest.json`
   - `evaluation_metrics.json`
   - `evaluation_metrics.csv`
   - `training_history.csv`
   - `classification_report.csv`
   - `confusion_matrix.csv`
   - `split_distribution.csv`
   - visual `.png`
   - `summary.txt`

## 10. Alur Dashboard Streamlit
1. Dashboard berjalan di `web/app.py`.
2. Streamlit membaca report dan artifact model dari struktur berbasis dataset.
3. Dashboard fokus ke:
   - ringkasan dataset,
   - daftar eksperimen/run,
   - visualisasi metrik,
   - inferensi/prediksi model.

## 11. Menambah Dataset Baru
1. Tambahkan entry di `configs/datasets.yaml`.
2. Isi minimal: `original_dir`, `split_dir`, `class_mode`, `split`, `seed`.
3. Jalankan:
```bash
python3 training/1.check_dataset.py --dataset <dataset_id>
python3 training/2.split_data_testing.py --dataset <dataset_id>
```

## 12. Menambah Model Baru
1. Tambah worker script model (disarankan di `model/legacy_or_wrappers/` atau struktur yang disepakati).
2. Tambah entry di `configs/models.yaml`.
3. Pastikan field utama terisi:
   - `display_name`
   - `script_path`
   - `framework`
   - `preprocess_key`
   - `enabled`

## 13. Menambah Method Training Baru
1. Tambah method baru di `configs/training_methods.yaml`.
2. Isi `description` dan `arg_overrides`.
3. Method otomatis tersedia pada CLI dan menu terminal.

## 14. Menjalankan Sistem
1. Menu terminal:
```bash
python3 main.py
```
2. Training langsung via CLI:
```bash
python3 training/train.py --dataset all --models all --all-methods --preprocessing-mode both --split-first
```
3. Dashboard:
```bash
streamlit run web/app.py
```

## 15. Catatan Sinkronisasi
1. `documentation/architecture.md` adalah sumber utama dokumentasi arsitektur.
2. `documentation/arsitectur.md` dipertahankan sebagai mirror agar kompatibel dengan kebutuhan project sebelumnya.
