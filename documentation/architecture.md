# Arsitektur Aplikasi Parkinson Prediction (Versi Dinamis v2)

## 1. Tujuan Arsitektur
Arsitektur ini dirancang agar pipeline machine learning:
1. Dinamis untuk pemilihan dataset, augmentasi, method training, dan model.
2. Modular dan mudah dipelihara.
3. Konsisten untuk artifact training dan report dashboard.
4. Aman untuk eksperimen karena setiap kombinasi punya jejak run terpisah.

## 2. Gambaran Umum
Komponen utama sistem:
1. `configs/*.yaml` sebagai sumber konfigurasi.
2. `src/` sebagai backend modular (dataset, model, training, reporting, inference, utils).
3. `training/train.py` sebagai orchestrator CLI kombinatorial.
4. `main.py` sebagai menu terminal dinamis (pipeline 1-10).
5. `web/app.py` sebagai dashboard pembaca report/model.

## 3. Struktur Folder
```text
parkinson_prediction/
├─ configs/
│  ├─ datasets.yaml
│  ├─ augmentations.yaml
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
│  │  ├─ augmentations.py
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
│  ├─ <dataset>/<augmentasi>/<method>/<model>/<run_id>/
│  └─ _workflow_runs/
├─ trained_models/
│  └─ <dataset>/<augmentasi>/<method>/<model>/<run_id>/
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
   ├─ pipeline.md
   ├─ history.md
   └─ requirements.md
```

## 4. Alur Dataset
1. Dataset dipilih dari `configs/datasets.yaml` melalui `DatasetRegistry`.
2. Tiap dataset bisa menentukan `augmentation_options` yang diperbolehkan.
3. Validasi kelas dilakukan oleh `src/datasets/validator.py` (`direct` atau `recursive_leaf`).
4. `src/datasets/splitter.py` melakukan split dengan balancing kelas minoritas (rotasi kecil) bila perlu.
5. Hasil split menyimpan metadata ke `dataset/split/<dataset_id>/_metadata/split_manifest.json`.

## 5. Registry Augmentasi
1. Augmentasi training terdaftar di `configs/augmentations.yaml`.
2. Registry dibaca oleh `src/training/augmentations.py`.
3. Kondisi saat ini:
   - `no_augment`
   - `augment_on_the_fly`
4. `training/train.py` bisa memakai:
   - `--augmentations no_augment,augment_on_the_fly`
   - atau kompatibilitas lama `--preprocessing-mode augment|no_augment|both`.

## 6. Model Registry
1. Model terdaftar di `configs/models.yaml`.
2. `src/models/registry.py` memuat metadata model (`framework`, `script_path`, `preprocess_key`, `enabled`).
3. Menambah model baru cukup tambah config + worker script.

## 7. Method Registry dan Training Strategy
1. Method training terdaftar di `configs/training_methods.yaml`.
2. Parameter default terpusat di `configs/default_training.yaml`.
3. `src/training/strategies.py` melakukan merge bertingkat:
   - default training,
   - override method,
   - override user (CLI).
4. Method aktif saat ini:
   - `baseline`
   - `transfer_learning`
   - `transfer_learning_mixed_precision`
   - `full_fine_tuning`

## 8. Alur Training CLI
1. Entry utama: `python3 training/train.py`.
2. Opsi penting:
   - `--dataset` (single, multi CSV, atau `all`)
   - `--augmentations` (single, multi CSV, atau `all`)
   - `--method` (single, multi CSV, atau `all`) atau `--all-methods`
   - `--models` (single, multi CSV, atau `all`)
   - `--check-first`, `--split-first`, `--augment-info`
   - `--on-existing {ask,retrain,skip}`
3. Orchestrator mengeksekusi kombinasi terpisah dengan urutan:
   - `dataset -> augmentasi -> method -> model`.
4. Setiap kombinasi diberi `experiment_id` unik.
5. Jika kombinasi sudah pernah ditraining:
   - `ask` akan menanyakan konfirmasi training ulang,
   - `retrain` langsung membuat run/model baru dengan timestamp baru,
   - `skip` melewati kombinasi tersebut.
6. Training didispatch ke runner framework melalui `src/training/trainer.py`.

## 9. Artifact Report dan Model
1. Report disimpan ke `report/<dataset>/<augmentasi>/<method>/<model>/<run_id>/`.
2. Model disimpan ke `trained_models/<dataset>/<augmentasi>/<method>/<model>/<run_id>/`.
3. File report utama:
   - `run_manifest.json`
   - `evaluation_metrics.json`
   - `evaluation_metrics.csv`
   - `training_history.csv`
   - `classification_report.csv`
   - `confusion_matrix.csv`
   - `split_distribution.csv`
   - visual `.png`
   - `summary.txt`
4. Ringkasan batch workflow disimpan di:
   - `report/_workflow_runs/workflow_summary_<timestamp>.json`
   - `report/_workflow_runs/workflow_summary_<timestamp>.csv`

## 10. Alur Dashboard Streamlit
1. Dashboard berjalan di `web/app.py`.
2. Sumber data utama adalah report (`run_manifest.json` + metrics).
3. Report dibaca secara indexed oleh `build_experiment_index()`.
4. Tampilan report dipisah:
   - tab `Explorer` (dataset -> augmentasi -> method -> model -> run)
   - tab `Perbandingan` (antar method, model, augmentasi, ranking val accuracy).
5. Tab prediksi memuat model dari `model_dir` yang direkam di manifest.

## 11. Menambah Dataset Baru
1. Tambahkan entry dataset di `configs/datasets.yaml`.
2. Isi minimal: `original_dir`, `split_dir`, `class_mode`, `split`, `seed`.
3. Tambahkan `augmentation_options` jika ingin membatasi opsi augmentasi per dataset.
4. Jalankan:
```bash
python3 training/1.check_dataset.py --dataset <dataset_id>
python3 training/2.split_data_testing.py --dataset <dataset_id>
```

## 12. Menambah Model Baru
1. Tambah worker script model (disarankan di `model/legacy_or_wrappers/`).
2. Tambah entry di `configs/models.yaml`:
   - `display_name`
   - `script_path`
   - `framework`
   - `preprocess_key`
   - `enabled`

## 13. Menambah Method Training Baru
1. Tambah method di `configs/training_methods.yaml`.
2. Isi `description` dan `arg_overrides`.
3. Method langsung bisa dipakai via CLI dan menu terminal.

## 14. Menambah Opsi Augmentasi Baru
1. Tambahkan entry augmentasi di `configs/augmentations.yaml`.
2. Pastikan field tersedia:
   - `label`
   - `description`
   - `disable_augmentation`
3. Tambahkan `augmentation_options` pada dataset terkait jika ingin diaktifkan untuk dataset tertentu saja.

## 15. Menjalankan Sistem
1. Menu terminal:
```bash
python3 main.py
```
2. CLI fleksibel (contoh):
```bash
python3 training/train.py \
  --dataset all \
  --augmentations all \
  --method all \
  --models resnet50,mobilenetv2,yolov8 \
  --split-first \
  --on-existing ask
```
3. Dashboard:
```bash
streamlit run web/app.py
```

## 16. Catatan Sinkronisasi
1. `documentation/architecture.md` adalah sumber utama arsitektur.
2. `documentation/arsitectur.md` adalah mirror isi agar kompatibel dengan penamaan lama.
