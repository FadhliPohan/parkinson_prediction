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
4. `main.py` sebagai menu terminal dinamis (pipeline 1-12).
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
│  ├─ optimizer/                 # (baru) registry optimizer seragam
│  │  ├─ __init__.py             # get_optimizer(), list_optimizers(), peta YOLO
│  │  ├─ adam.py                 # wrapper Adam
│  │  └─ no_optimize.py          # baseline SGD plain (pembanding)
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
1. Dataset dipilih dari `DatasetRegistry` (gabungan auto-discovery `dataset/original/*` + override opsional `configs/datasets.yaml`).
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
2. `src/models/registry.py` memuat metadata model (`framework`, `script_path`, `preprocess_key`, `family`, `enabled`).
3. Menambah model baru cukup tambah config + worker script.
4. Family model dipakai untuk pipeline tersegmentasi:
   - `cnn`
   - `transformer`
   - `yolo`
5. Daftar model aktif per family:
   - **CNN (9 model)**: `mobilenetv2`, `resnet50`, `vgg16`, `vgg19`, `resnext50`,
     `resnet152`, `inception_googlenet`, `efficientnet`, `densenet121`.
   - **Transformer (3 model)**: `vit`, `swintransformer`, `deit`.
   - **YOLO (1 model)**: `yolov8`.
6. Setiap `preprocess_key` di `configs/models.yaml` terdaftar di
   `src/inference/predictor.py` (`MODEL_PREPROCESSORS`) sebagai dokumentasi
   pemetaan preprocessing per-model. **Preprocessing TIDAK diterapkan lagi saat
   inferensi**: layer `preprocess_input`/`Rescaling` sudah tertanam di dalam graph
   model TF saat training (`training_common.py`), sehingga inference hanya memberi
   piksel mentah RGB [0,255] dan model menormalisasi sendiri. Menerapkan ulang akan
   menyebabkan double preprocessing (BLOCKER-01, sudah diperbaiki 2026-06-04).

## 7. Method Registry dan Training Strategy
1. Method training terdaftar di `configs/training_methods.yaml`.
2. Parameter default terpusat di `configs/default_training.yaml`.
3. `src/training/strategies.py` melakukan merge bertingkat:
   - default training,
   - override method,
   - override user (CLI global),
   - override runtime per method (`epochs`, `batch_size`, `fine_tune_epochs`) jika diisi user di awal workflow.
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
   - `--model-families` (single, multi CSV, atau `all`)
   - `--method-overrides-json` (override runtime per method)
   - `--check-first`, `--split-first`, `--augment-info`
   - `--shutdown-on-finish`
   - `--on-existing {ask,retrain,skip}`
   - `--on-existing-split {ask,resplit,skip}`
3. Orchestrator mengeksekusi kombinasi terpisah dengan urutan:
   - `dataset -> augmentasi -> method -> model`.
4. Setiap kombinasi diberi `experiment_id` unik.
5. Jika kombinasi sudah pernah ditraining:
   - `ask` akan menanyakan konfirmasi training ulang sekali di awal workflow,
   - `retrain` langsung membuat run/model baru dengan timestamp baru,
   - `skip` melewati kombinasi tersebut.
6. Jika `--split-first` dipakai, split existing bisa diputuskan:
   - `ask` (tanya split ulang),
   - `resplit` (paksa split ulang),
   - `skip` (pakai split lama).
7. Validasi balance split dilakukan sebelum training (after_counts + train/testing/validation).
8. Training didispatch ke runner framework melalui `src/training/trainer.py`.
9. `main.py` menanyakan konfigurasi runtime per method di awal (epoch/batch/fine-tune), lalu meneruskannya ke orchestrator.
10. `main.py` menyediakan pipeline penuh khusus family `cnn` dan family `transformer`.
11. Jika opsi shutdown diaktifkan, orchestrator menjalankan perintah shutdown OS setelah workflow selesai.

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
5. Ringkasan workflow juga menyimpan parameter runtime utama per kombinasi:
   - `epochs`
   - `batch_size`
   - `fine_tune_epochs`

## 10. Alur Dashboard Streamlit
1. Dashboard berjalan di `web/app.py`.
2. Sumber data utama adalah report (`run_manifest.json` + metrics).
3. Report dibaca secara indexed oleh `build_experiment_index()`.
4. Tampilan report dipisah:
   - tab `Explorer` (dataset -> augmentasi -> method -> model -> run)
   - tab `Perbandingan` (antar method, model, augmentasi, ranking val accuracy).
5. Detail run menampilkan konfigurasi runtime yang dipakai (`epoch`, `batch`, `fine-tune`) agar eksperimen mudah diaudit.
6. Tab prediksi memuat model dari `model_dir` yang direkam di manifest.

## 11. Menambah Dataset Baru
1. Tambahkan folder baru di `dataset/original/<nama_folder_dataset>`.
2. Jalankan check dan split (dataset otomatis muncul di registry).
3. Opsional: tambah entry di `configs/datasets.yaml` jika ingin override `dataset_id`, `split_dir`, `class_mode`, atau parameter lain.
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
   - `family` (`cnn` / `transformer` / `yolo` / custom)
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
  --on-existing-split ask \
  --on-existing ask
```
3. Dashboard:
```bash
streamlit run web/app.py
```

## 16. Registry Optimizer (baru)
1. Optimizer terdaftar di `src/optimizer/` dengan antarmuka seragam:
   `get_optimizer(name, learning_rate, **cfg)`.
2. Optimizer aktif:
   - `adam` — Adam adaptif (default).
   - `no_optimize` — SGD plain tanpa tuning (baseline pembanding).
3. Aliran optimizer ke pipeline:
   - `configs/default_training.yaml` -> field `optimizer`.
   - dilewatkan sebagai training param -> `--optimizer` -> worker.
   - di `training_common.py` dipakai pada `build_model` (stage 1) dan recompile stage 2.
   - untuk YOLO dipetakan ke optimizer Ultralytics via `get_yolo_optimizer_name`.
4. Bisa dipilih dari CLI (`--optimizer`), config, dan menu Training di dashboard.

## 17. Split Data Dinamis (baru)
1. Preset terdaftar di `src/datasets/splitter.py` (`SPLIT_PRESETS`): `80-10-10`, `70-15-15`.
2. Helper `validate_split_ratios` memastikan total = 100% dan `testing == validation`.
3. Dipakai di `training/2.split_data_testing.py` dan `training/train.py`
   via `--split-preset` atau `--train-ratio/--test-ratio/--val-ratio`.
4. Seed tetap dikunci (default 42) agar reproducible.

## 18. Penyempurnaan Training (rekomendasi diterapkan)
1. Default epoch/patience dinaikkan, LR fine-tune diperbesar (lihat
   `analisis_training_rekomendasi.md`).
2. Callback (ModelCheckpoint/EarlyStopping/ReduceLROnPlateau) diselaraskan ke
   `val_accuracy` dan dibuat baru per stage (state tidak terbawa antar stage).
3. `class_weight` 'balanced' diterapkan saat `model.fit` untuk menekan bias kelas.
4. Output layer di-set `dtype="float32"` agar stabil saat mixed precision.
5. YOLO memakai cosine LR + warmup + label smoothing.
6. **Catatan**: perbaikan strategi Transformer (pretrained/resep khusus) DITUNDA;
   placeholder/TODO ada di kode dan dokumen rekomendasi (REC-09).

## 19. Penambahan Model VGG16 dan ResNeXt50 (2026-06-04)

Dua model CNN baru ditambahkan ke family `cnn`:

### VGG16
- **Script**: `model/legacy_or_wrappers/vgg16.py`
- **Backbone**: `tf.keras.applications.VGG16` (pretrained ImageNet tersedia)
- **Preprocessing**: `tf.keras.applications.vgg16.preprocess_input` (VGG-style mean subtraction)
- **Preprocess key** (inference): `"vgg16"`
- **Catatan**: Arsitektur identik dengan VGG19 tetapi lebih ringan (16 layer konvolutif vs 19).

### ResNeXt-50-32x4d
- **Script**: `model/legacy_or_wrappers/resnext50.py`
- **Backbone**: Implementasi kustom di `model/legacy_or_wrappers/resnext_backbones.py`
- **Preprocessing**: ResNet50-style (keluarga arsitektur sama)
- **Preprocess key** (inference): `"resnext50"`
- **Catatan arsitektur**:
  - `cardinality = 32`, `base_width = 4` → ResNeXt-50-**32x4d**
  - 4 stage (3-4-6-3 blok), output 2048 channel
  - Grouped convolution: `tf.keras.layers.Conv2D(groups=32)` tersedia di TF 2.x
  - ~23 juta parameter
  - Bobot ImageNet pretrained tidak tersedia untuk implementasi kustom ini;
    `build_model()` di `training_common.py` menangani fallback ke random init secara
    otomatis melalui mekanisme `try/except` yang sudah ada.
- **File terkait**:
  - `model/legacy_or_wrappers/resnext_backbones.py` — arsitektur + preprocessing fn
  - `model/legacy_or_wrappers/resnext50.py` — wrapper entry point
- **Mengapa kustom?** `tf.keras.applications` tidak menyediakan ResNeXt secara native.
  Pola implementasi mengikuti `transformer_backbones.py` (builder function dengan
  signature `include_top`, `weights`, `input_shape`).

### Perubahan file terdampak
| File | Perubahan |
|---|---|
| `model/legacy_or_wrappers/vgg16.py` | Baru — wrapper VGG16 |
| `model/legacy_or_wrappers/resnext_backbones.py` | Baru — implementasi ResNeXt50 |
| `model/legacy_or_wrappers/resnext50.py` | Baru — wrapper ResNeXt50 |
| `configs/models.yaml` | Tambah entry `vgg16` dan `resnext50` |
| `src/inference/predictor.py` | Tambah `"vgg16"` dan `"resnext50"` di `MODEL_PREPROCESSORS` |

---

## 20. Catatan Sinkronisasi
1. `documentation/architecture.md` adalah sumber utama arsitektur.
2. `documentation/arsitectur.md` adalah mirror isi agar kompatibel dengan penamaan lama.
