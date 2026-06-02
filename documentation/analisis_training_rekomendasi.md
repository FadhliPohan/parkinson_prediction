# Analisis Training & Rekomendasi — Parkinson Prediction

> Dibuat: 2026-06-03
> Reviewer: Claude Sonnet 4.6 (AI Code Expert — Python, Machine Learning, Computer Vision)
> Status: **Dokumen rekomendasi — tidak ada perubahan kode**

---

## 1. Ringkasan Eksekutif

Project ini sudah menerapkan **early stopping** pada kedua framework (TensorFlow dan YOLO). Namun terdapat beberapa isu pada konfigurasi parameter default yang berpotensi menyebabkan training berhenti terlalu dini, model yang tersimpan tidak optimal, dan hasil eksperimen yang kurang reproducible untuk medical imaging.

| Aspek | Status Saat Ini | Rekomendasi |
|---|---|---|
| Early Stopping TF | **Ada** — `val_loss`, patience=4 | Naikkan patience, selaraskan metric |
| Early Stopping YOLO | **Ada** — built-in patience=4 | Naikkan patience |
| Jumlah Epoch | Terlalu kecil (8 epoch) | Minimal 20–30 epoch |
| Callback Stage 2 | Objek tidak direset | Buat callback baru per stage |
| Monitoring Metric | Inkonsisten (EarlyStopping vs Checkpoint) | Selaraskan ke `val_accuracy` |
| LR Fine-tuning | 1e-5 terlalu konservatif | Sesuaikan per backbone |
| **Transformer (ViT/Swin/DeiT)** | **Selalu dari nol — bobot ImageNet diabaikan** | Pakai pretrained / epoch jauh lebih banyak / unfreeze sejak awal |
| Class weight (imbalance) | Tidak dipakai di loss | Pertimbangkan `class_weight` |
| Mixed precision | Output layer tidak `float32` | Set `dtype="float32"` di layer akhir |

---

## 2. Status Implementasi Early Stopping

### 2.1 TensorFlow Models (CNN & Transformer)

File: `model/legacy_or_wrappers/training_common.py`, baris 751–773

```python
# Stage 1 — callbacks yang dipakai
callbacks = [
    checkpoint_callback,                          # simpan best val_accuracy
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",                        # monitor val_loss
        patience=args.early_stopping_patience,     # default: 4
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=max(1, args.early_stopping_patience // 2),  # default: 2
        verbose=1,
    ),
]

# Stage 2 (fine-tuning) — menggunakan OBJEK CALLBACK YANG SAMA
history_stage2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=args.fine_tune_epochs,
    callbacks=callbacks,    # <-- sama persis dengan stage 1
    verbose=1,
).history
```

**Kesimpulan:** Early stopping **sudah ada** tetapi mengandung beberapa isu teknis yang dijelaskan di bagian 3.

> **Catatan penting (hasil verifikasi):** Model **CNN** (MobileNetV2, ResNet50/152, VGG19, EfficientNet, DenseNet121, Inception) memanggil `tf.keras.applications.*` sebagai `backbone_builder`, sehingga benar-benar memuat bobot **ImageNet** saat `use_pretrained=True`.
>
> Sebaliknya, model **Transformer** (ViT, SwinTransformer, DeiT) memakai builder custom di `model/legacy_or_wrappers/transformer_backbones.py` (`build_vit_backbone`, `build_swin_transformer_backbone`, `build_deit_backbone`). Builder ini **mengabaikan argumen `weights`** dan **selalu membangun arsitektur dari nol (random init)** — tidak ada bobot pretrained. Akibatnya transfer learning tidak berlaku untuk transformer (lihat ISU-07).

---

### 2.2 YOLOv8

File: `model/legacy_or_wrappers/yolov8.py`, baris 439–457

```python
train_result = model.train(
    data=str(data_dir),
    epochs=total_epochs,            # epochs + fine_tune_epochs digabung
    patience=args.early_stopping_patience,  # default: 4
    ...
)
```

**Kesimpulan:** Early stopping **sudah ada** melalui mekanisme bawaan Ultralytics. Namun tidak ada pemisahan stage 1 dan stage 2 pada YOLO — semua epoch digabung menjadi satu sesi training.

---

## 3. Isu yang Ditemukan

### ISU-01: Patience Terlalu Kecil Relatif Terhadap Jumlah Epoch

**Lokasi:** `configs/default_training.yaml`

```yaml
epochs: 8
early_stopping_patience: 4   # 50% dari total epoch
```

**Dampak:**
- Dengan 8 epoch dan patience 4, training bisa berhenti di epoch ke-5 jika tidak ada improvement sejak epoch 1.
- Pada model besar (ResNet152, DenseNet121, ViT, SwinTransformer), bobot ImageNet perlu epoch lebih banyak untuk adaptasi ke domain medical imaging (spiral Parkinson).
- Dengan rasio patience/epoch = 50%, separuh waktu training bisa terbuang untuk "menunggu" sebelum model benar-benar konvergen.

**Standar umum:** patience sebaiknya 15–25% dari total epoch yang direncanakan, bukan 50%.

---

### ISU-02: Inkonsistensi Monitoring Metric antara EarlyStopping dan ModelCheckpoint

**Lokasi:** `model/legacy_or_wrappers/training_common.py`, baris 751–758

```python
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor="val_accuracy",   # simpan model terbaik berdasarkan val_accuracy
    mode="max",
    save_best_only=True,
)
tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",       # hentikan training berdasarkan val_loss
    restore_best_weights=True,
)
```

**Dampak:**
- `ModelCheckpoint` menyimpan bobot terbaik berdasarkan **val_accuracy**.
- `EarlyStopping` me-restore bobot berdasarkan **val_loss** terbaik.
- Kedua kondisi ini **tidak selalu sama**. Epoch dengan val_accuracy tertinggi belum tentu memiliki val_loss terendah.
- Hasil akhir: model yang dimuat dari `best_model.keras` (checkpoint) bisa berbeda dengan model yang ada di memori setelah EarlyStopping.
- Setelah training selesai, kode mengambil model dari `best_model.keras` (baris 811–817), yang berarti bobot yang digunakan pada evaluasi adalah dari checkpoint (val_accuracy), bukan dari EarlyStopping restore (val_loss). Ini membingungkan dan berpotensi tidak konsisten antar run.

---

### ISU-03: State Callback Tidak Direset antara Stage 1 dan Stage 2

**Lokasi:** `model/legacy_or_wrappers/training_common.py`, baris 800–808

```python
# Stage 2 menggunakan objek callback yang sama dengan Stage 1
history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.fine_tune_epochs,
    callbacks=callbacks,   # objek lama, state lama
    verbose=1,
).history
```

**Dampak:**
- Objek `EarlyStopping` menyimpan state internal: `best` (nilai terbaik yang pernah dilihat), `wait` (counter epoch tanpa improvement), dan `stopped_epoch`.
- Jika Stage 1 berakhir dengan `wait=3` (hampir trigger), Stage 2 akan langsung berhenti setelah 1 epoch pertama tanpa improvement — bahkan jika model baru saja mulai fine-tuning.
- Objek `ReduceLROnPlateau` juga tidak direset: LR yang sudah turun di akhir Stage 1 akan tetap pada nilai rendah di awal Stage 2.

---

### ISU-04: YOLO Tidak Memiliki Two-Stage Training

**Lokasi:** `model/legacy_or_wrappers/yolov8.py`, baris 419

```python
total_epochs = max(1, int(args.epochs) + max(0, int(args.fine_tune_epochs)))
```

**Dampak:**
- Parameter `fine_tune_epochs` hanya dijumlahkan ke total epoch — tidak ada pemisahan antara feature extraction dan fine-tuning.
- Argumen `--fine-tune-learning-rate` dan `--fine-tune-freeze-ratio` tidak digunakan di YOLO (dicatat sebagai kompatibilitas saja).
- User yang mengira ada dua stage fine-tuning di YOLO akan mendapat hasil yang berbeda dengan ekspektasi.

---

### ISU-05: Learning Rate Fine-Tuning Terlalu Konservatif untuk Sesi Pendek

**Lokasi:** `configs/default_training.yaml`

```yaml
fine_tune_learning_rate: 0.00001   # 1e-5
fine_tune_epochs: 2                 # hanya 2 epoch
```

**Dampak:**
- Dengan LR 1e-5 dan hanya 2 epoch fine-tuning, perubahan bobot sangat kecil — hampir tidak ada dampak yang signifikan dari fine-tuning.
- Untuk backbone dengan ratusan layer (ResNet152 memiliki 152 layer, DenseNet121 memiliki 121 layer), LR 1e-5 dalam 2 epoch hampir setara dengan tidak melakukan fine-tuning sama sekali.
- Ini membuat metode `transfer_learning` dan `full_fine_tuning` menghasilkan model yang hampir identik dengan `baseline` dalam hal eksplorasi bobot backbone.

---

### ISU-06: ReduceLROnPlateau Bisa Memicu Terlalu Awal

**Lokasi:** `model/legacy_or_wrappers/training_common.py`, baris 767–773

```python
tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,                                          # LR turun 70%
    patience=max(1, args.early_stopping_patience // 2),  # default: 2 epoch
    verbose=1,
),
```

**Dampak:**
- Dengan patience=2 dan total epoch=8, LR sudah bisa turun di epoch ke-3.
- Setelah LR turun, model konvergen lebih lambat — ini bisa menyebabkan EarlyStopping trigger lebih cepat karena improvement menjadi sangat kecil.
- Efek berantai: `ReduceLROnPlateau` → LR kecil → improvement kecil → `EarlyStopping` trigger.

---

### ISU-07: Transformer Selalu Dilatih dari Nol — Transfer Learning Tidak Berlaku (KRITIS)

**Lokasi:** `model/legacy_or_wrappers/transformer_backbones.py` (builder) + `model/legacy_or_wrappers/training_common.py:336-356` (build_model)

```python
# build_model di training_common.py (untuk SEMUA model TF)
weights = "imagenet" if use_pretrained else None
base_model = backbone_builder(include_top=False, weights=weights, input_shape=input_shape)
base_model.trainable = False   # Stage 1: backbone DIBEKUKAN

# Tetapi build_vit_backbone() di transformer_backbones.py:
def build_vit_backbone(include_top=False, weights=None, input_shape=None):
    input_shape = _validate_builder_args(include_top, weights, input_shape, "ViT")
    # ... argumen `weights` TIDAK PERNAH DIPAKAI di sini ...
    # arsitektur dibangun 100% dari nol dengan random init
```

**Dampak (sangat serius untuk ViT, SwinTransformer, DeiT):**
- Builder transformer custom **mengabaikan** `weights="imagenet"`. Tidak ada bobot pretrained yang dimuat — backbone selalu random init.
- Pada **Stage 1**, `base_model.trainable = False` membekukan backbone. Untuk transformer, ini berarti backbone menghasilkan **fitur acak yang dibekukan**, dan hanya head klasifikasi (GAP + Dropout + Dense) yang dilatih di atas fitur acak tersebut → praktis tidak belajar apa pun yang berguna.
- Pada **Stage 2** (fine-tuning), backbone baru di-unfreeze, tetapi dengan `fine_tune_learning_rate=1e-5` selama hanya 2 epoch → bobot transformer hampir tidak bergerak.
- Konsekuensi: **method `baseline`, `transfer_learning`, `transfer_learning_mixed_precision`, dan `full_fine_tuning` menghasilkan hasil yang nyaris sama untuk transformer**, karena semuanya efektif training-from-scratch dalam jumlah epoch yang sangat kecil.
- Transformer terkenal "data-hungry": tanpa pretrained, ViT/Swin/DeiT membutuhkan dataset sangat besar atau epoch sangat banyak (puluhan hingga ratusan) plus augmentasi/regularisasi kuat. Pada dataset Parkinson yang kecil, transformer dari nol kemungkinan besar **underfit berat** dan kalah jauh dari CNN pretrained.

**Catatan:** ini bukan bug sintaksis — kode tetap berjalan — tetapi secara metodologis hasil transformer pada konfigurasi saat ini tidak dapat dipakai untuk perbandingan yang adil terhadap CNN pretrained.

---

### ISU-08: Tidak Ada Penanganan Class Imbalance di Loss Function

**Lokasi:** `model/legacy_or_wrappers/training_common.py:379-393` (build_loss_and_metrics) + `model.fit(...)` baris 777-783 dan 801-807

```python
# Loss standar tanpa bobot kelas
tf.keras.losses.BinaryCrossentropy()
tf.keras.losses.SparseCategoricalCrossentropy()

# model.fit() dipanggil TANPA argumen class_weight
model.fit(train_ds, validation_data=val_ds, epochs=..., callbacks=...)
```

**Dampak:**
- Splitter sudah melakukan balancing kelas minoritas pada level data, namun jika masih ada sisa ketidakseimbangan (umum pada dataset medis), loss tidak mengompensasinya.
- Untuk diagnosis Parkinson, false negative (penderita terklasifikasi sehat) jauh lebih berisiko. Tanpa `class_weight` atau loss yang sensitif kelas, model cenderung bias ke kelas mayoritas.

---

### ISU-09: Output Layer Tidak Dipaksa float32 saat Mixed Precision Aktif

**Lokasi:** `model/legacy_or_wrappers/training_common.py:364-367`

```python
if num_classes == 2:
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)
else:
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="prediction")(x)
```

**Dampak:**
- Saat `mixed_precision` aktif (method `transfer_learning_mixed_precision` dan `full_fine_tuning`), global policy menjadi `mixed_float16`, sehingga layer output menghasilkan `float16`.
- Praktik resmi Keras menganjurkan layer aktivasi terakhir di-set `dtype="float32"` untuk stabilitas numerik (terutama softmax/sigmoid + loss). Tanpa ini, ada risiko `NaN`/instabilitas pada sebagian kombinasi hardware.
- Rekomendasi: `Dense(..., dtype="float32", name="prediction")`.

---

## 4. Apakah Cara Training Sudah Benar?

**Jawaban: Secara struktur sudah benar, tetapi parameter default kurang optimal untuk medical imaging.**

Yang sudah benar:
- Arsitektur pipeline modular dan bersih.
- Early stopping ada di semua framework.
- ModelCheckpoint untuk menyimpan best model ada.
- CPU/GPU fallback ditangani dengan baik.
- Mixed precision tersedia.
- Augmentasi on-the-fly tersedia.
- Seed global di-set untuk reproducibility.
- Data split 80:10:10 dengan balancing kelas.

Yang perlu diperbaiki (parameter, bukan struktur):
- Default epoch terlalu kecil.
- Patience terlalu kecil.
- Inkonsistensi monitoring metric.
- Callback tidak direset antar stage.
- Fine-tune LR terlalu kecil untuk sesi yang sangat pendek.

Yang perlu diperbaiki (metodologi — lebih serius dari sekadar parameter):
- **Transformer dilatih dari nol** sehingga method transfer learning tidak berdampak untuk family transformer (ISU-07). Ini menjadikan hasil transformer tidak sebanding dengan CNN pretrained.
- Tidak ada penanganan imbalance di loss (ISU-08).
- Output layer tidak `float32` saat mixed precision (ISU-09).

---

## 5. Rekomendasi Terperinci

### REC-01: Naikkan Default Epoch

**File:** `configs/default_training.yaml`

| Parameter | Nilai Saat Ini | Rekomendasi | Alasan |
|---|---|---|---|
| `epochs` | 8 | 25–30 | Medical imaging butuh lebih banyak iterasi |
| `fine_tune_epochs` | 2 | 8–10 | LR rendah butuh lebih banyak epoch |

Khusus untuk method `full_fine_tuning` di `configs/training_methods.yaml`:

| Parameter | Nilai Saat Ini | Rekomendasi |
|---|---|---|
| `fine_tune_epochs` | 6 | 15 |

---

### REC-02: Naikkan Early Stopping Patience

**File:** `configs/default_training.yaml`

| Parameter | Nilai Saat Ini | Rekomendasi | Alasan |
|---|---|---|---|
| `early_stopping_patience` | 4 | 8–10 | Rasio patience/epoch sebaiknya ≤25% |

Dengan epoch=25 dan patience=8, rasio menjadi 32% — masih lebih baik dari 50% saat ini.

Alternatif: setel patience secara proporsional. Contoh: patience = `epochs // 4`.

---

### REC-03: Selaraskan Monitoring Metric

**File:** `model/legacy_or_wrappers/training_common.py`

Pilih salah satu strategi dan terapkan konsisten:

**Opsi A — Monitor val_accuracy (direkomendasikan untuk klasifikasi)**
```python
# ModelCheckpoint
monitor="val_accuracy", mode="max"

# EarlyStopping
monitor="val_accuracy", mode="max"

# ReduceLROnPlateau
monitor="val_accuracy", mode="max"
```

**Opsi B — Monitor val_loss**
```python
# Semua callback
monitor="val_loss", mode="min"
```

Untuk tugas klasifikasi medis, **val_accuracy** lebih mudah diinterpretasikan dan langsung relevan dengan tujuan.

---

### REC-04: Reset Callback State antara Stage 1 dan Stage 2

**File:** `model/legacy_or_wrappers/training_common.py`

Buat objek callback baru untuk setiap stage, bukan menggunakan objek yang sama:

```python
# Stage 1
callbacks_stage1 = [
    tf.keras.callbacks.ModelCheckpoint(...),
    tf.keras.callbacks.EarlyStopping(...),
    tf.keras.callbacks.ReduceLROnPlateau(...),
]
model.fit(..., callbacks=callbacks_stage1, epochs=args.epochs)

# Stage 2 — objek BARU, state bersih
callbacks_stage2 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),   # tetap timpa best_model.keras
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=args.early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=max(1, args.early_stopping_patience // 2),
        verbose=1,
    ),
]
model.fit(..., callbacks=callbacks_stage2, epochs=args.fine_tune_epochs)
```

---

### REC-05: Sesuaikan Learning Rate Fine-Tuning

**File:** `configs/default_training.yaml`

| Parameter | Nilai Saat Ini | Rekomendasi | Catatan |
|---|---|---|---|
| `fine_tune_learning_rate` | 1e-5 | 5e-5 hingga 1e-4 | Lebih terasa dampaknya dalam 8–10 epoch |

Panduan umum per backbone:

| Model | Rekomendasi LR Fine-Tune |
|---|---|
| MobileNetV2, EfficientNet (ringan) | 1e-4 |
| ResNet50, VGG19, InceptionV3 | 5e-5 |
| ResNet152, DenseNet121 (besar) | 3e-5 |
| ViT, SwinTransformer, DeiT | 1e-5 hingga 3e-5 |

---

### REC-06: Pertimbangkan Cosine Annealing sebagai Alternatif ReduceLROnPlateau

`ReduceLROnPlateau` bersifat reaktif (LR turun ketika ada plateau), sedangkan `CosineAnnealingRestarts` bersifat proaktif dan lebih stabil untuk fine-tuning.

```python
# Alternatif scheduler yang lebih smooth
tf.keras.callbacks.LearningRateScheduler(
    lambda epoch, lr: lr * 0.5 ** (epoch // 5)   # halving setiap 5 epoch
)
```

Atau gunakan `tf.keras.optimizers.schedules.CosineDecay` langsung di optimizer.

---

### REC-07: Tambahkan Logging Epoch Aktual yang Dipakai

Setelah EarlyStopping, jumlah epoch aktual bisa berbeda dari yang dikonfigurasi. Pastikan `epochs_trained` yang disimpan di report menggunakan panjang history, bukan nilai `args.epochs`:

```python
# Sudah benar di kode saat ini:
"epochs_trained": int(len(history_df)),
```

Ini sudah diimplementasikan dengan benar — konfirmasi tidak perlu diubah.

---

### REC-08: Konfigurasi YOLO yang Lebih Eksplisit

**File:** `model/legacy_or_wrappers/yolov8.py`

Untuk YOLO, tambahkan parameter tambahan yang dapat meningkatkan hasil:

```python
model.train(
    ...
    # Parameter yang sudah ada
    patience=args.early_stopping_patience,
    lr0=args.learning_rate,

    # Parameter yang bisa ditambahkan
    lrf=0.01,        # learning rate final = lr0 * lrf (cosine decay bawaan YOLO)
    warmup_epochs=3, # warmup epoch agar LR naik bertahap di awal
    cos_lr=True,     # aktifkan cosine LR schedule
    label_smoothing=0.1,  # regularisasi untuk klasifikasi medis
)
```

---

### REC-09: Perbaiki Strategi Transformer (Prioritas Tinggi)

**File:** `model/legacy_or_wrappers/transformer_backbones.py` + `configs/training_methods.yaml`

Karena transformer custom selalu dari nol (ISU-07), pilih salah satu strategi berikut:

**Opsi A — Gunakan transformer pretrained (paling direkomendasikan).**
Ganti builder custom dengan implementasi pretrained, misalnya melalui library `keras-cv`, `transformers` (HuggingFace), atau `tfimm`. Dengan bobot ImageNet/Imagenet-21k, transformer bisa benar-benar memanfaatkan transfer learning seperti CNN.

**Opsi B — Jika tetap dari nol, sesuaikan resep training khusus transformer.**
- **Jangan bekukan backbone** pada Stage 1 (set `base_model.trainable = True` sejak awal) — membekukan backbone acak membuat Stage 1 sia-sia.
- Naikkan epoch jauh lebih banyak (50–150) dengan augmentasi kuat (RandAugment, Mixup/CutMix) dan regularisasi (weight decay, stochastic depth).
- Gunakan optimizer AdamW + warmup + cosine decay, LR awal ~1e-3 hingga 3e-4.
- Sadari batasannya: pada dataset medis kecil, transformer dari nol kemungkinan tetap kalah dari CNN pretrained. Dokumentasikan ini agar perbandingan eksperimen jujur.

**Opsi C — Nonaktifkan sementara family transformer** (`enabled: false` di `configs/models.yaml`) sampai strategi pretrained siap, agar tidak menghasilkan baris eksperimen yang menyesatkan.

---

### REC-10: Tambahkan Penanganan Class Imbalance

**File:** `model/legacy_or_wrappers/training_common.py`

Hitung `class_weight` dari distribusi train dan teruskan ke `model.fit`:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=np.array(train_labels),
)
class_weight = {i: float(w) for i, w in enumerate(weights)}

model.fit(..., class_weight=class_weight)
```

Catatan: untuk loss binary dengan label float, `class_weight` Keras tetap berlaku. Alternatif lain: focal loss untuk kasus imbalance berat.

---

### REC-11: Set Output Layer ke float32 saat Mixed Precision

**File:** `model/legacy_or_wrappers/training_common.py:364-367`

```python
if num_classes == 2:
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="prediction")(x)
else:
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="prediction")(x)
```

Aman diterapkan baik saat mixed precision aktif maupun tidak (pada float32 biasa, `dtype="float32"` tidak berdampak negatif).

---

## 6. Konfigurasi Rekomendasi Lengkap

Berikut rekomendasi nilai untuk `configs/default_training.yaml`:

```yaml
default_training:
  image_size: 224
  batch_size: 16
  epochs: 25                     # naik dari 8 → 25
  fine_tune_epochs: 10           # naik dari 2 → 10
  fine_tune_freeze_ratio: 0.7
  learning_rate: 0.001
  fine_tune_learning_rate: 0.00005   # naik dari 1e-5 → 5e-5
  dropout: 0.35
  early_stopping_patience: 8    # naik dari 4 → 8
  seed: 42
  ...
```

Rekomendasi untuk `configs/training_methods.yaml`:

```yaml
methods:
  baseline:
    arg_overrides:
      no_pretrained: true
      fine_tune_epochs: 0
      epochs: 30               # dari scratch butuh lebih banyak epoch
      early_stopping_patience: 10

  transfer_learning:
    arg_overrides:
      no_pretrained: false
      fine_tune_epochs: 10
      epochs: 25
      early_stopping_patience: 8

  transfer_learning_mixed_precision:
    arg_overrides:
      no_pretrained: false
      fine_tune_epochs: 10
      epochs: 25
      early_stopping_patience: 8
      mixed_precision: true

  full_fine_tuning:
    arg_overrides:
      no_pretrained: false
      fine_tune_epochs: 15
      epochs: 25
      early_stopping_patience: 10
      mixed_precision: true
```

---

## 7. Prioritas Implementasi

Urutkan berdasarkan dampak vs effort:

| Prioritas | Isu | Effort | Dampak |
|---|---|---|---|
| 1 | **Perbaiki strategi transformer (REC-09)** | Tinggi — ganti backbone / resep | **Sangat Tinggi** (untuk family transformer) |
| 2 | Naikkan epoch (REC-01) | Rendah — edit YAML | Tinggi |
| 3 | Naikkan patience (REC-02) | Rendah — edit YAML | Tinggi |
| 4 | Selaraskan monitoring metric (REC-03) | Sedang — edit training_common.py | Sedang |
| 5 | Reset callback antar stage (REC-04) | Sedang — edit training_common.py | Sedang |
| 6 | Sesuaikan LR fine-tuning (REC-05) | Rendah — edit YAML | Sedang |
| 7 | Class weight imbalance (REC-10) | Sedang — edit training_common.py | Sedang (klinis penting) |
| 8 | Output float32 mixed precision (REC-11) | Rendah — 2 baris | Rendah-Sedang (stabilitas) |
| 9 | YOLO tambahan parameter (REC-08) | Sedang — edit yolov8.py | Rendah-Sedang |
| 10 | Cosine LR scheduler (REC-06) | Tinggi — refactor | Rendah |

**Rekomendasi minimal yang paling efektif (tanpa sentuh kode):** Lakukan Prioritas 2 dan 3 (edit YAML — epoch & patience) untuk perbaikan cepat semua model.

**Rekomendasi paling berdampak (perlu ubah kode):** REC-09 — selama belum diperbaiki, hasil family transformer sebaiknya **tidak** dijadikan acuan perbandingan terhadap CNN.

---

## 8. Catatan Khusus untuk Medical Imaging (Parkinson)

Dataset Parkinson (spiral drawing) memiliki karakteristik khusus:

1. **Jumlah data sedikit** — dataset medis umumnya kecil (ratusan hingga ribuan gambar). Ini membuat setiap epoch lebih "berharga" dan model butuh lebih banyak epoch untuk konvergen.

2. **Kelas imbalanced** — penyakit vs sehat sering tidak seimbang. Sudah ditangani dengan balancing di splitter, namun perlu dipastikan data augmentasi juga membantu kelas minoritas.

3. **Feature yang subtle** — spiral tremor adalah perbedaan halus yang tidak terlihat di ImageNet. Transfer learning membutuhkan fine-tuning yang lebih dalam dan lebih lama dibanding domain lain.

4. **Validasi metric yang relevan** — untuk diagnosis medis, F1-score dan ROC-AUC lebih penting dari accuracy (terutama jika kelas imbalanced). Pertimbangkan menggunakan `monitor="val_f1"` dengan custom metric jika memungkinkan.

5. **Reproducibility** — seed sudah di-set dengan baik di semua framework. Ini penting untuk perbandingan eksperimen yang adil.

---

## 9. Kesimpulan

Early stopping **sudah diimplementasikan** di project ini, baik untuk TensorFlow (CNN + Transformer) maupun YOLO. Cara training yang dilakukan sudah mengikuti alur yang benar: split → augmentasi → training dengan callback → evaluasi → simpan artifact.

Namun ada **9 isu teknis** (ISU-01 s/d ISU-09) yang perlu diperhatikan, dengan **3 isu utama yang paling berdampak:**

1. **Transformer dilatih dari nol (ISU-07)** — builder ViT/Swin/DeiT mengabaikan bobot ImageNet, dan backbone dibekukan saat random init pada Stage 1. Akibatnya transfer learning tidak berlaku dan hasil family transformer tidak sebanding dengan CNN pretrained. **Ini temuan paling serius.**
2. **Parameter default terlalu kecil** (epoch=8, patience=4) — menyebabkan model tidak konvergen optimal.
3. **Inkonsistensi metric monitoring + state callback tidak direset** — menyebabkan perilaku training yang tidak deterministik antar stage.

Perbaikan termudah (tanpa sentuh kode): **edit `configs/default_training.yaml` dan `configs/training_methods.yaml`** untuk menaikkan `epochs` dan `early_stopping_patience`.

Perbaikan paling penting (perlu ubah kode): tangani strategi transformer (REC-09). Sampai itu dilakukan, perlakukan hasil family transformer sebagai "training-from-scratch", bukan transfer learning.
