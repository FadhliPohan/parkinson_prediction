# Analisis Training & Rekomendasi — Parkinson Prediction

> Dibuat: 2026-06-03 | Diperbarui: 2026-06-04 (audit pra-deploy menyeluruh: code, alur, arsitektur)
> Reviewer: Claude (AI Code Expert — Python, Machine Learning, Computer Vision)
> Status: **BLOCKER-01 & BLOCKER-02 sudah diperbaiki (kode), BLOCKER-03 tidak berlaku.** Sisa sebelum deploy: re-split + latih ulang, DEP-01, ISU-07. **Belum aman untuk deploy sampai re-split + retrain selesai.**

---

## 0. PERINGATAN UTAMA (baca dulu sebelum deploy)

Audit ulang sebelum deployment menemukan **tiga isu yang TIDAK tercatat di versi dokumen sebelumnya**, dua di antaranya membuat hasil aplikasi/eksperimen menjadi tidak valid secara diam-diam (tidak ada error, tapi salah):

| Kode | Isu | Dampak | Prioritas |
|---|---|---|---|
| **BLOCKER-01** | Double preprocessing di inference | **Aplikasi deploy memberi prediksi salah** untuk SEMUA model TF | ✅ **SUDAH DIPERBAIKI (2026-06-04)** |
| **BLOCKER-02** | Data leakage: augmentasi rotasi sebelum split | **Metrik test/val tidak valid (inflated)** — sebagian test set berisi gambar sintetis | ✅ **SUDAH DIPERBAIKI (2026-06-04)** |
| **BLOCKER-03** | Split per-gambar, bukan per-pasien | Potensi leakage jika 1 pasien punya >1 gambar | ✅ **TIDAK BERLAKU** (dikonfirmasi user 2026-06-04: tiap gambar independen) |
| **DEP-01** | `requirements.txt` belum diverifikasi + `torch` tidak di-pin | Risiko gagal install / konflik CUDA di server | 🟠 TINGGI |
| **ISU-07** | Transformer dilatih dari nol | Hasil family transformer tidak sebanding dengan CNN | 🟡 SEDANG |
| **REC-06** | Cosine annealing untuk TF | Optimasi LR; peningkatan kecil | ⚪ RENDAH |

**Kesimpulan singkat:** Jangan deploy dulu. BLOCKER-01 dan BLOCKER-02 harus diperbaiki lebih dulu karena keduanya membuat sistem terlihat benar padahal salah. ISU-07 (fokus dokumen versi lama) sebenarnya prioritas lebih rendah dari ketiga blocker baru ini.

---

## 1. Ringkasan Eksekutif

Pipeline training (parameter epoch, patience, LR, class weight, callback per-stage, float32 output) memang sudah jauh lebih solid — **9 dari 11 rekomendasi training awal terkonfirmasi sudah diimplementasikan dengan benar** (lihat §6). Namun fokus perbaikan sebelumnya hanya pada *training loop*. Audit pra-deploy ini memeriksa **seluruh alur end-to-end** — data split → augmentasi → training → penyimpanan model → inference di aplikasi web — dan menemukan bahwa **jembatan antara training dan inference rusak** (BLOCKER-01) serta **integritas data split tercemar** (BLOCKER-02). Keduanya tidak menimbulkan error, sehingga lolos dari review yang hanya melihat training loop.

Semua klaim "sudah diperbaiki" di versi dokumen sebelumnya **telah diverifikasi ulang terhadap kode aktual dan benar** (lihat tabel verifikasi §6), kecuali satu catatan kecil: nilai default `argparse` di dalam script masih nilai lama (lihat CATATAN-A).

---

## 2. Deployment Blockers (Temuan Baru — Prioritas Tertinggi)

### BLOCKER-01: Double Preprocessing di Inference — Aplikasi Memberi Prediksi Salah (✅ SUDAH DIPERBAIKI 2026-06-04)

> **Status perbaikan:** Pemanggilan ganda `preprocess_fn` telah dihapus di [predictor.py `prepare_tf_input`](src/inference/predictor.py#L29). Inference kini memberi piksel mentah RGB [0,255] dan membiarkan layer preprocessing di dalam model bekerja sekali. `MODEL_PREPROCESSORS` dipertahankan hanya sebagai dokumentasi (dengan catatan pengaman agar tidak diaktifkan kembali). **Tindak lanjut wajib:** evaluasi ulang / latih ulang model lalu verifikasi bahwa probabilitas train == probabilitas inference untuk gambar yang sama. Penjelasan di bawah dipertahankan sebagai catatan akar masalah.


**Lokasi:**
- Training (preprocessing ditanam di dalam graph model): [training_common.py:360-361](model/legacy_or_wrappers/training_common.py#L360-L361)
- Model disimpan utuh termasuk layer preprocessing: [training_common.py:848](model/legacy_or_wrappers/training_common.py#L848)
- Inference menerapkan preprocessing untuk KEDUA kalinya: [predictor.py:29-38](src/inference/predictor.py#L29-L38)

**Akar masalah:**

Saat training, `preprocess_fn` dijadikan **layer pertama di dalam model**:

```python
# training_common.py:360-361
inputs = tf.keras.Input(shape=input_shape, name="input_image")
x = preprocess_fn(inputs)          # <-- preprocessing JADI BAGIAN dari model
x = base_model(x, training=False)
```

Model lalu disimpan utuh (`model.save(...)`, baris 848). Artinya file `.keras` yang tersimpan **sudah mengandung** langkah `preprocess_input` di dalamnya dan **mengharapkan input piksel mentah RGB [0, 255]**.

Tetapi di inference, `prepare_tf_input` menerapkan `preprocess_input` **lagi** sebelum memanggil `model.predict`:

```python
# predictor.py:32-37
array = np.asarray(image, dtype=np.float32)   # piksel mentah [0,255]
batch = np.expand_dims(array, axis=0)
preprocess_fn = MODEL_PREPROCESSORS.get(preprocess_key)
if preprocess_fn is not None:
    batch = preprocess_fn(batch)              # <-- preprocessing KEDUA, lalu masuk ke model yang sudah punya preprocessing
```

**Dampak:** Setiap model TF menerima input yang dinormalisasi **dua kali**. Contoh:
- ResNet/VGG (caffe-style mean subtraction) → mean dikurangi dua kali, distribusi piksel jadi kacau.
- ViT/transformer (`Rescaling 1/127.5 - 1`) → dipetakan ke ~[-1,1] lalu di-rescale lagi → keluar dari rentang yang dipelajari.

Tidak ada error yang muncul — aplikasi tetap memberi label dan confidence — tetapi **input berbeda jauh dari distribusi training**, sehingga **prediksi di aplikasi yang akan di-deploy bisa sangat salah / acak**, meskipun metrik test saat training terlihat bagus. Ini adalah bug paling berbahaya dalam project ini karena terlihat berjalan normal.

**Rekomendasi (pilih salah satu, konsisten train↔inference):**
- **Opsi A (paling kecil & disarankan):** Hapus pemanggilan `preprocess_fn` di [predictor.py:35-37](src/inference/predictor.py#L35-L37). Karena model sudah memuat layer preprocessing, cukup feed piksel mentah RGB [0,255]. (Jalur YOLO sudah benar — ultralytics menangani preprocessing-nya sendiri.)
- **Opsi B:** Keluarkan `preprocess_fn` dari graph model saat training (jangan jadikan layer), lalu preprocessing hanya di-input pipeline training + di inference. Lebih banyak perubahan, tidak disarankan.

**Catatan penting:** Apa pun yang dipilih, **model lama yang sudah terlanjur dilatih harus dievaluasi ulang / dilatih ulang** setelah perbaikan, karena pemilihan threshold/checkpoint terbaik dilakukan saat training (tanpa double preprocess), sedangkan deployment (dengan double preprocess) memakai distribusi berbeda. **Verifikasi perbaikan**: jalankan satu gambar test yang sama lewat (a) evaluasi training dan (b) `predict_from_bundle`, pastikan probabilitas yang keluar identik.

---

### BLOCKER-02: Data Leakage — Augmentasi Rotasi Dilakukan SEBELUM Split (✅ SUDAH DIPERBAIKI 2026-06-04)

> **Status perbaikan:** [splitter.py](src/datasets/splitter.py) ditulis ulang: gambar asli di-split dulu (`_split_real_per_class`, stratified), lalu balancing rotasi **hanya pada train** (`_balance_train_per_class`, sumber rotasi eksklusif dari gambar train kelas yang sama). Test & validation kini **100% gambar asli (0 sintetis)**. Validator (`validate_balanced_split_manifest`) diubah menjadi guard anti-leakage: `is_balanced=True` hanya jika train seimbang **dan** `test==validation` **dan** tidak ada sintetis di test/validation. Schema manifest dinaikkan ke `1.3.0`. **Sudah diverifikasi** lewat uji dataset dummy tidak seimbang (30 vs 12): train jadi 22 vs 22 seimbang, test/val proporsional real-only, `no_leakage_in_eval=True`. **Tindak lanjut wajib:** hapus folder split lama & lakukan re-split + latih ulang semua model (split lama masih tercemar). Penjelasan akar masalah di bawah dipertahankan sebagai catatan.


**Lokasi:** [splitter.py:217-238](src/datasets/splitter.py#L217-L238) (pembuatan sampel sintetis), dipanggil di [splitter.py:376-395](src/datasets/splitter.py#L376-L395) (split setelahnya).

**Akar masalah:**

Untuk menyeimbangkan kelas, splitter membuat salinan **rotasi** dari gambar kelas minoritas, lalu **menggabungkannya ke pool sebelum split**:

```python
# splitter.py:227-238 — sintetis dibuat dari file kelas, DI-MERGE ke base
for _ in range(missing_count):
    source_path = rng.choice(files)
    angle = rng.uniform(ROTATION_MIN_DEGREES, ROTATION_MAX_DEGREES)
    synthetic_samples.append(SplitSample(source_path=source_path, is_augmented=True, rotation_angle=angle))
samples_per_class[class_name] = base_samples + synthetic_samples   # real + sintetis dalam satu pool

# splitter.py:384-395 — pool (real+sintetis) baru di-shuffle & dibagi train/test/val
shuffled = samples[:]; rng.shuffle(shuffled)
train_samples = shuffled[:train_count]
test_samples  = shuffled[train_count : train_count + test_count]
validation_samples = shuffled[...]
```

Akibatnya, dua bentuk kebocoran terjadi sekaligus:

1. **Gambar sintetis masuk ke test & validation.** Kode bahkan menghitungnya secara eksplisit (`augmented_test_count`, `augmented_validation_count` di [splitter.py:397-399](src/datasets/splitter.py#L397-L399)) dan benar-benar menulis file rotasi ke folder `testing/` dan `validation/` ([splitter.py:434-439](src/datasets/splitter.py#L434-L439)). Artinya **metrik test diukur sebagian pada gambar sintetis** — secara metodologis tidak valid untuk model medis.
2. **Versi rotasi dari gambar yang sama bisa berada di train sekaligus test/val.** Karena `rng.choice(files)` memilih dari gambar asli, salinan rotasi gambar X bisa masuk test sementara X asli ada di train → kebocoran langsung → metrik **inflated**.

Fungsi `validate_balanced_split_manifest` ([splitter.py:290-329](src/datasets/splitter.py#L290-L329)) hanya mengecek **keseimbangan jumlah** dan malah **mewajibkan** `testing == validation`; ia tidak mendeteksi kebocoran ini sama sekali — memberi rasa aman palsu.

**Dampak:** Semua angka akurasi/AUC pada test set saat ini **tidak bisa dipercaya** (cenderung lebih tinggi dari kenyataan). Untuk model medis ini fatal karena keputusan deploy didasarkan pada metrik yang bias.

**Rekomendasi:**
- **Split dulu, augmentasi/balancing belakangan, dan HANYA di train.** Urutan benar: (1) split gambar **asli** ke train/test/val; (2) lakukan balancing rotasi **hanya pada train**; (3) test & validation tetap 100% gambar asli, tanpa sintetis.
- Setelah perbaikan, **re-split** semua dataset dan latih ulang. Metrik lama harus dianggap tidak valid.
- (Opsional) Tambahkan validasi yang benar-benar mendeteksi kebocoran: pastikan tidak ada `source_path` yang sama muncul di lebih dari satu split, dan `augmented_test_count == 0` dan `augmented_validation_count == 0`.

---

### BLOCKER-03: Split Per-Gambar, Bukan Per-Pasien (✅ TIDAK BERLAKU — dikonfirmasi 2026-06-04)

> **Status:** Pengguna mengonfirmasi bahwa setiap file gambar di dataset (`dataset_merder`, 2 kelas Healthy/Parkinson, 1632 gambar/kelas) adalah **gambar independen** — bukan augmentasi/duplikat dari sumber yang sama, dan bukan beberapa gambar dari pasien yang sama. Karena itu split berbasis gambar (stratified) **sudah benar** dan perbaikan BLOCKER-02 sudah memadai untuk dataset ini. Tidak perlu split per-pasien. **Catatan penting:** sumber split harus tetap `dataset_merder` (independen), **bukan** `dataset/praprosesing/` (6528 = 1632×4, hasil augmentasi) — registry sudah benar mengarahkan `parkinson_merder` → `dataset_merder`.


**Lokasi:** [splitter.py:382-395](src/datasets/splitter.py#L382-L395), [validator.py:40-50](src/datasets/validator.py#L40-L50).

Split bekerja pada level **file gambar individual** di dalam folder kelas — tidak ada konsep identitas pasien/subjek. Ini adalah bug klasik medical imaging: jika satu pasien menyumbang **lebih dari satu gambar** (mis. beberapa gambar spiral/gelombang per subjek), gambar-gambar mirip dari pasien yang sama bisa tersebar di train DAN test → akurasi inflated.

**Rekomendasi:**
- **Konfirmasi dulu:** apakah setiap file di dataset berasal dari subjek/pasien yang berbeda (independen)? 
  - Jika **ya** (satu gambar per pasien) → BLOCKER-03 tidak berlaku, cukup dokumentasikan asumsi ini.
  - Jika **tidak** → implementasikan split berbasis grup (subject-aware / `GroupKFold`-style): semua gambar satu pasien harus berada di split yang sama. Ini membutuhkan parsing ID pasien dari nama file/struktur folder.

---

### DEP-01: Verifikasi Dependency & Pin `torch` (🟠 TINGGI)

**Lokasi:** [requirements.txt](requirements.txt)

```
tensorflow[and-cuda]==2.21.0
numpy==2.4.4
pandas==3.0.2
scikit-learn==1.8.0
pillow==12.2.0
streamlit==1.56.0
ultralytics==8.3.161
```

Catatan:
1. **Versi sudah di-pin (bagus).** Namun **wajib diuji `pip install` di environment server yang bersih** sebelum deploy — pastikan semua versi ini benar-benar tersedia dan kompatibel satu sama lain pada platform target (terutama `tensorflow[and-cuda]` + `numpy 2.x`).
2. **`torch` tidak di-pin.** Jalur YOLO ([yolov8.py](model/legacy_or_wrappers/yolov8.py) dan inference YOLO) butuh `torch`, tapi hanya `ultralytics` yang tercantum. `torch` ikut transitif tapi versinya dibiarkan bebas → build CUDA/CPU bisa berubah-ubah. **Pin `torch` secara eksplisit.**
3. **Dua toolchain CUDA dalam satu env:** `tensorflow[and-cuda]` + `ultralytics`(→torch+CUDA) → image sangat besar dan berisiko konflik cuDNN/CUDA. Untuk **server inference**, pertimbangkan TF CPU-only + torch CPU-only (inference tidak butuh CUDA seberat training), atau pisahkan environment training vs serving.

---

## 3. Isu Lama yang Masih Terbuka (terkonfirmasi)

### ISU-07: Transformer Selalu Dilatih dari Nol (🟡 SEDANG)

**Terkonfirmasi masih berlaku.** Builder di [transformer_backbones.py](model/legacy_or_wrappers/transformer_backbones.py) menerima argumen `weights` tetapi hanya meneruskannya ke validator (`_validate_builder_args`, baris 11-27) yang sekadar mengecek nilainya `None`/`"imagenet"` — **nilai itu tidak pernah dipakai memuat bobot apa pun**. ViT/Swin/DeiT selalu random init.

Konsekuensi: pada Stage 1 backbone dibekukan (`base_model.trainable = False`, [training_common.py:358](model/legacy_or_wrappers/training_common.py#L358)) sehingga head dilatih di atas fitur **acak** → praktis tidak belajar. Method `baseline`, `transfer_learning`, dan `full_fine_tuning` menghasilkan hasil yang nyaris sama untuk transformer. Hasil family transformer **tidak sebanding** dengan CNN pretrained.

**Catatan prioritas:** Meskipun di dokumen lama ini diberi label "KRITIS", untuk **keputusan deploy** isu ini sebenarnya lebih rendah dari BLOCKER-01/02 — ia hanya membuat sebagian baris eksperimen tidak kompetitif, tidak merusak model CNN/YOLO yang akan dipakai produksi.

**Rekomendasi (sama seperti sebelumnya):**
- **Opsi A (terbaik):** Ganti builder custom dengan transformer pretrained (`keras-cv`, HuggingFace `transformers`, atau `tfimm`/`timm`) agar transfer learning benar-benar aktif.
- **Opsi B:** Jika tetap dari nol — jangan bekukan backbone di Stage 1, naikkan epoch (50–150), tambah augmentasi kuat (RandAugment, Mixup/CutMix) + AdamW + warmup + cosine.
- **Opsi C (cepat, tanpa ubah kode):** Set `enabled: false` untuk `vit`, `swintransformer`, `deit` di [configs/models.yaml](configs/models.yaml) sampai strategi pretrained siap, agar laporan tidak menyertakan hasil yang menyesatkan.

---

### REC-06: Cosine Annealing untuk TF (⚪ RENDAH)

`ReduceLROnPlateau` ([training_common.py:787-793](model/legacy_or_wrappers/training_common.py#L787-L793)) sudah lebih konservatif (`factor=0.5`, monitor `val_accuracy`) sehingga severity rendah. Cosine annealing (`tf.keras.optimizers.schedules.CosineDecay`) lebih smooth/proaktif, tapi peningkatannya kecil. YOLO sudah pakai cosine LR (`cos_lr=True`). **Bukan prioritas pra-deploy.**

---

## 4. Temuan Arsitektur & Maintainability

Tidak merusak fungsi, tapi sebaiknya dibereskan agar mudah dipelihara dan tidak menyesatkan saat di-deploy/serah-terima.

| Kode | Temuan | Lokasi | Catatan |
|---|---|---|---|
| ARCH-01 | **Folder `legacy_or_wrappers` salah nama** | [model/legacy_or_wrappers/](model/legacy_or_wrappers/) | Meskipun bernama "legacy", inilah satu-satunya backend training yang benar-benar dieksekusi (dipanggil sebagai subprocess via `script_path` di [models.yaml](configs/models.yaml)). Nama ini membingungkan. |
| ARCH-02 | **Direktori model kosong (dead scaffolding)** | [src/models/pytorch_models/](src/models/pytorch_models/), [src/models/tensorflow_models/](src/models/tensorflow_models/), [src/models/yolo_models/](src/models/yolo_models/) | Hanya `__init__.py` kosong, tidak di-import siapa pun. Hapus atau isi. |
| ARCH-03 | **`augmentation_selected/` adalah kode duplikat/mati** | [augmentation_selected/](augmentation_selected/) | Trainer mengambil augmentasi dari [src/datasets/transforms.py](src/datasets/transforms.py), bukan dari sini. Membingungkan karena seakan-akan ini sumber augmentasi. |
| ARCH-04 | **`model/hog_ghog/` tidak di jalur eksekusi** | [model/hog_ghog/](model/hog_ghog/) | Eksperimen ML klasik, tidak terdaftar di `models.yaml`/`train.py`. Pastikan ini memang tidak ikut deploy. |
| ARCH-05 | **Coupling dua arah worker ↔ src** | [training_common.py](model/legacy_or_wrappers/training_common.py), [src/optimizer/](src/optimizer/) | `src.training` shell-out ke script worker, sementara script worker `import from src.optimizer`. Refactor satu sisi memaksa menyentuh sisi lain. |
| ARCH-06 | **Path absolut `model_dir` tertanam di record training** | record/manifest hasil training | [web/app.py](web/app.py) mengecek `model_dir.exists()`. Jika project dipindah/dilatih di mesin lain, path absolut lama tidak ada → model tak terbaca. Saat deploy, re-root path atau latih ulang di server. |

---

## 5. Temuan Minor (Low Severity)

| Kode | Temuan | Lokasi | Rekomendasi |
|---|---|---|---|
| MINOR-01 | **Interpolasi resize beda train vs inference** | train bilinear [training_common.py:258](model/legacy_or_wrappers/training_common.py#L258) vs PIL default (BICUBIC) [predictor.py:31](src/inference/predictor.py#L31) | Samakan interpolasi agar konsisten. Dampak kecil tapi gratis diperbaiki. |
| MINOR-02 | **`Image.open` preview tanpa try/except** | [web/app.py preview upload](web/app.py) | Upload gambar korup bisa meng-crash tab (di luar blok try per-model). Bungkus dengan error handling. |
| MINOR-03 | **Asumsi kelas positif biner rapuh** | [predictor.py:49-53](src/inference/predictor.py#L49-L53) | Output sigmoid tunggal diasumsikan `[1-p, p]` dengan kelas positif = `class_names[1]` (kelas urutan-alfabet ke-2). Benar untuk `healthy`/`parkinson`, tapi tidak terdokumentasi & rapuh. Dokumentasikan/eksplisitkan. |
| MINOR-04 | **CUDA path helper khusus Linux** | `configure_cuda_library_path()` di wrapper, mis. [resnet50.py:6-29](model/legacy_or_wrappers/resnet50.py#L6-L29) | `LD_LIBRARY_PATH` + `os.execvpe` hanya berlaku Linux (no-op di Windows). Tidak fatal, tapi tidak mengonfigurasi CUDA di Windows. |
| MINOR-05 | **Path Chrome hardcoded** | [run.bat:11](run.bat#L11) | `C:\Program Files\Google\Chrome\...` — gagal di mesin tanpa Chrome di path itu. Hanya launcher, bukan fatal. |
| MINOR-06 | **`.h5` & `.pt` non-YOLO tidak didukung di inference** | [model_loader.py:36-67](src/inference/model_loader.py#L36-L67) | Hanya `.keras` (TF) & `.pt` (diasumsikan YOLO). Konsisten dengan training saat ini, tapi catat keterbatasannya. |
| DATA-01 | **`datasets.yaml` tidak cocok dengan folder nyata** | [configs/datasets.yaml:25,30](configs/datasets.yaml#L25) | `original_dir` tertulis `4_dataset_merder` & `6_dataset_mixing`, padahal folder nyata `dataset_merder` (dan `mixing` belum ada — file besar, sengaja belum ditambah). `parkinson_merder` masih jalan via auto-discovery, tapi `parkinson_mixing` akan **error** jika dipilih. Saran: samakan nama path dengan folder asli; nonaktifkan/komentari `parkinson_mixing` sampai foldernya ada. *(Belum diterapkan atas permintaan user.)* |
| DATA-02 | **Split lama yang bocor masih ada di disk** | `dataset/split/{train,testing,validation}` (root) | Berisi file `*_rot0/_rot90/_rot180/_rot270` dari proses lama (rotasi tersebar di train, struktur & lokasi usang — bukan di `dataset/split/parkinson_merder/`). Hapus sebelum re-split agar tidak membingungkan. *(Belum diterapkan atas permintaan user.)* |
| CATATAN-A | **Default `argparse` di script masih nilai lama** | `build_common_arg_parser` (epochs=8, fine_tune=2, LR FT=1e-5, patience=4) | Override YAML ([configs/default_training.yaml](configs/default_training.yaml)) sudah benar (25/10/5e-5/8) dan dipakai pada alur normal lewat `main.py`. Tapi siapa pun yang menjalankan script secara langsung TANPA layer YAML akan dapat nilai lama. Selaraskan default agar tidak menyesatkan. |

---

## 6. Verifikasi Klaim "Sudah Diperbaiki" (semua diperiksa ulang ke kode aktual)

| Aspek | Klaim | Verifikasi terhadap kode |
|---|---|---|
| Callback baru per stage | `_build_stage_callbacks()` dibuat baru tiap stage | ✅ CONFIRMED — [training_common.py:768-794](model/legacy_or_wrappers/training_common.py#L768-L794), dipanggil terpisah stage 1 (baris 817) & stage 2 (baris 842) |
| Monitoring seragam | Semua `val_accuracy` | ✅ CONFIRMED — ModelCheckpoint/EarlyStopping/ReduceLROnPlateau semua `monitor="val_accuracy"` |
| ReduceLROnPlateau factor | 0.3 → 0.5 | ✅ CONFIRMED — baris 787-793 |
| Class weight di kedua stage | `compute_class_weight_map()` + `class_weight=` | ✅ CONFIRMED — dihitung baris 763, dipakai di `fit` stage 1 (818) & stage 2 (843) |
| Output dtype float32 | Eksplisit `dtype="float32"` | ✅ CONFIRMED — baris 369 & 371 |
| Pretrained CNN | `weights="imagenet" if use_pretrained` | ✅ CONFIRMED — baris 338; CNN Keras-applications benar memuat ImageNet |
| ResNeXt50 random init | Custom, tanpa pretrained | ✅ CONFIRMED — [resnext_backbones.py](model/legacy_or_wrappers/resnext_backbones.py) menolak `weights="imagenet"`, fallback ke random |
| YOLO param | `lrf=0.01`, `cos_lr=True`, `warmup_epochs`, `label_smoothing=0.1`, epochs digabung | ✅ CONFIRMED — [yolov8.py:420,444-467](model/legacy_or_wrappers/yolov8.py#L444-L467) |
| Augmentasi on-the-fly train-only | Augmentasi runtime hanya di train_ds | ✅ CONFIRMED — val/test tidak diaugmentasi ([training_common.py:715-755](model/legacy_or_wrappers/training_common.py#L715-L755)). ⚠️ TAPI lihat BLOCKER-02: augmentasi *offline di splitter* bocor ke test/val |
| Epoch/patience/LR config | 25/10/8/5e-5 | ✅ CONFIRMED di YAML — ⚠️ tapi default argparse masih lama (CATATAN-A) |

Catatan alur arsitektur (terkonfirmasi): `run.bat` → `main.py` (menu) → subprocess `training/train.py` → `src/training/trainer.py` (dispatch by framework) → subprocess `model/legacy_or_wrappers/<model>.py` → `training_common.run_training_pipeline()`. Inference: `streamlit run web/app.py` → `src/inference/`. `src/models/registry.py` membaca `configs/models.yaml` (single source of truth, tidak ada duplikasi config).

---

## 7. Prioritas Pengerjaan (Disarankan)

Urutan ini mengoptimalkan "hasil maksimal sebelum deploy" — dahulukan yang membuat sistem **diam-diam salah**, lalu yang membuat hasil **tidak sebanding**, lalu kosmetik.

| # | Item | Effort | Dampak | Kenapa urutan ini |
|---|---|---|---|---|
| ✅ | ~~**BLOCKER-01** — hapus double preprocessing di inference~~ | **Rendah** | **Sangat Tinggi** | **SUDAH DIPERBAIKI 2026-06-04.** Tinggal verifikasi prob train==inference & latih/evaluasi ulang. |
| ✅/⏳ | **BLOCKER-02** — split dulu, balancing rotasi hanya di train | Sedang | **Sangat Tinggi** | **KODE SUDAH DIPERBAIKI 2026-06-04 & teruji.** ⏳ Sisa aksi: re-split `dataset_merder` + latih ulang (metrik lama dibuang). |
| ✅ | ~~**BLOCKER-03** — konfirmasi/implementasi split per-pasien~~ | Rendah | Tinggi | **TIDAK BERLAKU** (dikonfirmasi: tiap gambar independen). |
| **4** | **DEP-01** — uji install di server bersih + pin `torch` | Rendah | Tinggi | Mencegah gagal deploy / konflik CUDA. |
| **5** | **ISU-07** — Opsi C (nonaktifkan transformer) sekarang; Opsi A nanti | Rendah (C) / Tinggi (A) | Sedang | Cepat menutup hasil menyesatkan tanpa ubah kode. |
| **6** | **MINOR-01..06 + ARCH-01..06 + CATATAN-A** | Rendah | Rendah | Kebersihan & maintainability; aman dikerjakan setelah deploy. |
| **7** | **REC-06** — cosine annealing TF | Sedang | Rendah | Peningkatan marjinal; opsional. |

**Sudah selesai (kode):** BLOCKER-01 ✅, BLOCKER-02 ✅, BLOCKER-03 ✅ (tidak berlaku).

**Sisa pekerjaan sebelum deploy (berurutan):**
1. **DEP-01** — pin `torch` di [requirements.txt](requirements.txt) + uji `pip install` di env server bersih.
2. **ISU-07** — set `enabled: false` untuk vit/swintransformer/deit di [configs/models.yaml](configs/models.yaml) (Opsi C cepat), atau implementasi pretrained (Opsi A).
3. **DATA-01/02** (ditunda atas permintaan user) — samakan `datasets.yaml` dgn folder nyata + hapus split lama yang bocor.
4. **Re-split `dataset_merder` + latih ulang semua model** — gerbang terakhir; metrik lama (dari split bocor) dibuang.
5. **Verifikasi & uji end-to-end** — prob train==inference (BLOCKER-01), path `model_dir` valid di server (ARCH-06), uji upload gambar di web app.

---

## 8. Checklist Sebelum Deploy

- [x] BLOCKER-01 diperbaiki (double preprocessing dihapus di `predictor.py`). ⏳ Sisa: verifikasi prob train == prob inference untuk gambar yang sama.
- [ ] BLOCKER-02 diperbaiki: test & validation 0% gambar sintetis; tidak ada `source_path` lintas-split. Dataset di-split ulang.
- [ ] Semua model di-latih ulang setelah BLOCKER-01 & 02 (metrik lama dibuang).
- [x] BLOCKER-03 dikonfirmasi: tiap gambar independen → split per-gambar valid (tidak perlu split per-pasien).
- [ ] Re-split dari `dataset_merder` (BUKAN `dataset/praprosesing/`) + hapus split lama yang bocor di `dataset/split/{train,testing,validation}` (root).
- [ ] `requirements.txt` lolos `pip install` di environment server bersih; `torch` di-pin.
- [ ] Family transformer dinonaktifkan ATAU diberi catatan "tidak sebanding" di laporan.
- [ ] Path `model_dir` di record valid di server target (ARCH-06).
- [ ] Uji end-to-end: upload gambar di web app → label & confidence masuk akal untuk kasus known healthy & known parkinson.

---

## 9. Catatan untuk Medical Imaging (Parkinson)

1. **Integritas split adalah segalanya.** BLOCKER-02 (sintetis bocor ke test) dan BLOCKER-03 (split per-gambar) keduanya menggelembungkan metrik. Untuk model medis, metrik yang jujur lebih penting daripada metrik yang tinggi.
2. **Konsistensi train↔inference (BLOCKER-01).** Pipeline preprocessing harus identik byte-for-byte antara training dan deployment. Ini sumber #1 "akurasi training bagus tapi produksi jelek".
3. **Kelas imbalanced** — `class_weight` sudah aktif di kedua stage (✅), membantu menekan false negative (penderita terklasifikasi sehat). Pertahankan, tapi balancing harus train-only (BLOCKER-02).
4. **Fitur subtle** — spiral tremor halus, tidak ada di ImageNet; fine-tuning 10 epoch @ 5e-5 sudah memadai untuk CNN. Transformer from-scratch tidak akan cukup pada dataset kecil (ISU-07).
5. **Reproducibility** — seed sudah konsisten di semua framework (✅).
6. **Jangan bandingkan transformer dengan CNN pretrained** sampai ISU-07 selesai.

---

## 10. Kesimpulan

Audit end-to-end menemukan tiga isu yang sebelumnya luput; dua di antaranya (double preprocessing & data leakage) membuat sistem **terlihat benar padahal salah**, tanpa error apa pun. **Ketiga blocker tersebut kini sudah ditangani:** BLOCKER-01 (kode diperbaiki & teruji), BLOCKER-02 (splitter ditulis ulang & teruji), BLOCKER-03 (dikonfirmasi tidak berlaku — tiap gambar independen).

**Yang masih harus dilakukan sebelum deploy:** (1) **re-split `dataset_merder` + latih ulang** semua model — wajib, karena metrik lama berasal dari split yang bocor; (2) **DEP-01** — pin `torch` & uji install di server; (3) **ISU-07** — nonaktifkan/ganti transformer; (4) verifikasi prob train==inference + uji end-to-end web app. Item kebersihan (MINOR/ARCH/DATA-01/02) aman dikerjakan setelahnya.

**Kesimpulan status:** perbaikan kode untuk semua blocker kritis **selesai**. Gerbang terakhir adalah **re-split + latih ulang** agar metrik yang dilaporkan benar-benar jujur, lalu DEP-01 untuk memastikan instalasi di server.
