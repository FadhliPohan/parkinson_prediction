# Dokumentasi Aplikasi Prediksi Parkinson

## 1. Ringkasan
Project ini adalah pipeline end-to-end untuk klasifikasi gambar hand-drawing Parkinson vs Healthy. Versi terbaru mendukung dua keluarga eksperimen utama:

1. Deep learning berbasis transfer learning dan transformer.
2. Classical machine learning berbasis ekstraksi fitur HOG/GHOG.

Fokus pembaruan terbaru:
- Training komparasi yang lebih statis.
- Eksperimen bisa dijalankan tanpa augmentasi, dengan augmentasi kustom, atau keduanya sekaligus.
- Naming run lebih konsisten: `namametode_namaaugmentasi_tanggal`.
- Report lebih mudah dibaca karena menyimpan metadata eksperimen, split metrics, dan artifact yang lebih eksplisit.

Contoh nama run:
- `mobilenetv2_tanpa_augmentasi_20260507_110530`
- `mobilenetv2_augmentasi_kustom_20260507_111015`
- `svm_hog_tanpa_augmentasi_20260507_113200`
- `random_forest_ghog_augmentasi_kustom_20260507_114010`

## 2. Alur Pipeline Utama
Pipeline dasar project:

1. `1.check_dataset.py`
   Mengecek distribusi dataset awal pada `dataset/original`.

2. `2.split_data_testing.py`
   Membagi data ke `train`, `testing`, dan `validation`.

3. `3.augmentasi.py`
   Menampilkan profile augmentasi yang tersedia untuk eksperimen:
   - `without_augment`
   - `mycostum_augment`

4. Training model
   Terdiri dari:
   - Deep learning: CNN + Transformer
   - Classical ML: HOG/GHOG + SVM/KNN/Random Forest/MLP

5. Dashboard Streamlit
   Menampilkan dataset, komparasi report, dan prediksi lintas model.

## 3. Mode Augmentasi Eksperimen
Project sekarang memakai dua profile augmentasi utama:

### A. `without_augment`
- Tidak ada augmentasi tambahan pada split train.
- Digunakan sebagai baseline komparasi.
- Validation dan testing tetap memakai data asli.

### B. `mycostum_augment`
- Rotation kecil sekitar +/-20 derajat.
- Translation kecil sekitar +/-5%.
- Zoom ringan sekitar +/-8%.
- Brightness ringan sekitar +/-8%.
- Contrast ringan sekitar +/-10%.
- Gaussian noise ringan.
- Random erasing kecil.

### C. Sifat Augmentasi Sekarang
Augmentasi tidak lagi diposisikan sebagai proses acak yang terus berubah setiap epoch untuk eksperimen komparasi utama. Sebagai gantinya:

1. Split train dibaca.
2. Jika mode augmentasi aktif, versi augmentasi dibangkitkan secara statis per run.
3. Train final berisi:
   - gambar train asli
   - ditambah salinan train hasil augmentasi
4. Validation dan testing selalu dievaluasi tanpa augmentasi acak.

Tujuan pendekatan ini adalah membuat komparasi baseline vs augmentasi lebih stabil dan lebih mudah dianalisis.

## 4. Alur Deep Learning
Semua model CNN dan transformer memakai `model/training_common.py`.

### Alur Umum
1. Baca `dataset/split/train`, `dataset/split/validation`, dan `dataset/split/testing`.
2. Pilih profile augmentasi:
   - tanpa augmentasi
   - augmentasi kustom
   - semua mode
3. Jika augmentasi aktif:
   - train asli tetap dipakai
   - train augmentasi statis ditambahkan ke train final
4. Model dilatih pada train final.
5. Fine-tuning opsional dilakukan pada stage kedua.
6. Evaluasi dilakukan pada validation dan testing tanpa augmentasi acak.
7. Artifact disimpan ke `report/` dan `trained_models/`.

### Model Deep Learning yang Didukung

#### 1. MobileNetV2
- Script: `model/mobilenetv2.py`
- Cocok untuk baseline CNN yang ringan.
- Menggunakan transfer learning dengan backbone MobileNetV2 pretrained ImageNet.

#### 2. ResNet50
- Script: `model/resnet50.py`
- Mengandalkan residual connection untuk menjaga aliran gradien pada jaringan yang lebih dalam.

#### 3. VGG19
- Script: `model/vgg19.py`
- Arsitektur CNN klasik yang mudah dianalisis tetapi relatif berat.

#### 4. ResNet152
- Script: `model/resnet152.py`
- Versi lebih dalam dari keluarga ResNet untuk kapasitas representasi yang lebih besar.

#### 5. Inception (GoogLeNet style)
- Script: `model/inception_googlenet.py`
- Implementasi menggunakan backbone `InceptionV3`.
- Menggabungkan beberapa receptive field dalam satu blok.

#### 6. EfficientNet
- Script: `model/efficientnet.py`
- Menggunakan scaling depth, width, dan resolution secara lebih efisien.

#### 7. DenseNet121
- Script: `model/densenet121.py`
- Setiap blok menerima koneksi dari banyak layer sebelumnya sehingga reuse fitur lebih kuat.

#### 8. ViT (Vision Transformer)
- Script: `model/vit.py`
- Gambar dibagi menjadi patch lalu diproses sebagai token transformer.

#### 9. SwinTransformer
- Script: `model/swintransformer.py`
- Memakai window attention bertingkat untuk efisiensi komputasi.

#### 10. DeiT
- Script: `model/deit.py`
- Varian transformer vision yang lebih efisien untuk training data terbatas.

### Catatan YOLOv8
- Script: `model/yolov8.py`
- Tetap tersedia sebagai model terpisah.
- Tidak dimasukkan ke menu komparasi augmentasi utama karena pipeline eksperimennya berbeda dari CNN/Transformer dan HOG/GHOG.

## 5. Alur Classical ML dengan HOG/GHOG
Pipeline classical ML berada di `model/hog_ghog/`.

### Alur yang Dipakai
Pipeline mengikuti urutan ini:

1. Split data dulu.
2. Augmentasi hanya pada data train.
3. Ekstraksi fitur dari:
   - gambar train asli
   - gambar train hasil augmentasi
4. Scaling fitur dengan `StandardScaler`.
5. Training classifier:
   - SVM
   - KNN
   - Random Forest
   - MLP
6. Evaluasi pada validation dan testing tanpa augmentasi acak.

### HOG
HOG (`Histogram of Oriented Gradients`) mengekstrak distribusi arah gradien lokal. Fitur ini kuat untuk pola tepi, goresan, dan bentuk tulisan tangan.

### GHOG
GHOG di project ini didefinisikan sebagai versi HOG berbasis gradien yang diperkuat:

1. Gambar diubah ke grayscale.
2. Dihitung magnitude gradien.
3. Dibuat descriptor HOG pada grayscale asli.
4. Dibuat descriptor HOG kedua pada peta magnitude gradien.
5. Kedua descriptor digabung.

Dengan definisi ini, GHOG memberi representasi yang lebih kaya daripada HOG tunggal karena menangkap pola intensitas sekaligus pola gradien yang dipertegas.

### Model Classical yang Didukung

#### 1. SVM + HOG/GHOG
- Script: `model/hog_ghog/svm_hog_ghog.py`
- Cocok untuk data fitur vektor yang cukup terpisah secara margin.

#### 2. KNN + HOG/GHOG
- Script: `model/hog_ghog/knn_hog_ghog.py`
- Mengklasifikasikan sampel baru berdasarkan kedekatan ke tetangga pada ruang fitur.

#### 3. Random Forest + HOG/GHOG
- Script: `model/hog_ghog/rf_hog_ghog.py`
- Ensemble banyak pohon keputusan.
- Lebih robust terhadap non-linearity pada ruang fitur.

#### 4. MLP + HOG/GHOG
- Script: `model/hog_ghog/mlp_hog_ghog.py`
- Jaringan saraf feed-forward di atas fitur HOG/GHOG yang sudah di-scale.

## 6. Menjalankan Aplikasi

### A. Menu Utama
```bash
python main.py
```

Menu terbaru:
1. Instalasi dependency + virtual environment
2. Check distribusi dataset
3. Split data
4. Ringkasan profile augmentasi eksperimen
5. Training MobileNetV2
6. Training ResNet50
7. Training VGG19
8. Training ResNet152
9. Training Inception
10. Training EfficientNet
11. Training DenseNet121
12. Training ViT
13. Training SwinTransformer
14. Training DeiT
15. Training YOLOv8
16. Training semua model deep learning komparasi
17. Training SVM + HOG/GHOG
18. Training KNN + HOG/GHOG
19. Training Random Forest + HOG/GHOG
20. Training MLP + HOG/GHOG
21. Training semua model HOG/GHOG + Classical ML
22. Training semua eksperimen komparasi
23. Jalankan pipeline penuh
24. Jalankan dashboard Streamlit

### B. Training Deep Learning Langsung
Contoh:
```bash
python model/mobilenetv2.py --augmentation-profile all --augmentation-copies 1
python model/vit.py --augmentation-profile without_augment
python model/resnet50.py --augmentation-profile mycostum_augment --augmentation-copies 2
```

Argumen penting:
- `--augmentation-profile {without_augment,mycostum_augment,all}`
- `--augmentation-copies`
- `--epochs`
- `--fine-tune-epochs`
- `--batch-size`

### C. Training Classical ML Langsung
Contoh:
```bash
python model/hog_ghog/svm_hog_ghog.py --feature-extractor all --augmentation-profile all
python model/hog_ghog/knn_hog_ghog.py --feature-extractor hog --augmentation-profile without_augment
python model/hog_ghog/rf_hog_ghog.py --feature-extractor ghog --augmentation-profile mycostum_augment
python model/hog_ghog/mlp_hog_ghog.py --feature-extractor all --augmentation-profile all
```

Argumen penting:
- `--feature-extractor {hog,ghog,all}`
- `--augmentation-profile {without_augment,mycostum_augment,all}`
- `--augmentation-copies`
- `--hog-orientations`
- `--hog-pixels-per-cell`
- `--hog-cells-per-block`

### D. Menjalankan Dashboard
```bash
streamlit run web/app.py
```

## 7. Struktur Output Report dan Model

### A. Lokasi
- Report: `report/<model_name>/<run_name>/`
- Model: `trained_models/<model_name>/<run_name>/`

### B. File Report Utama
Umumnya setiap run baru menyimpan:
- `run_metadata.json`
- `experiment_summary.json`
- `evaluation_metrics.json`
- `split_metrics.json`
- `summary.txt`
- `split_distribution.csv`
- `split_distribution.png`
- `evaluation_table.png`
- `classification_report_validation.csv`
- `classification_report_testing.csv`
- `confusion_matrix_validation.csv`
- `confusion_matrix_testing.csv`
- `roc_curve_validation.png`
- `roc_curve_testing.png`

Tambahan untuk deep learning:
- `training_history.csv`
- `training_curves.png`

Tambahan untuk HOG/GHOG:
- `feature_summary.json`

### C. File Model Utama
Deep learning:
- `best_model.keras`
- `final_model.keras`
- `class_names.json`

YOLOv8:
- `best_model.pt`
- `final_model.pt`

Classical ML:
- `best_model.joblib`
- `final_model.joblib`
- `class_names.json`

## 8. Dashboard Streamlit
`web/app.py` sekarang memiliki tiga area utama:

### A. Dataset & Praprosesing
- Distribusi dataset original
- Distribusi split train/testing/validation
- Ringkasan mode augmentasi eksperimen

### B. Training Report
- Ringkasan run terbaru per model dan mode augmentasi
- Matriks komparasi accuracy dan F1-score
- Detail metadata eksperimen
- Split metrics validation vs testing
- Visualisasi confusion matrix, ROC, split distribution, dan kurva training bila tersedia

### C. Prediksi Model
- Mendukung perbandingan prediksi lintas model
- Dapat memuat:
  - model TensorFlow/Keras
  - YOLOv8
  - model classical ML `.joblib`

## 9. Catatan Analisis dan Desain
Beberapa keputusan desain penting:

1. Validation dan testing sengaja tidak di-augment secara acak.
   Tujuannya agar evaluasi merepresentasikan performa model pada data yang lebih stabil.

2. Train asli tetap dipertahankan saat augmentasi aktif.
   Ini membantu model tetap melihat distribusi asli sambil mendapat variasi tambahan.

3. Classical ML tetap memakai scaling untuk semua classifier.
   Walaupun Random Forest tidak selalu membutuhkan scaling, langkah ini diseragamkan agar pipeline komparasi konsisten.

4. GHOG didokumentasikan secara eksplisit.
   Karena istilah GHOG bisa bermakna berbeda pada berbagai paper, project ini mendefinisikannya secara lokal agar eksperimen bisa direproduksi.

5. YOLOv8 dipisahkan dari pipeline komparasi utama.
   Alasannya agar eksperimen baseline vs augmentasi untuk CNN/Transformer dan HOG/GHOG tetap konsisten secara metodologis.

## 10. Troubleshooting Singkat

### A. Dashboard tidak bisa memuat model classical ML
- Pastikan dependency sudah di-install ulang setelah update `requirements.txt`.
- Jalankan:
```bash
pip install -r requirements.txt
```

### B. HOG/GHOG gagal karena package belum ada
- Pastikan `scikit-image` sudah terinstall.

### C. Report lama tidak memiliki metadata baru
- Dashboard tetap mencoba membaca run lama.
- Run lama akan ditandai sebagai `Legacy / Unknown` pada bagian augmentasi.

### D. GPU kehabisan memori
- Kurangi `--batch-size`
- Kurangi `--augmentation-copies`
- Aktifkan `--mixed-precision`
- Biarkan fallback CPU aktif bila perlu
