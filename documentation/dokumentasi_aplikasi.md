# Dokumentasi Aplikasi Prediksi Parkinson

## 1. Ringkasan
Aplikasi ini adalah pipeline end-to-end untuk klasifikasi gambar hand-drawing Parkinson vs Healthy, mulai dari pengecekan dataset, split data, augmentasi training on-the-fly, training multi-arsitektur, evaluasi, sampai dashboard prediksi berbasis Streamlit.

Pipeline utama:
1. `1.check_dataset.py` -> cek distribusi dataset original.
2. `2.split_data_testing.py` -> split train/testing/validation dari `dataset/original` ke `dataset/split`, sekaligus normalisasi file pada split `train` menjadi `224x224`.
3. `3.augmentasi.py` -> kebijakan augmentasi training on-the-fly untuk split `train` saja.
4. Training model CNN/Transformer/YOLOv8.
5. Monitoring hasil + inferensi di `web/app.py`.

## 2. Teknologi
- Python
- TensorFlow + Keras (model CNN + Transformer)
- PyTorch + Ultralytics YOLOv8
- Scikit-learn (metrik evaluasi)
- Streamlit (dashboard)

Dependency utama ada di `requirements.txt`.

## 3. Struktur Proyek (Ringkas)
```text
parkinson_prediction/
|- dataset/
|  |- original/
|  |- split/
|- model/
|  |- training_common.py
|  |- mobilenetv2.py
|  |- resnet50.py
|  |- vgg19.py
|  |- resnet152.py
|  |- inception_googlenet.py
|  |- efficientnet.py
|  |- densenet121.py
|  |- transformer_backbones.py
|  |- vit.py
|  |- swintransformer.py
|  |- deit.py
|  |- yolov8.py
|- report/
|- trained_models/
|- web/
|  |- app.py
|- main.py
|- documentation/
|  |- dokumentasi_aplikasi.md
```

## 4. Model yang Didukung
Model CNN dan Transformer menggunakan `training_common.py` (pipeline training, evaluasi, plotting, dan penyimpanan artifact seragam).

1. MobileNetV2
- Script: `model/mobilenetv2.py`
- `model_name`: `mobilenetv2`

2. ResNet50
- Script: `model/resnet50.py`
- `model_name`: `resnet50`

3. VGG19
- Script: `model/vgg19.py`
- `model_name`: `vgg19`

4. ResNet152
- Script: `model/resnet152.py`
- `model_name`: `resnet152`

5. Inception (GoogLeNet style)
- Script: `model/inception_googlenet.py`
- `model_name`: `inception_googlenet`
- Implementasi backbone memakai `InceptionV3` dari `tf.keras.applications`.

6. EfficientNet
- Script: `model/efficientnet.py`
- `model_name`: `efficientnet`
- Implementasi backbone memakai `EfficientNetB0`.

7. DenseNet121
- Script: `model/densenet121.py`
- `model_name`: `densenet121`

8. ViT (Vision Transformer)
- Script: `model/vit.py`
- `model_name`: `vit`

9. SwinTransformer
- Script: `model/swintransformer.py`
- `model_name`: `swintransformer`

10. DeiT
- Script: `model/deit.py`
- `model_name`: `deit`

11. YOLOv8
- Script: `model/yolov8.py`
- `model_name`: `yolov8`

## 5. Menjalankan Aplikasi

### A. Menjalankan Menu Utama
```bash
python main.py
```
Menu yang tersedia:
1. Instal dependency + virtual environment
2. Cek distribusi dataset
3. Split data
4. Kebijakan augmentasi training
5. Training MobileNetV2
6. Training ResNet50
7. Training VGG19
8. Training ResNet152
9. Training Inception (GoogLeNet style)
10. Training EfficientNet
11. Training DenseNet121
12. Training ViT (Vision Transformer)
13. Training SwinTransformer
14. Training DeiT
15. Training YOLOv8
16. Training semua model (CNN + Transformer + YOLOv8)
17. Jalankan pipeline penuh (2 -> 4 -> 16)
18. Jalankan dashboard Streamlit

### B. Menjalankan Training Langsung per Script
Contoh:
```bash
python model/vgg19.py
python model/resnet152.py
python model/inception_googlenet.py
python model/efficientnet.py
python model/densenet121.py
python model/vit.py
python model/swintransformer.py
python model/deit.py
```

Semua script CNN/Transformer menerima argumen training umum, contoh:
```bash
python model/vit.py --epochs 8 --fine-tune-epochs 2 --batch-size 16 --mixed-precision
```

### C. Menjalankan Dashboard
```bash
streamlit run web/app.py
```

## 6. Output Training
Setiap training menghasilkan dua kelompok artifact:

1. Artifact report
- Lokasi: `report/<model_name>/<run_id>/`
- Isi utama:
  - `evaluation_metrics.json`
  - `training_history.csv`
  - `classification_report.csv`
  - `confusion_matrix.csv`
  - `split_distribution.csv`
  - `training_curves.png`
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `evaluation_table.png`
  - `summary.txt`

2. Artifact model
- Lokasi: `trained_models/<model_name>/<run_id>/`
- Isi utama:
  - `best_model.keras` / `best_model.pt`
  - `final_model.keras` / `final_model.pt`
  - `class_names.json`

File `latest_run.txt` otomatis diperbarui pada folder model terkait untuk menandai run terbaru.

## 7. Dashboard Streamlit (`web/app.py`)
Dashboard memiliki 3 tab:

1. Dataset & Praprosesing
- Ringkasan distribusi `dataset/original` dan `dataset/split`.
- Augmentasi training berjalan on-the-fly pada split `train`, sehingga tidak menambah file baru di disk.

2. Training Report
- Menampilkan ringkasan run terbaru per model.
- Detail metrik + CSV + visualisasi untuk run yang dipilih.

3. Prediksi Model
- Upload 1 gambar.
- Prediksi lintas beberapa model sekaligus.
- Bandingkan label prediksi dan probabilitas antar model.

## 8. Catatan Implementasi
1. Fallback CPU otomatis tersedia saat GPU tidak ada/penuh (tergantung konfigurasi).
2. Mixed precision bisa diaktifkan via flag `--mixed-precision` untuk hemat memori GPU.
3. Input size default CNN/Transformer dan YOLOv8 adalah `224x224` (bisa diubah via argumen).
4. Augmentasi train CNN/Transformer dilakukan saat training dengan rotation kecil, translation kecil, zoom ringan, brightness/contrast ringan, Gaussian noise ringan, dan random erasing kecil.
5. Folder dasar model baru sudah disiapkan:
- `report/vgg19`, `report/resnet152`, `report/inception_googlenet`, `report/efficientnet`, `report/densenet121`, `report/vit`, `report/swintransformer`, `report/deit`
- `trained_models/vgg19`, `trained_models/resnet152`, `trained_models/inception_googlenet`, `trained_models/efficientnet`, `trained_models/densenet121`, `trained_models/vit`, `trained_models/swintransformer`, `trained_models/deit`

## 9. Troubleshooting Singkat
1. GPU tidak terdeteksi
- Pastikan driver CUDA/cuDNN + TensorFlow GPU sudah benar.
- Jika tetap tidak ada GPU, training dapat lanjut di CPU jika fallback aktif.

2. Out Of Memory (OOM)
- Kurangi `--batch-size`.
- Aktifkan `--mixed-precision`.
- Set `--gpu-memory-limit-mb` sesuai kapasitas.

3. Dataset split kosong
- Jalankan ulang urutan: check dataset -> split.

4. Dashboard tidak menemukan model
- Pastikan training sudah berhasil dan artifact ada di `trained_models/<model_name>/<run_id>/`.
