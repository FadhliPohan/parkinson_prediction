# Export to Linux

## 1. Analisa Kelayakan (Sebelum Menjalankan)

Status: **sangat direkomendasikan dijalankan di Linux** untuk workflow training yang panjang dan/atau GPU.

Kesimpulan cepat:
1. Arsitektur pipeline (`main.py`, `training/train.py`, `web/app.py`) kompatibel di Linux.
2. Path virtualenv Linux (`.venv/bin/python`) sudah didukung otomatis oleh runtime aplikasi.
3. Opsi auto-shutdown setelah training selesai sudah mendukung Linux.
4. Untuk training GPU, Linux umumnya lebih stabil dibanding Windows native.

## 2. Prasyarat Linux

1. Distro Linux 64-bit (disarankan Ubuntu 22.04/24.04 LTS atau distro setara).
2. Python 3.11+ tersedia di terminal.
3. `pip` dan `venv` tersedia.
4. Git tersedia.
5. Ruang disk cukup (minimal 20 GB disarankan untuk dataset + model + report).
6. RAM minimal 8 GB (disarankan 16 GB untuk training lebih nyaman).
7. (Opsional GPU) NVIDIA Driver yang kompatibel, dan `nvidia-smi` bisa dijalankan.

## 3. Instalasi Paket Sistem (Ubuntu/Debian)

Jalankan:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git libgl1 libglib2.0-0
```

Fungsi paket penting:
1. `python3`, `python3-pip`, `python3-venv` untuk environment Python.
2. `git` untuk clone/pull project.
3. `libgl1` dan `libglib2.0-0` untuk kompatibilitas library visi komputer (dipakai dependency YOLO/OpenCV).

## 4. Setup Environment Project di Linux

```bash
cd /path/to/parkinson_prediction
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Cek cepat setelah install:

```bash
python3 --version
pip --version
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

## 5. Menjalankan Pipeline dari Terminal Linux

### A. Menu interaktif
```bash
python3 main.py
```

### B. Training langsung via CLI
Ganti `parkinson_merder` dengan ID dataset yang muncul di registry.

```bash
python3 training/train.py --dataset parkinson_merder --models mobilenetv2 --method transfer_learning --on-existing ask
```

### C. Pipeline penuh model CNN
```bash
python3 training/train.py --dataset parkinson_merder --models all --model-families cnn --all-methods --split-first --on-existing ask
```

### D. Pipeline penuh model Transformer
```bash
python3 training/train.py --dataset parkinson_merder --models all --model-families transformer --all-methods --split-first --on-existing ask
```

### E. Auto-shutdown setelah training selesai
```bash
python3 training/train.py --dataset parkinson_merder --models mobilenetv2 --method transfer_learning --shutdown-on-finish
```

Di Linux, shutdown memakai:
`shutdown -h now`

## 6. Menjalankan Dashboard di Linux

```bash
streamlit run web/app.py
```

Lalu buka URL lokal dari output Streamlit (biasanya `http://localhost:8501`).

## 7. Troubleshooting Linux

1. `ModuleNotFoundError` atau import `src` gagal:
   - Jalankan command dari root project.
   - Pastikan virtualenv aktif (`source .venv/bin/activate`).

2. TensorFlow tidak mendeteksi GPU:
   - Cek `nvidia-smi`.
   - Cek dengan `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
   - Jika kosong, lanjutkan dulu mode CPU untuk validasi pipeline.

3. Error `libGL.so.1` saat training/inference:
   - Pastikan `libgl1` sudah terpasang.

4. Install dependency gagal:
   - Upgrade pip: `python3 -m pip install --upgrade pip`.
   - Coba ulang `pip install -r requirements.txt`.

5. Auto-shutdown tidak berjalan:
   - Uji manual `shutdown -h now`.
   - Pada beberapa sistem, perlu hak akses sudo/policy tertentu.

## 8. Rekomendasi Operasional

1. Validasi dulu end-to-end ringan (1 model, 1 method, 1 dataset).
2. Setelah stabil, naikkan ke pipeline multi-model dan multi-method.
3. Gunakan `--on-existing ask` agar retrain tidak terjadi tanpa konfirmasi.
4. Simpan eksperimen bertahap agar histori run mudah dianalisis dari folder `report/`.
