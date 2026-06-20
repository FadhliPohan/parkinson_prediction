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

## 9. Update Kode & Restart Dashboard (WSL / Linux) — `deploy.sh`

Untuk VM Windows yang diakses via **WSL Ubuntu**, dashboard sebaiknya dijalankan
di background agar tidak mati saat terminal/SSH ditutup. Skrip `deploy.sh`
(di root project) melakukan: `git pull` → restart Streamlit → jalan di background.

### A. Persiapan sekali saja
```bash
cd /path/ke/parkinson_prediction
chmod +x deploy.sh
# Jika skrip pernah error "bad interpreter: ...^M" (akibat CRLF dari Windows):
sed -i 's/\r$//' deploy.sh
```
> Catatan: `.gitattributes` sudah memaksa `*.sh` memakai LF, jadi hasil `git pull`
> berikutnya tidak akan ber-CRLF lagi.

### B. Pemakaian harian
```bash
./deploy.sh            # git pull + restart Streamlit (default)
./deploy.sh restart    # restart TANPA git pull
./deploy.sh stop       # hentikan Streamlit
./deploy.sh status     # cek status + URL
./deploy.sh logs       # pantau log Streamlit (tail -f)
```

Variabel opsional:
```bash
PORT=8502 ./deploy.sh            # ganti port (default 8501)
ADDRESS=127.0.0.1 ./deploy.sh    # batasi akses ke localhost (default 0.0.0.0)
STOP_TRAINING=1 ./deploy.sh      # ikut hentikan training train.py yang berjalan
```

### C. Perilaku terhadap proses background
1. `git pull` **tidak** memengaruhi training yang sedang berjalan (kode sudah
   dimuat ke memori). Perubahan baru hanya berlaku untuk training berikutnya.
2. Secara default `deploy.sh` **hanya me-restart Streamlit**; training (proses
   detached) dibiarkan jalan. Pakai `STOP_TRAINING=1` bila ingin menghentikannya.
3. Setelah Streamlit di-restart, tab Monitor mungkin tidak lagi menampilkan
   training lama (state UI hilang), tetapi prosesnya **tetap berjalan** dan
   lognya tetap bertambah di `report/_web_runs/train_*.log`.
4. Streamlit dijalankan dengan `nohup ... < /dev/null`, sehingga training yang
   diluncurkan dari web **tidak mewarisi tty** dan tidak akan menggantung di
   prompt konfirmasi (lihat juga flag `--non-interactive` di `train.py`).

### D. Catatan WSL
1. WSL2 mem-forward `localhost`, jadi buka `http://localhost:8501` dari browser
   Windows. Untuk akses dari perangkat lain di jaringan, pakai `ADDRESS=0.0.0.0`
   (default) dan buka IP WSL/Windows yang sesuai.
2. Untuk performa training, taruh project di filesystem **native WSL** (mis.
   `~/parkinson_prediction`), bukan di `/mnt/c/...` (akses lintas-FS lambat).
