# Export to Windows

## 1. Analisa Kelayakan (Sebelum Menjalankan)

Status: **bisa dijalankan di Windows**, dengan catatan penting di dependency deep learning.

Kesimpulan cepat:
1. Pipeline terminal (`main.py`, `training/train.py`) secara arsitektur sudah kompatibel lintas OS.
2. Path virtualenv Windows (`.venv/Scripts/python.exe`) sudah didukung otomatis.
3. Opsi auto-shutdown setelah training selesai sudah mendukung Windows.
4. Risiko utama ada di stack GPU TensorFlow dan driver CUDA Windows.

Rekomendasi praktik terbaik:
1. Jika targetmu **stabil dan minim masalah**, gunakan **WSL2 Ubuntu** untuk training berat/GPU.
2. Jika tetap native Windows, mulai dari mode CPU atau batch kecil dulu untuk validasi pipeline.

## 2. Prasyarat Windows

1. Windows 10/11 (64-bit).
2. Python 3.11+ terpasang dan tersedia di terminal.
3. Git terpasang.
4. (Opsional) NVIDIA Driver + CUDA-compatible environment jika ingin akselerasi GPU.

## 3. Cek Kompatibilitas Dependency

Project ini memakai dependency utama:
- TensorFlow
- Ultralytics (YOLO)
- Streamlit

Catatan penting:
1. TensorFlow GPU di Windows native sering lebih sensitif dibanding Linux/WSL2.
2. Jika instalasi `tensorflow[and-cuda]` bermasalah, gunakan salah satu strategi:
   - Pindah ke WSL2 (disarankan untuk GPU).
   - Jalankan CPU-only dulu untuk validasi pipeline.

## 4. Langkah Instalasi di Windows (Native)

Gunakan PowerShell atau Command Prompt.

```powershell
cd "C:\path\to\parkinson_prediction"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika command `python` tidak dikenali, gunakan `py`:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Menjalankan Pipeline dari Terminal Windows

### A. Menu interaktif
```powershell
python main.py
```

### B. Training langsung via CLI
```powershell
python training\train.py --dataset parkinson_merder --models mobilenetv2 --method transfer_learning --on-existing ask
```

### C. Pipeline penuh model CNN
```powershell
python training\train.py --dataset parkinson_merder --models all --model-families cnn --all-methods --split-first --on-existing ask
```

### D. Pipeline penuh model Transformer
```powershell
python training\train.py --dataset parkinson_merder --models all --model-families transformer --all-methods --split-first --on-existing ask
```

### E. Auto-shutdown setelah training selesai
```powershell
python training\train.py --dataset parkinson_merder --models mobilenetv2 --method transfer_learning --shutdown-on-finish
```

Di Windows, shutdown memakai perintah internal:
`shutdown /s /t 0`

## 6. Menjalankan Dashboard di Windows

```powershell
streamlit run web\app.py
```

Lalu buka URL lokal dari output Streamlit (biasanya `http://localhost:8501`).

## 7. Troubleshooting Windows

1. `ModuleNotFoundError` atau import `src` gagal:
   - Pastikan menjalankan command dari root project.
   - Pastikan virtualenv aktif.

2. Install TensorFlow gagal:
   - Coba ulang setelah upgrade `pip`.
   - Jika tetap gagal, gunakan WSL2 Ubuntu untuk environment training.

3. Training lambat atau freeze:
   - Mulai dari satu model, satu method, dan dataset kecil.
   - Turunkan `batch_size`.

4. Auto-shutdown tidak jalan:
   - Jalankan terminal dengan hak akses user yang boleh shutdown.
   - Uji manual `shutdown /s /t 0` di terminal.

## 8. Rekomendasi Operasional

1. Validasi dulu end-to-end ringan (1 model, 1 method).
2. Setelah stabil, naikkan ke pipeline full CNN/Transformer.
3. Gunakan `--on-existing ask` supaya run lama tidak tertimpa tanpa sengaja.
4. Aktifkan `--shutdown-on-finish` hanya saat kamu yakin job training sudah benar.

