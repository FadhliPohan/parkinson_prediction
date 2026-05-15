# History Pengerjaan Proyek

Dokumen ini digunakan sebagai jejak kerja agar sesi bisa dilanjutkan dengan cepat jika percakapan terputus atau token habis.

## Format yang Dipakai
- Tanggal/Waktu
- Tujuan sesi
- Aktivitas yang sudah dilakukan
- Temuan penting
- Keputusan
- Next step

## 2026-05-15  (Asia/Jakarta) - Sesi 1

### Tujuan sesi
Membaca dan menganalisis kebutuhan proyek dari `documentation/requirements.md`, lalu menyiapkan fondasi dokumentasi histori pengerjaan.

### Aktivitas yang sudah dilakukan
1. Membaca penuh `documentation/requirements.md`.
2. Membaca dokumentasi aplikasi di `documentation/dokumentasi_aplikasi.md`.
3. Mengecek struktur folder project saat ini (dataset, model, report, trained_models, web, documentation, dan folder terkait lainnya).
4. Menyiapkan dokumen histori ini agar progres bisa dilanjutkan lintas sesi.

### Temuan penting
1. `requirements.md` berisi arahan refactor besar ke arsitektur yang lebih dinamis:
   - dataset registry/config,
   - model registry/config,
   - training method registry/config,
   - report schema standar,
   - terminal flow yang lebih modular,
   - Streamlit fokus sebagai pembaca report/artifact.
2. `requirements.md` juga menekankan urutan kerja:
   - analisis dulu,
   - ajukan pertanyaan klarifikasi,
   - baru implementasi setelah jawaban user.
3. `documentation/history.md` sebelumnya kosong, sehingga belum ada jejak progres lintas sesi.

### Keputusan
1. Menjadikan `documentation/history.md` sebagai sumber histori utama per sesi.
2. Setiap sesi berikutnya akan menambahkan entry baru (append), bukan menimpa entry lama.
3. Format dibuat ringkas agar mudah dipakai sebagai konteks saat melanjutkan pekerjaan.

### Next step
1. Menyusun hasil analisis struktur lama secara lebih terstruktur.
2. Menyusun daftar pertanyaan klarifikasi yang benar-benar krusial sebelum perubahan kode.
3. Setelah jawaban klarifikasi tersedia, lanjut ke rancangan arsitektur baru dan implementasi bertahap.

## 2026-05-15  (Asia/Jakarta) - Sesi 2

### Tujuan sesi
Implementasi refactor arsitektur dinamis sampai runnable: registry dataset/model/method, CLI training dinamis, report schema baru, dashboard dinamis, dan update dokumentasi.

### Aktivitas yang sudah dilakukan
1. Analisis ulang `documentation/requirements.md`, struktur project, pipeline training, dan dashboard.
2. Menjawab tahap analisis + mengumpulkan klarifikasi kebutuhan dari user.
3. Membuat struktur baru:
   - `configs/`
   - `src/datasets`
   - `src/models`
   - `src/training`
   - `src/reporting`
   - `src/inference`
   - `src/utils`
4. Membuat config:
   - `configs/datasets.yaml`
   - `configs/models.yaml`
   - `configs/training_methods.yaml`
   - `configs/default_training.yaml`
5. Refactor script lama:
   - `1.check_dataset.py` menjadi dinamis berbasis registry.
   - `2.split_data_testing.py` menjadi dinamis berbasis registry.
   - `3.augmentasi.py` ditambah argumen `--dataset-dir`.
6. Menambahkan CLI orchestrator baru: `train.py`.
7. Merombak `main.py` jadi menu terminal dinamis yang memanggil `train.py`.
8. Update `model/training_common.py` dan `model/yolov8.py`:
   - support argumen dataset/method/report-root/models-root,
   - output artifact ke format baru `report/<dataset>/<model>/<run_id>` dan `trained_models/<dataset>/<model>/<run_id>`,
   - menulis `run_manifest.json` schema v2.
9. Rewrite `web/app.py` agar membaca report/model dari struktur dinamis baru.
10. Update dependency `requirements.txt` dengan `PyYAML`.
11. Update dokumentasi:
   - `documentation/architecture.md`
   - `documentation/dokumentasi_aplikasi.md`

### Validasi yang dilakukan
1. Syntax check:
   - `python3 -m compileall -q .` -> sukses.
2. Validasi dataset check:
   - `python3 1.check_dataset.py --dataset parkinson_multiclass ...` -> sukses, terdeteksi 8 kelas.
3. Validasi split dataset:
   - `python3 2.split_data_testing.py --dataset parkinson_multiclass` -> sukses.
4. Smoke test training minimal (1 model, data kecil):
   - awalnya gagal pada metric AUC multi-class sparse label.
   - dilakukan perbaikan di `model/training_common.py` (hapus AUC metric saat training multi-class).
   - smoke test ulang berhasil dan menghasilkan report + model artifact baru.

### Temuan penting
1. Mode multi-class mengungkap bug metric training lama: `tf.keras.metrics.AUC(multi_label=True)` tidak kompatibel dengan label sparse pada setting ini.
2. Setelah perbaikan, training minimal berjalan sukses dan report schema baru terbentuk lengkap.

### Keputusan
1. Struktur report/model resmi dipindah ke skema berbasis dataset.
2. Dashboard membaca skema baru secara dinamis.
3. Training dari Streamlit tetap pending; terminal/CLI tetap utama.
4. Dokumentasi arsitektur dipusatkan ke satu file: `documentation/architecture.md`.

### Next step
1. Menjalankan training penuh (all model + all method) sesuai resource yang tersedia.
2. Membersihkan report/model lama (jika diperlukan) dan memulai eksperimen ulang dari skema baru.
3. Menambahkan method tambahan sesuai kebutuhan eksperimen lanjutan.
