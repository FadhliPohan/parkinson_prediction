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

## 2026-05-15  (Asia/Jakarta) - Sesi 3

### Tujuan sesi
Menyelesaikan pekerjaan yang tersisa, menyesuaikan struktur file dengan rekomendasi, dan membersihkan file/folder yang tidak terpakai.

### Aktivitas yang sudah dilakukan
1. Menambahkan modul augmentasi terpusat:
   - `src/datasets/transforms.py`
2. Merapikan `3.augmentasi.py` menjadi wrapper CLI tipis yang memanggil modul transform terpusat.
3. Mengubah `model/training_common.py` agar tidak lagi load augmentasi dari file script dengan `importlib`, tetapi langsung dari `src/datasets/transforms.py`.
4. Menambah struktur trainer per framework:
   - `src/training/cli_args.py`
   - `src/training/tensorflow_trainer.py`
   - `src/training/pytorch_trainer.py`
   - `src/training/yolo_trainer.py`
   - `src/training/trainer.py` di-refactor menjadi dispatcher runner per framework.
5. Menambahkan struktur folder rekomendasi:
   - `dataset/processed/.gitkeep`
   - `model/legacy_or_wrappers/README.md`
6. Cleanup file/folder tidak terpakai:
   - hapus `method/` (kosong)
   - hapus `main.ipynb` (kosong)
   - hapus `dataset_distribution.png` lama
   - hapus seluruh `__pycache__` dan `*.pyc`
   - hapus artifact report/model format lama (`report/<model>/...` dan `trained_models/<model>/...`), sisakan struktur baru berbasis dataset.
7. Update dokumentasi arsitektur dan aplikasi agar sinkron dengan struktur final.

### Temuan penting
1. Import TensorFlow di level modul transform menyebabkan `train.py`/`3.augmentasi.py` gagal di Python non-venv.
2. Solusi: lazy import TensorFlow di `src/datasets/transforms.py` sehingga fungsi ringkasan augmentasi tetap bisa dipakai tanpa TensorFlow aktif di interpreter tersebut.

### Validasi yang dilakukan
1. `python3 -m compileall -q .` -> sukses.
2. `python3 3.augmentasi.py --dataset-dir dataset/split/parkinson_multiclass` -> sukses.
3. `python3 train.py ... --augment-info ...` smoke test -> sukses, training tetap berjalan dan report/artifact baru terbentuk.

### Keputusan
1. Struktur pipeline dipertahankan dinamis dengan registry + trainer dispatcher per framework.
2. Script nomor (`1.check_dataset.py`, `2.split_data_testing.py`, `3.augmentasi.py`) tetap dipertahankan sebagai entrypoint CLI kompatibilitas, tetapi logika inti sudah dipusatkan ke `src/`.
3. Artifact lama dibersihkan agar dashboard dan report reader fokus pada skema baru.

### Next step
1. Menambahkan model tambahan ke registry setelah script worker siap.
2. Menjalankan full experiment (`all models + all methods`) pada dataset final.
3. Opsional: migrasi bertahap script worker legacy dari `model/` ke wrapper area `model/legacy_or_wrappers/` atau trainer native penuh di `src/training/`.

## 2026-05-15  (Asia/Jakarta) - Sesi 4

### Tujuan sesi
Menyamakan struktur folder/file agar lebih dekat ke struktur rekomendasi final di `requirements.md`.

### Aktivitas yang sudah dilakukan
1. Menambahkan sub-folder model framework pada `src/models/`:
   - `src/models/tensorflow_models/`
   - `src/models/pytorch_models/`
   - `src/models/yolo_models/`
2. Membuat `documentation/arsitectur.md` sebagai mirror dari `documentation/architecture.md`.
3. Membersihkan split legacy lama:
   - hapus `dataset/split/train`
   - hapus `dataset/split/testing`
   - hapus `dataset/split/validation`
   - mempertahankan hanya struktur baru `dataset/split/parkinson_multiclass/...`
4. Memindahkan seluruh script model legacy ke `model/legacy_or_wrappers/`.
5. Mengupdate `configs/models.yaml` agar seluruh `script_path` mengarah ke `model/legacy_or_wrappers/*.py`.
6. Menyesuaikan path root pada script legacy yang dipindah (`parents[1]` -> `parents[2]`) agar runtime tetap benar.
7. Menyesuaikan default `--dataset-dir` pada trainer legacy ke `dataset/split/parkinson_multiclass`.
8. Menghapus file/folder tidak terpakai:
   - `referensi_jurnal/`
   - PDF jurnal duplikat lama di root proyek
9. Sinkronisasi loader inference agar import custom object transformer tetap mengarah ke lokasi model legacy baru.

### Validasi yang dilakukan
1. `python3 1.check_dataset.py --dataset parkinson_multiclass` -> sukses.
2. `python3 3.augmentasi.py --dataset-dir dataset/split/parkinson_multiclass` -> sukses.
3. `python3 train.py ... --models mobilenetv2 --method baseline` smoke test -> sukses.

### Keputusan
1. Struktur `model/` dipusatkan untuk area legacy/wrapper saja.
2. Struktur split lama dibersihkan agar tidak membingungkan alur dataset baru.
3. Dokumentasi arsitektur disediakan dalam dua nama file (`architecture.md` dan `arsitectur.md`) untuk memenuhi kebutuhan eksplisit di requirement.

### Next step
1. Menjalankan training penuh lintas semua model/method pada struktur final.
2. Menambahkan model baru langsung ke registry jika dibutuhkan.

## 2026-05-15  (Asia/Jakarta) - Sesi 5

### Tujuan sesi
Menyelaraskan pipeline agar benar-benar dinamis sesuai kebutuhan user:
- pilihan dataset `merder` / `mixing` / `keduanya`,
- pilihan preprocessing `augment` / `tanpa augment` / `keduanya`,
- tetap kompatibel dengan menu terminal dan CLI.

### Aktivitas yang sudah dilakukan
1. Menambah konfigurasi dataset dinamis di `configs/datasets.yaml`:
   - `parkinson_merder`
   - `parkinson_mixing`
   - `parkinson_multiclass` (gabungan).
2. Menambah mode preprocessing dinamis di `training/train.py`:
   - argumen baru `--preprocessing-mode {augment,no_augment,both}`.
3. Menyesuaikan loop orchestrator training agar saat mode `both` menjalankan dua eksperimen otomatis (`aug_off` dan `aug_on`) untuk method/model yang sama.
4. Menambah dukungan flag `--disable-augmentation` pada arg builder (`src/training/cli_args.py`) dan meneruskannya ke worker model.
5. Mengubah worker TensorFlow (`model/legacy_or_wrappers/training_common.py`) agar augmentasi benar-benar bisa ON/OFF.
6. Mengubah worker YOLO (`model/legacy_or_wrappers/yolov8.py`) agar menerima opsi nonaktif augmentasi.
7. Menyesuaikan `main.py`:
   - prompt mode preprocessing di menu training,
   - otomatis kirim `--preprocessing-mode` ke `training/train.py`.
8. Memperbarui dokumentasi:
   - `documentation/dokumentasi_aplikasi.md`
   - `documentation/architecture.md`
   - `documentation/arsitectur.md` (sinkron/mirror).

### Validasi yang dilakukan
1. `python3 -m py_compile` pada file perubahan utama -> sukses.
2. Validasi dataset check:
   - `parkinson_merder` -> 2 kelas,
   - `parkinson_mixing` -> 6 kelas,
   - `parkinson_multiclass` -> 8 kelas.
3. Validasi split:
   - `parkinson_merder` -> sukses,
   - `parkinson_mixing` -> sukses.
4. Smoke test training:
   - `training/train.py --dataset parkinson_merder --preprocessing-mode both ...`
   - berhasil menjalankan dua run (`baseline__aug_off` dan `baseline__aug_on`) tanpa error.
5. Validasi menu `main.py`:
   - pilihan dataset tampil dinamis (merder/mixing/multiclass),
   - pilihan preprocessing tampil dinamis (augment/no_augment/both),
   - command training terbentuk sesuai mode terpilih.

### Temuan penting
1. Environment `.venv` saat ini belum memasang stack YOLO/PyTorch (`torch` belum ada), sehingga runtime YOLO belum diuji penuh pada sesi ini.
2. Jalur TensorFlow dan alur report/artifact sudah tervalidasi untuk mode preprocessing ON/OFF/BOTH.

### Keputusan
1. Kontrol dataset dan preprocessing dipusatkan pada config + orchestrator CLI agar tidak hard-code per script.
2. Untuk pembandingan eksperimen, run metadata membedakan mode preprocessing via suffix method (`__aug_off` / `__aug_on`).

### Next step
1. Jika ingin menguji YOLO end-to-end, lakukan instalasi dependency penuh dari `requirements.txt` di `.venv`.
2. Lanjutkan full experiment semua model + semua method + preprocessing mode sesuai resource komputasi.

## 2026-05-16  (Asia/Jakarta) - Sesi 6

### Tujuan sesi
1. Menambahkan balancing kelas otomatis sebelum split dataset.
2. Mengupdate seluruh dokumentasi agar sesuai kondisi code terbaru.

### Aktivitas yang sudah dilakukan
1. Mengupdate `src/datasets/splitter.py`:
   - menambahkan deteksi ketidakseimbangan kelas,
   - menambahkan augmentasi rotasi kecil (`-20` s/d `+20` derajat) khusus kelas minoritas,
   - menyejajarkan jumlah data tiap kelas ke kelas mayoritas sebelum split,
   - menambahkan metadata baru ke split manifest (`class_balancing`, `split_generated_stats`),
   - menaikkan `schema_version` split manifest ke `1.1.0`.
2. Mengupdate `training/2.split_data_testing.py` agar menampilkan ringkasan balancing kelas.
3. Mengupdate `training/train.py` agar output `--split-first` menampilkan status balancing.
4. Menjalankan validasi:
   - `python3 -m compileall -q ...` sukses,
   - split dataset `parkinson_multiclass` sukses,
   - uji mini dataset seimbang/tidak seimbang sukses.
5. Mengupdate dokumentasi berikut agar sinkron:
   - `documentation/architecture.md`,
   - `documentation/arsitectur.md` (mirror),
   - `documentation/dokumentasi_aplikasi.md`,
   - `documentation/requirements.md` (catatan status implementasi terbaru),
   - `documentation/history.md` (entry sesi ini).

### Temuan penting
1. Dataset `parkinson_multiclass` awal tidak seimbang, sehingga balancing aktif.
2. Balancing menambah data hanya pada kelas minoritas, sesuai kebutuhan user.
3. Informasi jumlah data hasil augmentasi kini terdokumentasi jelas di manifest split.

### Keputusan
1. Balancing kelas diletakkan di pipeline split agar berlaku konsisten untuk semua entrypoint (`train.py` maupun script split langsung).
2. Rentang rotasi augmentation balancing ditetapkan tetap pada `-20` s/d `+20` derajat.
3. Dokumen `architecture.md` dijadikan sumber utama; `arsitectur.md` dipertahankan sebagai mirror kompatibilitas.

### Next step
1. Jika diperlukan, tambahkan opsi konfigurasi untuk mengaktifkan/menonaktifkan balancing per dataset di `configs/datasets.yaml`.
2. Pertimbangkan balancing yang hanya diterapkan pada split train (opsional eksperimen lanjutan).

## 2026-05-16  (Asia/Jakarta) - Sesi 7

### Tujuan sesi
1. Menambahkan opsi training semua dataset secara berurutan.
2. Menghapus opsi dataset multiclass dari konfigurasi aktif.

### Aktivitas yang sudah dilakukan
1. Mengupdate `configs/datasets.yaml`:
   - menghapus entry `parkinson_multiclass`,
   - mengubah default dataset menjadi `parkinson_merder`.
2. Mengupdate `src/datasets/registry.py`:
   - urutan dataset mengikuti urutan pada file config (tidak di-sort alfabet),
   - fallback default dataset mengikuti dataset pertama pada config.
3. Mengupdate `training/train.py`:
   - menambahkan dukungan `--dataset all`,
   - menambahkan dukungan list dataset dipisah koma,
   - eksekusi training kini bisa loop beberapa dataset berurutan.
4. Mengupdate `main.py`:
   - pilihan dataset kini punya opsi `all (jalankan berurutan semua dataset)`,
   - aksi check/split/augment bisa dijalankan berurutan untuk semua dataset,
   - training dari menu meneruskan `--dataset all` ke orchestrator.
5. Menyesuaikan default pada script lain agar tidak lagi mengarah ke split multiclass:
   - `training/3.augmentasi.py`
   - `model/legacy_or_wrappers/training_common.py`
   - `model/legacy_or_wrappers/yolov8.py`
6. Memperbarui dokumentasi agar sinkron:
   - `documentation/architecture.md`
   - `documentation/arsitectur.md`
   - `documentation/dokumentasi_aplikasi.md`
   - `documentation/history.md`

### Keputusan
1. Mode `all` dijalankan sesuai urutan dataset pada `configs/datasets.yaml`, sehingga saat ini urutannya: `parkinson_merder` lalu `parkinson_mixing`.
2. Dataset multiclass dinonaktifkan dari opsi aktif sesuai permintaan user.

### Next step
1. Jika nanti dibutuhkan lagi, `parkinson_multiclass` bisa dikembalikan sebagai dataset opsional terpisah di config.
2. Tambahkan unit/integration test kecil untuk validasi mode `--dataset all` jika project ingin menambah coverage test otomatis.

## 2026-05-16  (Asia/Jakarta) - Sesi 8

### Tujuan sesi
Membuat dokumentasi pipeline eksekusi yang lengkap dan mudah dipahami, termasuk penjelasan menyeluruh untuk Pipeline 8.

### Aktivitas yang sudah dilakukan
1. Membaca ulang flow aktual di `main.py` dan `training/train.py`.
2. Membaca script pendukung pipeline:
   - `training/1.check_dataset.py`
   - `training/2.split_data_testing.py`
   - `training/3.augmentasi.py`
3. Membaca konfigurasi aktif dataset, model, method, dan default training.
4. Menyusun dokumen baru `documentation/pipeline.md` berisi:
   - peta pipeline 1-9,
   - tujuan, aktivitas, output, dan catatan tiap pipeline,
   - mapping menu ke command backend,
   - contoh skenario penggunaan,
   - troubleshooting singkat,
   - bedah lengkap Pipeline 8 (input, command, loop runtime, output, perilaku error, estimasi workload).

### Keputusan
1. Dokumentasi pipeline dipisah ke file khusus `documentation/pipeline.md` agar fokus operasional tidak bercampur dengan dokumen arsitektur.
2. Pipeline 8 dijelaskan dengan level detail eksekusi nyata agar bisa dipakai sebagai panduan eksperimen end-to-end.

### Next step
1. Jika diperlukan, tambahkan contoh template eksperimen standar (misalnya baseline ringan, benchmark menengah, benchmark penuh) di dokumen pipeline.
2. Sinkronkan `documentation/dokumentasi_aplikasi.md` agar menautkan dokumen `pipeline.md` sebagai referensi operasional utama.

## 2026-05-16  (Asia/Jakarta) - Sesi 9

### Tujuan sesi
Menyelaraskan seluruh dokumentasi dengan implementasi terbaru workflow kombinasi training dan report bertingkat.

### Aktivitas yang sudah dilakukan
1. Memperbarui `documentation/architecture.md`:
   - menambah registry augmentasi (`configs/augmentations.yaml`),
   - menambah modul `src/training/augmentations.py`,
   - memperbarui alur CLI kombinasi (`--dataset`, `--augmentations`, `--method`, `--models`),
   - memperbarui struktur artifact ke format bertingkat.
2. Menyamakan isi `documentation/arsitectur.md` sebagai mirror dari `architecture.md`.
3. Memperbarui `documentation/dokumentasi_aplikasi.md`:
   - command CLI terbaru,
   - output path report/model terbaru,
   - penjelasan tab report Streamlit (`Explorer` dan `Perbandingan`).
4. Memperbarui `documentation/pipeline.md`:
   - peta pipeline 1-10,
   - detail pipeline 10 workflow fleksibel,
   - flow runtime `training/train.py` berbasis kombinasi dataset -> augmentasi -> method -> model,
   - contoh command terbaru.
5. Memperbarui catatan status implementasi di `documentation/requirements.md` agar mencerminkan struktur artifact terbaru.

### Keputusan
1. `architecture.md` tetap dijadikan sumber utama dokumentasi arsitektur.
2. `arsitectur.md` dipertahankan sebagai mirror identik agar kompatibel dengan naming lama.
3. Dokumentasi operasional dipisah:
   - `pipeline.md` untuk flow eksekusi,
   - `dokumentasi_aplikasi.md` untuk panduan penggunaan harian.

### Next step
1. Jika diperlukan, tambahkan screenshot dashboard terbaru di dokumentasi untuk memperjelas alur Explorer dan Perbandingan.
2. Tambahkan contoh skenario benchmark bertahap (small/medium/full) di `pipeline.md`.

## 2026-05-16  (Asia/Jakarta) - Sesi 10

### Tujuan sesi
Menambahkan validasi kombinasi training yang sudah pernah dijalankan, lalu meminta konfirmasi retrain sesuai struktur folder artifact terbaru.

### Aktivitas yang sudah dilakukan
1. Validasi struktur folder artifact terkini:
   - `report/<dataset>/<augmentasi>/<method>/<model>/<run_id>/`
   - `trained_models/<dataset>/<augmentasi>/<method>/<model>/<run_id>/`
2. Menambahkan kontrol run existing di `training/train.py`:
   - argumen baru `--on-existing {ask,retrain,skip}`,
   - cek histori run per kombinasi dataset+augmentasi+method+model,
   - mode `ask` meminta konfirmasi retrain,
   - mode `retrain` langsung membuat run/model baru,
   - mode `skip` melewati kombinasi lama.
3. Menambahkan prompt di `main.py` sebelum training:
   - user memilih perilaku saat kombinasi sudah pernah training (`ask/retrain/skip`).
4. Memperbarui dokumentasi agar sinkron:
   - `documentation/architecture.md`
   - `documentation/arsitectur.md`
   - `documentation/dokumentasi_aplikasi.md`
   - `documentation/pipeline.md`
   - `documentation/requirements.md`

### Keputusan
1. Default perilaku existing run adalah `ask` agar aman dan tidak menimpa analisis tanpa konfirmasi user.
2. Training ulang selalu membuat run/model baru karena `run_id` berbasis timestamp.

### Next step
1. Jika dibutuhkan, tambahkan opsi konfirmasi global sekali per batch (bukan per kombinasi) pada mode `ask`.
2. Tambahkan badge/status "skipped_existing" di UI report jika ingin terlihat langsung di dashboard.

## 2026-05-16  (Asia/Jakarta) - Sesi 11

### Tujuan sesi
Memastikan split dataset tervalidasi seimbang sebelum training, dan menambahkan konfirmasi jika folder split sudah ada.

### Aktivitas yang sudah dilakukan
1. Menambahkan validasi balance split di `src/datasets/splitter.py`:
   - fungsi `validate_balanced_split_manifest(...)`,
   - metadata `balance_validation` ditulis ke `split_manifest.json`,
   - split akan gagal jika hasil tidak seimbang.
2. Menambahkan kontrol split existing di `training/train.py`:
   - argumen baru `--on-existing-split {ask,resplit,skip}`,
   - jika `--split-first` aktif dan split folder sudah ada, user bisa pilih split ulang atau pakai split lama,
   - split existing wajib valid (struktur split lengkap + manifest tersedia),
   - validasi balance dijalankan sebelum training.
3. Menambahkan kontrol split existing di `training/2.split_data_testing.py`:
   - argumen `--on-existing-split {ask,resplit,skip}`,
   - perilaku konsisten dengan orchestrator training,
   - validasi balance tetap dijalankan saat memakai split existing.
4. Menyesuaikan `main.py`:
   - prompt baru untuk perilaku split existing (`ask/resplit/skip`),
   - diteruskan ke pipeline split standalone dan training.
5. Memperbarui dokumentasi:
   - `documentation/architecture.md`
   - `documentation/arsitectur.md`
   - `documentation/dokumentasi_aplikasi.md`
   - `documentation/pipeline.md`
   - `documentation/requirements.md`

### Keputusan
1. Default split existing dibuat `ask` untuk mencegah split lama tertimpa tanpa konfirmasi.
2. Training tidak boleh lanjut jika split existing tidak valid atau tidak terverifikasi seimbang.

### Next step
1. Jika diperlukan, tambahkan mode konfirmasi global sekali per dataset untuk split existing pada batch besar.
2. Tambahkan tampilan status `balance_validation` di dashboard agar audit split lebih mudah.
