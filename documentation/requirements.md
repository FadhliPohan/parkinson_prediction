## Catatan Status Implementasi (Update: 2026-05-16)
- Dokumen ini adalah requirement awal proyek (historical requirement).
- Status implementasi terkini dapat dilihat di:
  - `documentation/architecture.md`
  - `documentation/arsitectur.md`
  - `documentation/dokumentasi_aplikasi.md`
- Fitur terbaru yang sudah aktif di pipeline:
  - Registry dataset/model/method/augmentasi dinamis.
  - Split dataset otomatis dengan balancing kelas sebelum split.
  - Balancing kelas menggunakan augmentasi rotasi kecil `-20` s/d `+20` derajat untuk kelas minoritas.
  - Eksekusi kombinasi eksperimen terpisah dengan urutan `dataset -> augmentasi -> method -> model`.
  - Setiap kombinasi memiliki `experiment_id` unik.
  - Validasi run existing: sistem bisa `ask/retrain/skip` saat kombinasi sudah pernah ditraining.
  - Report dan artifact training berbasis struktur `report/<dataset>/<augmentasi>/<method>/<model>/<run_id>`.
  - Dashboard report bertingkat (Explorer + Perbandingan) berbasis indeks manifest.

Kamu adalah Senior Software Architect, Machine Learning Engineer, dan Web Application Engineer yang ahli dalam Python, TensorFlow/Keras, PyTorch, Ultralytics YOLO, Streamlit, struktur project ML, pipeline training, reporting, dan dokumentasi teknis.

Saya sedang membangun aplikasi web/pipeline machine learning untuk klasifikasi gambar, saat ini digunakan untuk klasifikasi Parkinson vs Healthy berdasarkan gambar hand-drawing.

Saya sudah memiliki project dengan struktur awal seperti berikut:
- dataset/original
- dataset/split
- model/
- report/
- trained_models/
- web/app.py
- main.py
- documentation/dokumentasi_aplikasi.md

Aplikasi saat ini memiliki alur:
1. Cek distribusi dataset.
2. Split dataset menjadi train, validation, dan testing.
3. Augmentasi data training secara on-the-fly.
4. Training beberapa model CNN, Transformer, dan YOLOv8.
5. Menyimpan hasil training ke folder report.
6. Menyimpan model hasil training ke folder trained_models.
7. Streamlit membaca report dan model untuk menampilkan dashboard evaluasi dan prediksi.

Tujuan utama saya adalah membuat aplikasi ini menjadi lebih dinamis, rapi, mudah dikembangkan, dan tidak terlalu bergantung pada hard-code.

Kebutuhan utama:
1. Dataset harus bersifat dinamis.
   - User/developer bisa memilih dataset mana yang akan digunakan untuk training.
   - Struktur dataset harus bisa dikelola dengan lebih rapi.
   - Sistem harus tetap mendukung dataset gambar klasifikasi dengan folder kelas.
   - Dataset original, split, dan hasil preprocessing harus mudah dilacak.

2. Model harus bersifat dinamis.
   - Model yang tersedia tidak boleh terlalu hard-code di banyak tempat.
   - Buat pendekatan registry/config agar model baru bisa ditambahkan dengan mudah.
   - Jika saya menambahkan model baru, model tersebut bisa muncul dalam pilihan training tanpa harus mengubah terlalu banyak file.
   - Model TensorFlow/Keras, PyTorch, dan YOLO boleh tetap didukung, tetapi struktur pemanggilannya harus dibuat lebih konsisten.

3. Metode/teknik training harus bersifat dinamis.
   - User/developer bisa memilih metode training yang digunakan.
   - Contoh metode: baseline training, transfer learning, fine-tuning, augmentation on-the-fly, mixed precision, atau metode lain yang sesuai.
   - Parameter seperti epochs, batch size, image size, learning rate, fine-tune epochs, optimizer, dan mixed precision harus lebih mudah diatur melalui CLI/config.

4. Training tetap dijalankan melalui terminal.
   - Saya ingin tetap bisa menjalankan aplikasi dan memilih training melalui terminal.
   - Boleh memperbaiki `main.py` agar menu lebih rapi, dinamis, dan tidak hard-code berlebihan.
   - Boleh menambahkan CLI argument/config file jika memang membuat sistem lebih baik.

5. Report harus menjadi sumber utama untuk dashboard Streamlit.
   - Setiap training harus menghasilkan report yang konsisten.
   - Streamlit harus membaca folder report dan menampilkan hasil training secara detail.
   - Report harus mencakup metrik, history training, classification report, confusion matrix, grafik evaluasi, metadata dataset, metadata model, parameter training, waktu training, dan informasi run.
   - Struktur report harus dibuat standar agar dashboard mudah membaca semua model dan semua run.

6. Dashboard Streamlit harus tetap fokus pada visualisasi hasil.
   - Streamlit membaca report dan trained model.
   - Streamlit menampilkan ringkasan dataset, daftar eksperimen/run, metrik performa, grafik training, confusion matrix, classification report, dan prediksi gambar.
   - Jangan membuat Streamlit terlalu bergantung pada logic training.
   - Logic training sebaiknya tetap di pipeline/backend, bukan di UI.

7. Struktur project harus diperbaiki agar lebih rapi dan scalable.
   - Analisis seluruh struktur file dan folder yang ada.
   - Identifikasi bagian yang masih hard-code, duplikatif, sulit dikembangkan, atau sulit dipelihara.
   - Usulkan struktur folder baru yang lebih baik.
   - Terapkan perubahan secara bertahap dan hati-hati.
   - Pastikan perubahan tidak merusak alur lama tanpa alasan jelas.

8. Dokumentasi wajib diperbarui.
   - Setelah analisis selesai, tulis hasil analisis ke file:
     `documentation/arsitectur.md`
   - Jika memungkinkan, gunakan nama yang lebih benar:
     `documentation/architecture.md`
     Namun jika project sudah memakai `arsitectur.md`, tetap buat/update file tersebut agar sesuai permintaan.
   - Dokumentasi harus menjelaskan:
     - masalah struktur lama,
     - tujuan perbaikan,
     - arsitektur baru,
     - struktur folder baru,
     - alur dataset,
     - alur training,
     - alur model registry,
     - alur method/training strategy,
     - alur report,
     - alur dashboard Streamlit,
     - format artifact training,
     - cara menambahkan model baru,
     - cara menambahkan metode training baru,
     - cara menjalankan training dari terminal,
     - cara menjalankan dashboard.

9. Sebelum melakukan perubahan kode apa pun, kamu wajib melakukan tahap analisis dan mengajukan pertanyaan klarifikasi terlebih dahulu.
   - Jangan langsung mengubah file.
   - Baca dan pahami struktur project.
   - Baca dokumentasi yang sudah ada.
   - Identifikasi kebutuhan yang belum jelas.
   - Ajukan pertanyaan yang benar-benar penting saja.
   - Setelah saya menjawab, baru lanjutkan perubahan.

10. Saat melakukan perubahan, ikuti prinsip berikut:
   - Jangan menghapus fitur yang sudah ada tanpa alasan kuat.
   - Jangan mengubah nama file/folder penting tanpa memperbarui semua referensinya.
   - Jangan membuat logic yang sama berulang di banyak file.
   - Gunakan pendekatan modular.
   - Pisahkan logic dataset, model, training, reporting, dan UI.
   - Buat struktur yang mudah dibaca oleh developer lain.
   - Pastikan semua perubahan terdokumentasi.
   - Berikan catatan migrasi jika ada struktur lama yang berubah.

Tahapan kerja yang harus kamu lakukan:

Tahap 1 — Analisis awal:
- Baca seluruh struktur project.
- Baca dokumentasi aplikasi.
- Pahami alur training saat ini.
- Pahami cara model dipanggil.
- Pahami cara report dibuat.
- Pahami cara Streamlit membaca report/model.
- Temukan bagian yang hard-code, duplikatif, atau sulit dikembangkan.

Tahap 2 — Pertanyaan klarifikasi:
Sebelum mengubah file, tanyakan hal-hal penting seperti:
- Apakah target klasifikasi tetap Parkinson vs Healthy atau harus mendukung multi-class?
- Apakah format dataset tetap folder-per-class seperti ImageFolder?
- Apakah user memilih dataset/model/metode lewat terminal menu, CLI argument, config file, atau kombinasi?
- Apakah report lama harus tetap kompatibel dengan dashboard baru?
- Apakah model TensorFlow, PyTorch, dan YOLO tetap harus didukung dalam satu sistem registry?
- Apakah training semua model sekaligus masih dibutuhkan?
- Apakah Streamlit hanya membaca report, atau boleh juga memicu training?
- Apakah nama file dokumentasi harus tetap `arsitectur.md` atau boleh diperbaiki menjadi `architecture.md`?

Tahap 3 — Rancangan arsitektur:
Buat rancangan arsitektur baru sebelum coding, meliputi:
- struktur folder baru,
- dataset registry/config,
- model registry,
- training method/strategy registry,
- report schema,
- artifact schema,
- CLI/menu flow,
- integrasi Streamlit.

Tahap 4 — Implementasi:
Setelah rancangan disetujui atau setelah pertanyaan dijawab:
- Refactor struktur file secara hati-hati.
- Buat registry/config untuk model.
- Buat registry/config untuk dataset.
- Buat registry/config untuk metode training.
- Rapikan pipeline training.
- Standarkan format report.
- Update Streamlit agar membaca report secara dinamis.
- Update menu terminal agar pilihan dataset, metode, dan model bisa dinamis.

Tahap 5 — Dokumentasi:
Update seluruh dokumentasi yang relevan:
- `documentation/arsitectur.md`
- `documentation/dokumentasi_aplikasi.md`
- README jika tersedia
- dokumentasi cara menjalankan training
- dokumentasi cara menambahkan model baru
- dokumentasi format report

Tahap 6 — Validasi:
Lakukan pengecekan akhir:
- Pastikan training tetap bisa dipilih via terminal.
- Pastikan minimal satu model bisa dijalankan.
- Pastikan report dibuat sesuai schema.
- Pastikan Streamlit bisa membaca report.
- Pastikan dokumentasi sesuai dengan struktur terbaru.
- Berikan ringkasan file apa saja yang diubah dan alasan perubahannya.

Output yang saya harapkan dari kamu:
1. Daftar hasil analisis struktur lama.
2. Daftar pertanyaan klarifikasi sebelum perubahan.
3. Rancangan arsitektur baru.
4. Implementasi perubahan setelah saya menjawab pertanyaan.
5. Update dokumentasi ke `documentation/arsitectur.md`.
6. Ringkasan perubahan.
7. Instruksi menjalankan aplikasi melalui terminal.
8. Instruksi menjalankan dashboard Streamlit.
9. Instruksi menambahkan dataset baru.
10. Instruksi menambahkan model baru.
11. Instruksi menambahkan metode training baru.

Ingat:
- Jangan langsung coding sebelum bertanya.
- Jangan membuat perubahan besar tanpa menjelaskan alasannya.
- Jangan menghilangkan kemampuan training via terminal.
- Fokus pada aplikasi yang dinamis, modular, rapi, dan mudah dikembangkan.
- histori pengerjaan bisa dibuat kedalam [history.md](documentation/history.md) , sehingga bisa token saya habis maka bisa melanjutkan pekerjaan dengan membaca historionya

MODE KERJA WAJIB:
Untuk respons pertama, kamu hanya boleh melakukan analisis dan bertanya.
Kamu dilarang mengubah file apa pun pada respons pertama.
Setelah saya menjawab pertanyaan klarifikasi, baru kamu boleh membuat rencana implementasi dan melakukan perubahan.

Prioritas utama bukan menambah fitur sebanyak-banyaknya, tetapi membuat struktur aplikasi lebih dinamis, modular, konsisten, dan mudah dikembangkan.

Jangan fokus hanya membuat model baru. Fokus utama adalah memperbaiki fondasi:
- dataset registry,
- model registry,
- training method registry,
- report schema,
- terminal training flow,
- Streamlit report reader,
- dokumentasi arsitektur.

bentuk saran arcitectur
parkinson_prediction/
├─ configs/
│  ├─ datasets.yaml
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
│  │
│  ├─ models/
│  │  ├─ registry.py
│  │  ├─ tensorflow_models/
│  │  ├─ pytorch_models/
│  │  └─ yolo_models/
│  │
│  ├─ training/
│  │  ├─ trainer.py
│  │  ├─ strategies.py
│  │  ├─ tensorflow_trainer.py
│  │  ├─ pytorch_trainer.py
│  │  └─ yolo_trainer.py
│  │
│  ├─ reporting/
│  │  ├─ report_writer.py
│  │  ├─ report_reader.py
│  │  ├─ schemas.py
│  │  └─ plots.py
│  │
│  ├─ inference/
│  │  ├─ predictor.py
│  │  └─ model_loader.py
│  │
│  └─ utils/
│     ├─ paths.py
│     ├─ logging.py
│     └─ config.py
│
├─ model/
│  └─ legacy_or_wrappers/
│
├─ report/
│  └─ <dataset_name>/<model_name>/<run_id>/
│
├─ trained_models/
│  └─ <dataset_name>/<model_name>/<run_id>/
│
├─ web/
│  └─ app.py
│
├─ main.py
├─ train.py
├─ requirements.txt
└─ documentation/
   ├─ dokumentasi_aplikasi.md
   └─ arsitectur.md
