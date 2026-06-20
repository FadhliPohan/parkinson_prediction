# PROGRESS — Tracking Pekerjaan

> Update: 2026-06-04. Checklist seluruh tugas. `- [x]` selesai, `- [ ]` belum,
> `- [~]` sebagian/ditunda dengan catatan.

## T1 — Terapkan rekomendasi dokumen
- [x] Naikkan default `epochs` (8 → 25) di `configs/default_training.yaml` (REC-01)
- [x] Naikkan `early_stopping_patience` (4 → 8) (REC-02)
- [x] Naikkan `fine_tune_learning_rate` (1e-5 → 5e-5) (REC-05)
- [x] Tambah override epoch/patience per method di `configs/training_methods.yaml`
- [x] Selaraskan monitoring callback ke `val_accuracy` (REC-03)
- [x] Buat callback baru per stage (state tidak terbawa) (REC-04)
- [x] Terapkan `class_weight` 'balanced' pada `model.fit` (REC-10)
- [x] Output layer `dtype="float32"` untuk mixed precision (REC-11)
- [x] YOLO: cosine LR + warmup + label smoothing (REC-08)
- [~] REC-06 (cosine scheduler TF) — ditunda (low impact, butuh refactor)
- [~] REC-09 (Transformer pretrained/resep khusus) — DITUNDA sesuai batasan (TODO terdokumentasi)

## T1b — Audit pra-deploy (2026-06-04)
- [x] **BLOCKER-01** — Hapus double preprocessing di inference (`src/inference/predictor.py`,
      `prepare_tf_input`). Model TF sudah memuat layer preprocessing; inference kini
      memberi piksel mentah [0,255]. Verifikasi: prob train == prob inference utk gambar sama.
- [x] **BLOCKER-02** — Splitter ditulis ulang: split dulu, balancing train-only, test/val real-only
      (`src/datasets/splitter.py`, schema manifest 1.3.0). Validator jadi guard anti-leakage. Lulus uji dummy.
      ⏳ Sisa: hapus split lama + re-split + latih ulang semua model (split lama tercemar).
- [x] **BLOCKER-03** — Dikonfirmasi user (2026-06-04): tiap gambar `dataset_merder` independen → split
      per-gambar valid, tidak perlu split per-pasien. Sumber split = `dataset_merder` (BUKAN `praprosesing` 4×).
- [ ] **DATA-01/02** (belum dikerjakan, atas permintaan user): samakan `datasets.yaml` dgn folder nyata
      + nonaktifkan `parkinson_mixing`; hapus split lama yang bocor di `dataset/split/{train,testing,validation}`.
- [ ] **DEP-01** — Uji `pip install` di server bersih + pin `torch` di `requirements.txt`.
- [ ] **ISU-07** — Transformer dari nol; opsi cepat: `enabled: false` vit/swin/deit di `configs/models.yaml`.

## T1c — Tab Dokumentasi di dashboard (2026-06-04)
- [x] Tambah `render_documentation_tab()` + tab "Dokumentasi" di `web/app.py` (9 sub-bagian:
      cara menjalankan, alur sistem, split, preprocessing, augmentasi, model, training, evaluasi, batasan).
      Konten daftar model/method/optimizer/preset diambil dinamis dari registry. Lulus smoke test render.

## T2 — Folder optimizer (`src/optimizer/`)
- [x] `src/optimizer/adam.py` (wrapper Adam)
- [x] `src/optimizer/no_optimize.py` (SGD plain baseline)
- [x] `src/optimizer/__init__.py` — `get_optimizer(name, learning_rate, **cfg)`, `list_optimizers()`
- [x] Wiring ke `build_model` + recompile stage 2 (`training_common.py`)
- [x] Argumen `--optimizer` di worker TF (`build_common_arg_parser`)
- [x] Argumen `--optimizer` di YOLO + pemetaan Adam/SGD
- [x] `optimizer: adam` di `configs/default_training.yaml`
- [x] Argumen `--optimizer` + override di `training/train.py`

## T3 — Laporan Streamlit lengkap
- [x] Tambah metrik ROC-AUC, Precision (macro), Recall (macro) pada detail run
- [x] Kurva training (loss/accuracy) — sudah ada, dipertahankan
- [x] Confusion matrix + classification report (CSV) — sudah ada, dipertahankan
- [x] Ringkasan konfigurasi run (epoch/batch/fine-tune) — sudah ada

## T4 — Perbandingan antar-model (overlay)
- [x] Sub-tab "Overlay Model" (pilih ≥2 run)
- [x] Tabel metrik berdampingan
- [x] Grafik overlay kurva training (pilih metrik)

## T5 — Download report berfilter
- [x] Filter dataset/method/augmentasi/**model spesifik** + rentang tanggal
- [x] Download CSV
- [x] Download HTML printable (A4 → Save as PDF) tanpa dependency baru

## T6 — Update dokumentasi
- [x] `architecture.md` (folder `src/optimizer`, optimizer, split dinamis, penyempurnaan)
- [x] `pipeline.md` (tab dashboard baru)
- [x] `documentation/fitur_baru.md` (dokumen komprehensif fitur baru)
- [x] `documentation/analisis_training_rekomendasi.md` (sumber rekomendasi, sudah ada)

## T7 — Training via web (Streamlit)
- [x] Tab "Training" baru
- [x] Pilihan model, optimizer, method, augmentasi, dataset
- [x] Hyperparameter: epoch, batch size, learning rate, fine-tune epoch, seed
- [x] Pilihan split preset + `--split-first`
- [x] Eksekusi `training/train.py` via subprocess + log real-time
- [x] Tampilkan ringkasan hasil setelah selesai

## T8 — Dynamic data split
- [x] Preset `80-10-10` dan `70-15-15` (`SPLIT_PRESETS`)
- [x] `validate_split_ratios` (total=100%, testing==validation)
- [x] Override rasio manual (`--train-ratio/--test-ratio/--val-ratio`)
- [x] Seed dikunci (default 42)
- [x] Terintegrasi di `2.split_data_testing.py`, `train.py`, dan tab Training web

## T9 — Rapikan struktur file & folder
- [~] Pendekatan **aditif & aman** (keputusan user): nama folder lama dipertahankan
      (`dataset/`, `web/`, `report/`, `documentation/`) agar tidak ada path yang patah.
      Hanya menambah `src/optimizer/`. Reorg rename penuh tidak dilakukan.

## T10 — File tracking
- [x] `documentation/PROGRESS.md` (file ini)

## T11 — Tambah Model VGG16 & ResNeXt50 (2026-06-04)
- [x] `model/legacy_or_wrappers/vgg16.py` — wrapper VGG16 (tf.keras.applications, pretrained ImageNet)
- [x] `model/legacy_or_wrappers/resnext_backbones.py` — implementasi ResNeXt-50-32x4d dari scratch
      (grouped conv, cardinality=32, base_width=4, ~23M params)
- [x] `model/legacy_or_wrappers/resnext50.py` — wrapper ResNeXt50
- [x] `configs/models.yaml` — tambah entry `vgg16` dan `resnext50` (family cnn)
- [x] `src/inference/predictor.py` — tambah `"vgg16"` dan `"resnext50"` di `MODEL_PREPROCESSORS`
- [x] Dokumentasi diperbarui:
      `architecture.md` (section 6 + section 19 baru),
      `pipeline.md` (daftar model aktif),
      `analisis_training_rekomendasi.md` (catatan ISU + REC-05 tabel LR),
      `PROGRESS.md` (file ini)
- [x] Verifikasi build: VGG16 output (None,7,7,512) ✓ | ResNeXt50 output (None,7,7,2048) ✓
- [x] Verifikasi registry: kedua model terbaca oleh `ModelRegistry` ✓

## T7 — Preset split multi + visibilitas split + optimizer terminal + Streamlit non-blocking (2026-06-09)
- [x] `splitter.py`: helper `normalize_preset`, `preset_label_for_ratios`, `split_dir_for_preset`,
      `resolve_preset_list`; `split_dataset(split_preset=...)` + `split_manifest.json` simpan `split_preset` (schema 1.4.0)
- [x] `training/2.split_data_testing.py`: arg `--split-presets` (both/config/list) → folder split terpisah per preset
- [x] `training/train.py`: arg `--split-presets`, loop preset (folder via `dataclasses.replace`), cetak preset aktif per `[RUN]`,
      `split_preset` masuk experiment_id + deteksi run existing + summary CSV
- [x] `training_common.py`: baca `split_manifest.json` → cetak "Split Dataset Dipakai" + simpan `dataset.split_preset`/`split_ratio` di `run_manifest.json`
- [x] `report_reader.py`: surface `split_preset` + `split_ratio`; key latest-summary menyertakan preset
- [x] `main.py`: menu Split (no.3) & semua menu training (5/6/7/8/10/11/12) menanyakan **preset split** (incl. `both`) + **optimizer** (adam/no_optimize)
- [x] `src/training/recommendations.py` (baru): default & clue hyperparameter per model/method
- [x] `web/app.py`: training **non-blocking** (subprocess + file log + polling ~2s, tombol Hentikan/reset),
      selectbox preset split (incl. `both`), auto-fill default + peringatan per model/method, kolom `split_preset` di report (ringkasan/ranking/download/overlay), preset selector di tab Dataset
- [x] Dokumentasi: `fitur_baru.md` (section 6), `PROGRESS.md` (file ini)

## T8 — Tab Training Streamlit multi-select + fan-out optimizer (2026-06-09)
- [x] `train.py`: arg `--optimizers` (all/list) + loop optimizer (dalam loop method); optimizer masuk
      `experiment_id`, `_find_existing_runs`, estimasi kombinasi, result/summary CSV, dan baris `[RUN]`
- [x] `web/app.py`: tab Training — **dataset, model, method, optimizer, augmentasi, preset split** semua
      multi-select; `_build_training_command` kirim CSV (`--dataset`/`--models`/`--method`/`--augmentations`/`--optimizers`/`--split-presets`);
      tampil estimasi total kombinasi; opsi augmentasi = gabungan dataset terpilih
- [x] Dokumentasi: `fitur_baru.md` (section 6.6), `PROGRESS.md`
- [x] Verifikasi: `py_compile` OK; parse argv multi gaya web OK; `_resolve_optimizer_list` (all/list/single/none) OK

## T9 — Fix training web menggantung di prompt "retrain" + konfirmasi via UI (2026-06-20)
- [x] **Akar masalah**: subprocess `train.py` dari web mewarisi tty milik `streamlit run`,
      jadi `sys.stdin.isatty()`=True → train.py memanggil `input("Lanjutkan training ulang? [y/N]")`
      yang menggantung selamanya (tidak ada terminal untuk mengetik). `--on-existing retrain`
      yang sudah dikirim web pun tetap terblokir karena guard lama hanya cek `isatty`.
- [x] `web/app.py` `_start_training_process`: tambah `stdin=subprocess.DEVNULL` →
      subprocess tidak mewarisi tty; train.py terdeteksi non-interaktif dengan benar.
- [x] `training/train.py`: flag baru `--non-interactive` + helper `_session_is_interactive()`
      (`NON_INTERACTIVE` global). Semua cek `not sys.stdin.isatty()` diganti `not _session_is_interactive()`.
      `input()` di `_ask_resplit_for_dataset` & `_ask_global_retrain_confirmation` dibungkus
      `try/except EOFError` agar tak pernah menggantung.
- [x] `web/app.py`: konfirmasi **via UI sebelum training** (menggantikan ketikan y/N terminal).
      Helper `_count_existing_training_combos()` + `_web_preset_to_record_label()` mendeteksi
      kombinasi yang sudah pernah ditraining; bila ada, tampil radio **retrain** vs **skip**.
      `_build_training_command` kirim `--non-interactive` + `--on-existing <pilihan user>`.
- [x] Verifikasi: `py_compile` train.py & app.py OK; `train.py --help` memunculkan `--non-interactive`.

## T10 — Hapus run dari tabel "Ringkasan Run Terbaru per Kombinasi" (2026-06-20)
- [x] **Tujuan**: bersihkan model yang kurang baik & hemat storage langsung dari halaman Report.
- [x] `web/app.py`: refactor `_render_filterable_dataframe` → ekstrak `_apply_table_filters`
      (filter dapat dipakai ulang oleh tabel berbasis `st.data_editor`).
- [x] Helper baru di `web/app.py`:
      `_resolve_model_dir()` (mirror `run_dir` dari `REPORT_ROOT` → `TRAINED_MODELS_ROOT`),
      `_cleanup_empty_parents()` (hapus folder kombinasi kosong),
      `_refresh_latest_run_marker()` (perbarui/hapus `latest_run.txt` bila menunjuk run terhapus),
      `_delete_run_artifacts()` (hapus folder report + model satu run, kembalikan status).
- [x] `_render_latest_summary_with_delete()`: tabel `st.data_editor` dengan kolom checkbox
      **Hapus** + konfirmasi (checkbox "Saya mengerti" + tombol) → hapus permanen, lalu rerun
      dengan feedback hasil. Dipanggil dari `render_report_tab`.
- [x] Verifikasi: `py_compile web/app.py` OK; uji `_delete_run_artifacts` dgn struktur dummy →
      folder report+model run terhapus, `latest_run.txt` ter-update ke run tersisa, folder
      kombinasi kosong ikut dibersihkan.

## Verifikasi
- [x] `py_compile` semua file yang diubah → OK
- [x] Uji fungsi optimizer registry & resolusi/validasi split → OK
- [x] Uji build backbone VGG16 + ResNeXt50 → OK (verified 2026-06-04)
- [x] (2026-06-09) `py_compile` semua file diubah → OK; `train.py --help` memunculkan `--split-presets`/`--optimizer`;
      helper preset & modul rekomendasi/clue diuji manual → OK
- [ ] Uji training end-to-end nyata (butuh dataset + GPU/CPU; dijalankan user)
