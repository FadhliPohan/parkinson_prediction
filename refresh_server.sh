#!/usr/bin/env bash
# refresh_server.sh — dipanggil oleh tombol "Refresh Server" di dashboard.
#
# Urutan:
#   1) git pull (ambil kode terbaru)
#   2) kill SEMUA screen (hentikan training, bebaskan RAM)
#   3) restart Streamlit
#
# Skrip ini sengaja dijalankan TER-DETACH (sesi sendiri) oleh dashboard, sehingga
# tetap berjalan walau Streamlit (pemanggilnya) ikut dimatikan pada langkah 3.
# Tidak me-reboot OS — hanya merestart aplikasi.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

RUN_DIR="$PROJECT_DIR/.run"
mkdir -p "$RUN_DIR"
# Catat seluruh output refresh ke file agar bisa diperiksa bila gagal.
exec >> "$RUN_DIR/refresh.log" 2>&1

echo "==================================================================="
echo "[refresh] mulai refresh server (pid $$)"

# Beri jeda agar respons HTTP ke browser sempat terkirim sebelum Streamlit mati.
sleep 1

echo "[refresh] (1/3) git pull --ff-only"
if ! git pull --ff-only; then
  echo "[refresh] WARNING: git pull gagal (ada perubahan lokal/konflik). Lanjut restart pakai kode lama."
fi

echo "[refresh] (2/3) hentikan semua screen"
bash "$PROJECT_DIR/deploy.sh" kill-screens || true

echo "[refresh] (3/3) restart Streamlit"
if bash "$PROJECT_DIR/deploy.sh" restart; then
  echo "[refresh] selesai: Streamlit hidup kembali."
else
  echo "[refresh] ERROR: gagal me-restart Streamlit. Cek log di atas."
fi
