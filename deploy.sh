#!/usr/bin/env bash
# deploy.sh — update kode + (re)start dashboard Streamlit di WSL Ubuntu.
#
# Tujuan: "git pull lalu jalankan ulang aplikasi" dengan andal, sehingga setelah
# kode diperbarui Streamlit otomatis hidup kembali di background (tidak mati saat
# terminal/SSH ditutup).
#
# Pemakaian (jalankan dari folder project):
#   ./deploy.sh            # git pull + restart Streamlit (default)
#   ./deploy.sh restart    # restart Streamlit TANPA git pull
#   ./deploy.sh stop       # hentikan Streamlit
#   ./deploy.sh status     # cek status Streamlit
#   ./deploy.sh logs       # ikuti log Streamlit (tail -f)
#
# Variabel opsional:
#   PORT=8502 ./deploy.sh           # ganti port (default 8501)
#   ADDRESS=127.0.0.1 ./deploy.sh   # batasi akses (default 0.0.0.0)
#   STOP_TRAINING=1 ./deploy.sh     # ikut hentikan proses training train.py
#
# Catatan penting:
# - Training (train.py) berjalan sebagai proses TERPISAH (detached). Skrip ini
#   secara default TIDAK menghentikannya — hanya me-restart Streamlit. Gunakan
#   STOP_TRAINING=1 bila memang ingin menghentikan training juga.
# - git pull TIDAK memengaruhi training yang sudah berjalan (kode sudah dimuat ke
#   memori). Perubahan baru hanya berlaku untuk training yang diluncurkan setelahnya.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-0.0.0.0}"
RUN_DIR="$PROJECT_DIR/.run"
PID_FILE="$RUN_DIR/streamlit.pid"
LOG_FILE="$RUN_DIR/streamlit.log"
mkdir -p "$RUN_DIR"

# Pilih cara memanggil Streamlit; utamakan virtualenv (.venv). Pakai array agar
# aman terhadap path yang mengandung spasi (mis. /mnt/c/MY DATA/...).
if [ -x "$PROJECT_DIR/.venv/bin/streamlit" ]; then
  STREAMLIT_CMD=("$PROJECT_DIR/.venv/bin/streamlit")
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
  STREAMLIT_CMD=("$PROJECT_DIR/.venv/bin/python" -m streamlit)
elif command -v streamlit >/dev/null 2>&1; then
  STREAMLIT_CMD=(streamlit)
else
  echo "[deploy] ERROR: streamlit tidak ditemukan. Aktifkan venv / install dulu." >&2
  exit 1
fi

# Cetak PID Streamlit aktif (stdout) lalu return 0; return 1 bila tidak ada.
is_running() {
  if [ -f "$PID_FILE" ]; then
    local pid; pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      echo "$pid"; return 0
    fi
  fi
  # Fallback: cari proses streamlit untuk web/app.py.
  local pid; pid="$(pgrep -f 'streamlit run .*web/app.py' 2>/dev/null | head -n1 || true)"
  if [ -n "${pid:-}" ]; then echo "$pid"; return 0; fi
  return 1
}

stop_streamlit() {
  local pid
  if pid="$(is_running)"; then
    echo "[deploy] Menghentikan Streamlit (PID $pid)…"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
  else
    echo "[deploy] Streamlit tidak sedang berjalan."
  fi
  rm -f "$PID_FILE"
}

stop_training_if_requested() {
  if [ "${STOP_TRAINING:-0}" = "1" ]; then
    echo "[deploy] STOP_TRAINING=1 → menghentikan proses training (train.py)…"
    pkill -f 'training/train.py' 2>/dev/null || true
  fi
}

start_streamlit() {
  echo "[deploy] Menjalankan Streamlit di background (port $PORT, address $ADDRESS)…"
  # nohup + stdin /dev/null + & → tetap hidup walau terminal/SSH ditutup, dan
  # train.py yang diluncurkan dari web tidak mewarisi tty (tidak akan menggantung).
  nohup "${STREAMLIT_CMD[@]}" run web/app.py \
    --server.port "$PORT" \
    --server.address "$ADDRESS" \
    --server.headless true \
    >> "$LOG_FILE" 2>&1 < /dev/null &
  local pid=$!
  disown 2>/dev/null || true
  echo "$pid" > "$PID_FILE"
  sleep 2
  local active
  if active="$(is_running)"; then
    echo "[deploy] OK. Streamlit berjalan (PID $active)."
    echo "[deploy] URL : http://localhost:$PORT"
    echo "[deploy] Log : $LOG_FILE   (pantau: ./deploy.sh logs)"
  else
    echo "[deploy] GAGAL start. 30 baris terakhir log:" >&2
    tail -n 30 "$LOG_FILE" 2>/dev/null || true
    exit 1
  fi
}

git_pull() {
  echo "[deploy] git pull --ff-only …"
  if ! git pull --ff-only; then
    echo "[deploy] ERROR: git pull gagal (ada perubahan lokal/konflik?)." >&2
    echo "[deploy] Periksa 'git status'. Aplikasi TIDAK di-restart." >&2
    exit 1
  fi
}

cmd="${1:-deploy}"
case "$cmd" in
  deploy)
    git_pull
    stop_streamlit
    stop_training_if_requested
    start_streamlit
    ;;
  restart)
    stop_streamlit
    stop_training_if_requested
    start_streamlit
    ;;
  stop)
    stop_streamlit
    stop_training_if_requested
    ;;
  status)
    if pid="$(is_running)"; then
      echo "[deploy] Streamlit BERJALAN (PID $pid) → http://localhost:$PORT"
    else
      echo "[deploy] Streamlit MATI."
    fi
    ;;
  logs)
    echo "[deploy] Mengikuti log: $LOG_FILE (Ctrl-C untuk berhenti)"
    touch "$LOG_FILE"
    tail -n 100 -f "$LOG_FILE"
    ;;
  *)
    echo "Pemakaian: $0 {deploy|restart|stop|status|logs}" >&2
    exit 1
    ;;
esac
