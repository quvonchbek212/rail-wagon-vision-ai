#!/usr/bin/env bash
# VagonDetect -- ishga tushirish (birinchi marta bo'lsa, avval o'zi o'rnatadi).
#   ./run.sh video.mp4                -- video faylni real vaqtda tahlil qilish
#   ./run.sh "rtsp://admin:parol@192.168.1.64:554/Streaming/Channels/101"   -- IP-kamera, 24/7
#   ./run.sh 0                        -- USB kamera
# Qo'shimcha: --oynasiz (oynasiz)   --natijalar <papka>   --tez (faylni kutmasdan, tez)
DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$DIR"
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import torch, cv2, ultralytics' 2>/dev/null; then
  ./install.sh || exit 1
fi
exec .venv/bin/python -u logic_N_T_V.py "$@"
