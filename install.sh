#!/usr/bin/env bash
# =====================================================================
#  VagonDetect -- bitta buyruq bilan o'rnatish (Linux: Ubuntu/Debian, x86_64 yoki ARM64)
#
#   ./install.sh                         -- barcha kerakli paketlarni o'rnatadi
#   ./install.sh --servis "<manba>"      -- + 24/7 ishlaydigan systemd servis qiladi
#        manba: "rtsp://admin:parol@192.168.1.64:554/Streaming/Channels/101" yoki 0 (USB kamera)
#
#  Nima qiladi:
#   1. Tizim paketlari yetishmasa (python3-venv, libGL) -- apt orqali o'rnatadi (sudo so'raydi)
#   2. Loyiha ichida .venv yaratadi (tizim Python'iga tegmaydi)
#   3. NVIDIA GPU va drayverning CUDA versiyasini aniqlab, mos PyTorch o'rnatadi
#      (GPU bo'lmasa -- CPU versiya)
#   4. requirements.txt dagi paketlarni o'rnatadi va hammasini tekshiradi
# =====================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$DIR"
say() { echo -e "\n==> $*"; }

# ---------------------------------------------------------- 0. modellar
for m in detection4.pt classificationvagon.pt detect-t-v.pt classmodel.pt; do
  if [ ! -f "$m" ]; then
    echo "XATO: '$m' topilmadi. To'rtta model fayli logic_N_T_V.py bilan bir papkada bo'lishi kerak."
    exit 1
  fi
done

# ---------------------------------------------------------- 1. tizim paketlari
need=()
python3 -c 'import venv, ensurepip' 2>/dev/null || need+=(python3-venv python3-pip)
ldconfig -p 2>/dev/null | grep -q 'libGL.so.1'       || need+=(libgl1)
ldconfig -p 2>/dev/null | grep -q 'libglib-2.0.so.0' || need+=(libglib2.0-0)
if ! command -v python3 >/dev/null; then need+=(python3); fi
if [ ${#need[@]} -gt 0 ]; then
  say "Tizim paketlari o'rnatilmoqda: ${need[*]}"
  if command -v apt-get >/dev/null; then
    sudo apt-get update && sudo apt-get install -y "${need[@]}"
  else
    echo "apt-get topilmadi -- quyidagilarni qo'lda o'rnating: ${need[*]}"; exit 1
  fi
fi
python3 -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9+ kerak"'

# ---------------------------------------------------------- 2. virtual muhit
if [ ! -x .venv/bin/python ]; then
  say "Virtual muhit yaratilmoqda (.venv)"
  python3 -m venv .venv
fi
PY="$DIR/.venv/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools -q

# ---------------------------------------------------------- 3. PyTorch (GPU ga mos)
TORCH_INDEX=cpu
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
  CUDA=${CUDA:-0}
  if   [ "$CUDA" -ge 126 ]; then TORCH_INDEX=cu126
  elif [ "$CUDA" -ge 124 ]; then TORCH_INDEX=cu124
  elif [ "$CUDA" -ge 121 ]; then TORCH_INDEX=cu121
  elif [ "$CUDA" -ge 118 ]; then TORCH_INDEX=cu118
  fi
  say "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), drayver CUDA ${CUDA:0:${#CUDA}-1}.${CUDA: -1} -> PyTorch '$TORCH_INDEX'"
else
  say "NVIDIA GPU topilmadi -> PyTorch CPU versiyasi (real vaqt uchun NVIDIA GPU tavsiya etiladi)"
fi
if ! "$PY" -c 'import torch' 2>/dev/null; then
  "$PY" -m pip install --index-url "https://download.pytorch.org/whl/$TORCH_INDEX" torch torchvision
else
  say "PyTorch allaqachon o'rnatilgan: $("$PY" -c 'import torch; print(torch.__version__)')"
fi

# ---------------------------------------------------------- 4. qolgan paketlar
say "Paketlar o'rnatilmoqda (requirements.txt)"
"$PY" -m pip install -r requirements.txt
if [ "$TORCH_INDEX" = "cpu" ] && [ "$(uname -m)" = "x86_64" ]; then
  # GPU yo'q Intel/AMD kompyuter: t/v detektori OpenVINO bilan 2-3 barobar tez ishlaydi (ixtiyoriy)
  say "GPU yo'q -- CPU tezlatkichi (OpenVINO) o'rnatilmoqda"
  "$PY" -m pip install openvino || echo "  OpenVINO o'rnatilmadi -- oddiy CPU rejimida ishlaydi"
fi

# Qt oynasi shrift ogohlantirishlari chiqmasin (tizim shriftlari ulanadi)
QTDIR=$("$PY" -c 'import cv2, os; print(os.path.join(os.path.dirname(cv2.__file__), "qt"))')
for fd in /usr/share/fonts/truetype/dejavu /usr/share/fonts/dejavu /usr/share/fonts/TTF; do
  if [ -d "$fd" ] && [ -d "$QTDIR" ] && [ ! -e "$QTDIR/fonts" ]; then ln -s "$fd" "$QTDIR/fonts"; break; fi
done

# ---------------------------------------------------------- 5. tekshiruv
say "Tekshiruv"
"$PY" - <<'EOF'
import torch, cv2, ultralytics, lap, numpy
gpu = torch.cuda.is_available()
print(f"  torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if gpu else 'YOQ (CPU)'}")
print(f"  opencv {cv2.__version__} | ultralytics {ultralytics.__version__} | numpy {numpy.__version__}")
if gpu:
    x = torch.randn(512, 512, device="cuda"); assert torch.isfinite(x @ x).all()
    print("  GPU hisoblash testi: OK")
EOF

# ---------------------------------------------------------- 6. 24/7 servis (ixtiyoriy)
if [ "${1:-}" = "--servis" ]; then
  SRC="${2:-}"
  if [ -z "$SRC" ]; then echo "Manba ko'rsatilmadi: ./install.sh --servis \"rtsp://...\""; exit 1; fi
  UNIT=/etc/systemd/system/vagon-detect.service
  say "systemd servis yaratilmoqda: $UNIT"
  sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=VagonDetect - vagon raqami, tonna va hajmni aniqlash (24/7)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(id -un)
WorkingDirectory=$DIR
ExecStart=$PY -u $DIR/logic_N_T_V.py "$SRC" --oynasiz
Restart=always
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now vagon-detect.service
  say "Servis ishga tushdi. Holat:   systemctl status vagon-detect"
  echo "    Jurnal (jonli): journalctl -u vagon-detect -f"
  echo "    To'xtatish:     sudo systemctl stop vagon-detect"
fi

say "Tayyor. Ishga tushirish:  ./run.sh <video yoki kamera manzili>"
