# 🚂 VagonDetect — Railway Wagon Number, Tonnage & Volume AI Recognition

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange.svg)](https://ultralytics.com)
[![OpenVINO](https://img.shields.io/badge/Intel-OpenVINO-0071C5.svg)](https://openvino.ai)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Temir yo'l vagonlarini real-vaqt rejimida (RTSP IP-kameralar yoki videolardan) avtomatlashtirilgan tarzda aniqlab, har bir vagonning **8 xonali raqami**, **yuk ko'tarish quvvati (tonna)** va **kuzov hajmini (m³)** sun'iy intellekt yordamida o'quvchi va logistika hisobotini shakllantiruvchi 24/7 sanoat AI tizimi.

---

## 📑 Mundarija
- [Asosiy Vazifasi](#-asosiy-vazifasi)
- [Tizim Arxitekturasi](#-tizim-arxitekturasi)
- [Ishlatiladigan AI Modellar](#-ishlatiladigan-ai-modellar)
- [O'rnatish (Bitta buyruq bilan)](#-ornatish-bitta-buyruq-bilan)
- [Ishga Tushirish](#-ishga-tushirish)
- [24/7 Tizimli Servis (Linux Daemon)](#-247-tizimli-servis-linux-daemon)
- [Bog'lanish](#-boglanish)

---

## 🎯 Asosiy Vazifasi

Temir yo'l orqali yuk tashish, logistika terminallari va yirik sanoat korxonalari (ko'mir, kimyo, metallurgiya) uchun mo'ljallangan:
* **8 xonali raqam:** Har bir o'tayotgan vagonning davlat/tarmoq ro'yxat raqami aniqlanadi;
* **Yuk ko'tarish (Tonna) & Hajm (m³):** Vagon korpusidagi texnik parametrlar o'qiladi;
* **24/7 Ishlash Rejimi:** Tizim kecha-yu kunduz, noqulay ob-havo sharoitida ham avtomatlashtirilgan holda poyezdlar harakatini kutadi va yozib boradi.

---

## 🏗 Tizim Arxitekturasi

```mermaid
graph TD
    A[RTSP IP Kamera / Video Oqimi] --> B[logic_N_T_V.py Asosiy Pipeline]
    B --> C[1. detection4.pt: Vagon raqami plastinkasini topish]
    C --> D[2. classificationvagon.pt: 8 xonali vagon raqamini o'qish]
    B --> E[3. detect-t-v.pt: Tonna va hajm plastinkasini topish]
    E --> F[4. classmodel.pt: Tonna va hajm parametrlarini o'qish]
    D --> G[Hisobot generatori & Ma'lumotlarni tekshirish]
    F --> G
    G --> H[(24/7 Logistika Hisoboti & Tasvirlar)]
```

---

## 📦 Ishlatiladigan AI Modellar

Tizim ko'p bosqichli konveyerda 4 ta maxsus o'qitilgan modellardan foydalanadi:

| Model Fayli | Vazifasi | Hajmi |
|---|---|---|
| `detection4.pt` | Vagon raqami yozilgan plastinkani aniqlash | ~6.2 MB |
| `classificationvagon.pt` | Plastinkadagi 8 xonali raqamlarni o'qish (OCR) | ~6.2 MB |
| `detect-t-v.pt` | Tonna va hajm yozilgan texnik hududni aniqlash | ~20.4 MB |
| `classmodel.pt` | Tonna va kubometr belgilarini o'qish | ~20.3 MB |
| `detect-t-v_openvino_model/` | Intel protsessorlari uchun OpenVINO tezlashtirilgan model | ~38 MB |

---

## 💻 O'rnatish (Bitta buyruq bilan)

### Linux (Ubuntu/Debian, x86_64 yoki ARM64):
```bash
git clone https://github.com/quvonchbek212/vagon-detect-ai.git
cd vagon-detect-ai
chmod +x install.sh run.sh
./install.sh
```

### Windows 10/11:
1. [Python 3.9+](https://www.python.org/downloads/) o'rnating ("Add to PATH" belgilansin).
2. `install.bat` faylini ikki marta bosing (u avtomatik virtual muhit yaratadi, kerakli paketlarni va GPU drayverlarini tekshirib o'rnatadi).

---

## 🚀 Ishga Tushirish

| Vazifa | Linux | Windows |
|---|---|---|
| **IP-kamera (RTSP 24/7)** | `./run.sh "rtsp://admin:parol@192.168.1.64:554/Streaming/Channels/101"` | `run.bat "rtsp://..."` |
| **USB kamera** | `./run.sh 0` | `run.bat 0` |
| **Video fayl (Real vaqtda)** | `./run.sh train_video.MOV` | `run.bat train_video.MOV` |

### Qo'shimcha parametrlar:
* `--oynasiz` — Grafiksiz (headless) fonda ishlash;
* `--natijalar <papka>` — Natijalarni ko'rsatilgan papkaga yozish;
* `--tez` — Video faylni kutmasdan eng yuqori tezlikda tahlil qilish.

---

## ⚙️ 24/7 Tizimli Servis (Linux Systemd Daemon)

Kompyuter yoqilganda dastur o'zi avtomatik ishga tushishi uchun:

```bash
./install.sh --servis "rtsp://admin:parol@192.168.1.64:554/Streaming/Channels/101"
```

* Holatni tekshirish: `systemctl status vagon-detect`
* Jonli monitoring: `journalctl -u vagon-detect -f`

---

## 📬 Bog'lanish

* **Muallif:** Quvonchbek Begimqulov
* **LinkedIn:** [linkedin.com/in/kuvonchbek-begimkulov](https://linkedin.com/in/kuvonchbek-begimkulov)
* **Email:** [begimqulovquvonchbek53@gmail.com](mailto:begimqulovquvonchbek53@gmail.com)
* **GitHub:** [@quvonchbek212](https://github.com/quvonchbek212)
