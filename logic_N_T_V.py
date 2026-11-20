import cv2
import os
import sys
import numpy as np
import torch
import pathlib
import datetime
from ultralytics import YOLO
from collections import defaultdict, deque
import warnings
import time
import threading
import signal
import shutil
import traceback
import platform
import importlib.util

# VIDEO_SOURCE = './IMG_2085.MP4'
VIDEO_SOURCE = './IMG_2719.MOV'

# ============ VIDEO O'RNIGA -- HAR BIR VAGON UCHUN RASM SAQLASH ============
SAVE_RESULT = False
OUTPUT_FILENAME = 'realtime_result.mp4'
SAVE_RESIZE_WIDTH = None
SHOW_WINDOW = True
# Fayl videosi o'z tezligida (real vaqtda) o'tadi: 5 daqiqalik video roppa-rosa
# 5 daqiqada. Ishlov tez bo'lsa kutiladi, orqada qolsa eskirgan kadrlar tashlab
# yuboriladi -- jonli kameradagi kabi. False (yoki --tez) -- fayl imkon qadar tez ishlanadi.
# Jonli manbaga (kamera, RTSP) ta'sir qilmaydi: u o'zi real vaqtda keladi.
REALTIME_PLAYBACK = True

# ================== 24/7 ISHLASH (jonli kamera) ==================
# Kamera/RTSP uzilsa -- o'zi qayta ulanadi (1 s dan 30 s gacha oshib boruvchi kutish bilan).
LIVE_RECONNECT_MAX_WAIT = 30
# Kamera ulangan, lekin shuncha soniya yangi kadr bermasa -- oqim qotgan deb qayta ulanadi.
LIVE_STALL_SECONDS = 20
# FFMPEG ulanish/o'qish timeout'lari (ms) -- aks holda uzilgan RTSP da read() abadiy qotadi.
LIVE_OPEN_TIMEOUT_MS = 10000
LIVE_READ_TIMEOUT_MS = 10000
# Jurnalga har shuncha soniyada holat yoziladi (ishlayaptimi, fps, qayta ulanishlar).
HEARTBEAT_SECONDS = 600
# Kutilmagan xatodan keyin dastur shuncha soniyadan so'ng o'zi qayta ishga tushadi.
RESTART_DELAY_SECONDS = 5
# Natijalarni avtomatik tozalash: shuncha kundan eski rasm/hisobotlar o'chiriladi. 0 -- o'chirilmaydi.
KEEP_DAYS = 0

# ================== QURILMA (GPU / CPU) ==================
# "auto" -- NVIDIA GPU bo'lsa GPU, bo'lmasa CPU. Majburlash uchun: "cuda" yoki "cpu" (yoki --cpu).
DEVICE = "auto"
# CPU da video fayl: har bir kadr tahlil qilinadi (aniqlik GPU dagidek), real vaqt kutilmaydi --
# CPU ulgurmaydi va real vaqt rejimida kadrlarning ko'pi tashlanib, aniqlik tushardi.
CPU_PROCESS_EVERY_FRAME = True

SAVE_WAGON_IMAGES = True
WAGON_IMAGES_SUBDIR = "images"
COMBINED_IMAGE_MAX_WIDTH = 700
# =====================================================================

def ensure_lap_support():
    """
    Ensure ultralytics ByteTrack works even if C++ lap package is not compiled/installed.
    Provides a pure-Python Hungarian algorithm shim using scipy.optimize.linear_sum_assignment.
    """
    if 'lap' in sys.modules:
        return
    if importlib.util.find_spec('lap') is not None:
        return
    try:
        import types
        from scipy.optimize import linear_sum_assignment

        m = types.ModuleType('lap')
        m.__version__ = '0.5.13'

        def lapjv(cost_matrix, extend_cost=False, cost_limit=np.inf, return_cost=True):
            cost_matrix = np.asarray(cost_matrix, dtype=float)
            n_rows, n_cols = cost_matrix.shape
            x = np.full(n_rows, -1, dtype=int)
            y = np.full(n_cols, -1, dtype=int)
            if n_rows == 0 or n_cols == 0:
                return (0.0, x, y) if return_cost else (x, y)
            dim = max(n_rows, n_cols)
            max_c = np.max(cost_matrix) if cost_matrix.size > 0 else 0
            fill_val = max(max_c * 2 + 10.0, 1e6)
            square = np.full((dim, dim), fill_val, dtype=float)
            square[:n_rows, :n_cols] = cost_matrix
            row_ind, col_ind = linear_sum_assignment(square)
            total_cost = 0.0
            for r, c in zip(row_ind, col_ind):
                if r < n_rows and c < n_cols:
                    cost = cost_matrix[r, c]
                    if cost <= cost_limit:
                        x[r] = c
                        y[c] = r
                        total_cost += cost
            return (total_cost, x, y) if return_cost else (x, y)

        m.lapjv = lapjv
        sys.modules['lap'] = m
    except Exception:
        pass


ensure_lap_support()

warnings.filterwarnings("ignore", category=FutureWarning)
if os.name == "nt":
    # Linux'da saqlangan PosixPath .pt fayllar Windows'da ochilishi uchun
    try:
        pathlib.PosixPath = pathlib.WindowsPath
    except Exception:
        pass
else:
    # Windows'da saqlangan WindowsPath .pt fayllar Linux'da ochilishi uchun
    try:
        pathlib.WindowsPath = pathlib.PosixPath
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAGON_NUMBER_DETECTION = os.path.join(BASE_DIR, 'detection4.pt')
VAGON_NUMBER_CLASSIFICATION = os.path.join(BASE_DIR, 'classificationvagon.pt')

# =====================================================================
# ==================  t / v  (TONNA / M.KUB) BLOKI  ===================
# =====================================================================
# Raqam oqimi bilan BIR VAQTNING O'ZIDA ishlaydi, lekin undan MUTLAQO
# MUSTAQIL: raqam mantig'iga hech qanday ta'sir qilmaydi.
#
# MUHIM: bu blokda HECH QANDAY QIYMAT CHEGARASI YO'Q.
# Model nimani o'qisa -- o'sha ko'rsatiladi (75.0, 92, 24.5, 120 -- farqi yo'q).
TV_ENABLED = True
TV_DETECTION = os.path.join(BASE_DIR, 'detect-t-v.pt')
TV_CLASSIFICATION = os.path.join(BASE_DIR, 'classmodel.pt')

# O'nlik nuqta (belgi modelida vergul/nuqta klassi yo'q) -- tv_apply_decimal ga qarang.
# 3 xonali o'qish shu qiymatdan oshsa, o'nlik son deb talqin qilinadi.
# Chegaralar haqiqiy vagon xususiyatlaridan: eng katta yuk ko'tarish 125 t
# (8 o'qli polуvagon 12-508), eng katta kuzov hajmi ~250 m3 (krytyy vagon 11-1709),
# odatdagi hajmlar 73..158 m3.
TV_TONNAGE_MAX_WHOLE = 125     # t:  695 -> 69.5,  266 -> 26.6;  120 -> 120
TV_VOLUME_MAX_WHOLE = 265      # m3: 101, 120, 138, 158, 250 -- butun;  856 -> 85.6
# Arifmetika ikkiga bo'lina olmaydigan tor oraliq (t: 100..125, m3: 200..265) --
# faqat shu yerda vergul o'rni belgilar geometriyasidan aniqlanadi.
TV_AMBIGUOUS_MIN = {"t": 100, "v": 200}
# Fizik chegaralar: bulardan tashqaridagi qiymat vagonniki bo'lishi mumkin emas
# (eng yengil vagonning yuk ko'tarishi ~20 t, eng kichik kuzov hajmi ~20 m3).
# Chala o'qilgan plastinkadan chiqadigan "5 m3", "6 t" kabi qiymatlarni to'sadi.
TV_VALUE_RANGE = {"t": (20.0, 125.0), "v": (20.0, 265.0)}
# Plastinkada doim "т" yoki "м³" belgisi bo'ladi. Uni ko'rmasdan qilingan o'qish
# qabul qilinmaydi: aks holda vagondagi oddiy yozuvlar ("25 12 35", "ДР 25 12 27")
# ham qiymat bo'lib ketadi -- qidiruv oynasi butun vagonni qamragani uchun bu muhim.
TV_REQUIRE_UNIT = True
# Vergul turgan joyda kengroq bo'shliq qoladi. 361 ta haqiqiy plastinkada o'lchangan:
# vergulli qiymatlarda bo'shliq raqam kengligining 0.50-0.57 qismi, butunlarda 0.05.
TV_DOT_GAP_MIN = 0.35
TV_DOT_GAP_MARGIN = 0.20       # boshqa bo'shliqlardan shuncha kengroq bo'lishi shart
TV_DOT_HEIGHT_MAX = 0.85       # yoki o'nlik qism shu nisbatdan past bo'lsa

TV_DET_CONF = 0.45          # blur/yarim ko'ringan plastinkalar garbage qiymat beradi
# Quyidagi to'rttasi 159 ta haqiqiy cropda to'liq qidiruv bilan tanlangan.
# imgsz=640 ni O'ZGARTIRMANG: model 320 da o'qitilgan, 640 da aniqlik
# 77% dan 20% ga tushadi.
TV_CLS_CONF = 0.25
TV_CLS_IOU = 0.45
TV_CLS_AGNOSTIC = True
TV_DET_IMGSZ = 1024
TV_CLS_IMGSZ = 320
# t/v detektori (1024 px, har kadrda) eng og'ir qism: i5-1135G7 da PyTorch CPU 740 ms,
# OpenVINO + Intel iGPU 111 ms/kadr, natijasi bir xil. None -- oddiy PyTorch CPU.
TV_DET_OPENVINO_DEVICE = None   # avtomatik: setup_device() GPU yo'q Intel/AMD kompyuterda "intel:cpu" qiladi

# Tezlik: plastinka detektori (1024 px) + har bir plastinkani o'qish eng og'ir qism.
# Bir vagon kadrda 100+ marta o'qiladi, holbuki qiymat uchun o'nga yaqini yetarli.
TV_FRAME_STRIDE = 1         # har kadrda qidiriladi. 2 da 96291976 vagonining "81,64 m3"
                            # plastinkasi yo'qolardi: u butun videoda atigi 5 kadrda
                            # aniqlangan. Asosiy tejamkorlik TV_ENOUGH_READS da.
TV_READ_MIN_H = 11          # bundan past plastinka o'qilmaydi. 640x360 videolarda
                            # plastinkalar 10-14 px bo'ladi va ba'zilari o'qiladi;
                            # axlat o'qishlarni birlik va qiymat filtrlari to'sadi.
TV_ENOUGH_READS = 25        # slotda shuncha o'qish bo'lsa, yangilari qiymatni o'zgartirmaydi
# Vagon raqami kadrda topilmagan paytda ham plastinka o'z vagoniga biriktirilsin:
# trekning oxirgi qutisi o'lchangan tezlik bo'yicha shuncha kadrgacha oldinga siljitiladi.
# (Plastinka eng katta ko'ringan payt -- ko'pincha raqam kadrdan chiqib ketgan payt.)
TV_PREDICT_MAX_FRAMES = 120
# Vagon raqami kadrdan chiqib ketgach trek FRAMES_TO_FORGET dan keyin yopiladi va vagon
# hisobotga yoziladi. Plastinka esa ko'pincha shundan keyin kadrga kiradi (u raqamdan
# o'ngroqda). Shuning uchun yopilgan vagonga ham shuncha kadr davomida qiymat qo'shiladi.
TV_LATE_VALUE_FRAMES = 200
# Plastinka vagon raqamidan O'NGDA, shu balandlikda joylashadi. O'lchov:
# to'g'ri biriktirishlarda dx=+1.1..+2.2, dy<0.2; xatolarida dx manfiy, dy>0.65.
TV_ASSOC_MAX_DX = 3.0       # raqam qutisi kengligining ulushi -- o'ngga
TV_ASSOC_MIN_DX = -1.0      # chapga: plastinka raqamning ostida/tepasida ham bo'lishi mumkin
TV_ASSOC_MAX_DY = 3.00      # raqam qutisi balandligining ulushi -- butun vagon bo'ylab
                            # (plastinka vagonning tepasida yoki eng pastida ham
                            # bo'lishi mumkin; vertikal yo'nalishda raqib vagon yo'q)
# Bitta tasodifiy o'qish qiymat bo'lib qolmasin: 2 687 kadrlik yozuvda bitta vagonda
# tonna 77 marta, hajm esa atigi 1 marta o'qilgan va o'sha yolg'on "68 m3" hisobotga tushgan.
TV_MIN_READS = 2
# Yakka o'qish ham shu balldan yuqori bo'lsa qabul qilinadi (ball = belgi ishonchi x
# plastinka ishonchi). Yozuvda haqiqiy o'qishlar 0.83-0.90 oralig'ida, yolg'on
# "68 m3" esa 0.76 bo'lgan -- shuning uchun chegara 0.82.
TV_SINGLE_READ_MIN_SCORE = 0.82
# G'olib qiymat AYNAN shu ko'rinishda kamida shuncha marta o'qilgan bo'lsin va o'z
# uzunligidagi o'qishlarning shu ulushini egallasin. Past sifatli (640x360) videoda
# mayda plastinka bir-ikki marta xato o'qilib, "70,0т" -> "100" yoki "20.0" bo'lib
# ketardi: bunday o'qishlar bir-biriga o'xshamaydi, shuning uchun ulushi past bo'ladi.
TV_MIN_SAME_READS = 2
TV_MIN_SHARE = 0.75
# G'olib satr o'qishlarining o'rtacha ishonchi (belgi modeli x plastinka detektori) shundan
# past bo'lsa -- qiymat xira kadrlardan yig'ilgan, ishonchsiz: chiqarilmaydi ("--").
TV_MIN_AVG_SCORE = 0.4
TV_BIG_PLATE_H = 25            # shundan baland plastinka aniq hisoblanadi (yakka o'qishga ruxsat)

# Birlik ("t"/"m3") qutisi ichidagi raqamni soxta deb tashlash (tv_suppress_inside_unit).
# Eski cllas-detect.pt "М" shtrixini "1" deb o'qigani uchun kerak edi. classmodel.pt da
# O'CHIQ: haqiqiy 419 kesimda u 30 ta to'g'ri o'qishni buzdi (856 -> 85, 92 -> 9,
# 731 -> 73) -- raqam aniqligi 95.5% dan 88.3% ga tushdi.
TV_SUPPRESS_INSIDE_UNIT = False
# Raqam qutisi shu ulushdan ko'prog'i bilan birlik ("t"/"m3") qutisi ichida
# yotsa -- u birlik glifidan chiqqan soxta raqam deb tashlanadi.
TV_UNIT_OVERLAP_DISCARD = 0.55
# Birlik qutisi tekshiruvdan oldin chapga shu ulushda kengaytiriladi: model
# kirill "М" ning chap shtrixini "1" deb o'qiydi, lekin "m3" qutisini shtrixdan
# KEYIN boshlaydi -- shuning uchun shtrix quti tashqarisida qolib ketadi.
TV_UNIT_LEFT_MARGIN_RATIO = 0.40
# Kesilgan o'qish ("52" o'rniga "5") o'rniga uzunrog'i olinishi uchun
# uzunroq variant shu ulushda ovoz to'plagan bo'lishi kerak.
TV_LONGER_PREFIX_MIN_SHARE = 0.50

TV_CROP_TARGET_HEIGHT = 96  # kichik croplar shu balandlikka cho'ziladi

# Plastinka qiymati -- birlik belgisining ("t", "m3") CHAP tomonidagi, bir-biriga
# yopishgan raqamlar guruhi. Kesimga tushib qolgan chetdagi begona raqamlar
# ("59 74 m3" -> 5974 -> 59.74) va birlikdan o'ngdagilar olinmaydi.
TV_GROUP_GAP_MAX = 1.2          # raqamlar orasidagi bo'shliq, raqam kengligi ulushida
TV_UNIT_GAP_MAX = 3.0           # oxirgi raqam bilan birlik orasidagi bo'shliq
TV_DEBUG = False            # har bir o'qishni konsolga chiqarish (True = debug uchun)

# Belgi modelidagi nomlar -> ma'no
TV_DOT_NAMES = {"nuqta", "dot", "point", "."}
TV_UNIT_T_NAMES = {"t", "т", "tonna"}
TV_UNIT_M3_NAMES = {"m3", "м3", "m³", "v", "mkub"}
# =====================================================================

# ============ SOZLAMALAR ============
DIGIT_CONF = 0.38
# detection4.pt da bitta obyekt ikki nomda o'qitilgan ("number_zone" va "Wagon-number",
# qutilari deyarli bir xil). Klass bo'yicha NMS ikkalasini qoldiradi -- har plastinkaga
# ikki quti, ikki trek. Klassdan qat'i nazar NMS bitta quti qoldiradi.
WAGON_DETECTOR_AGNOSTIC_NMS = True
# Raqam modeli bitta belgiga ikki xil klass qutisini qo'yadi ("2" va "9") -- 9 xonali
# o'qish chiqadi. Klassdan qat'i nazar NMS + boshqa qutining ichida qolgan quti
# (masalan "7" ning vertikal chizig'i alohida "1" bo'lib) olib tashlanadi.
DIGIT_AGNOSTIC_NMS = True
DIGIT_CONTAIN_RATIO = 0.60      # kichik qutining shuncha qismi kattasining ichida -- takror
WAGON_DETECTOR_CONF = 0.35
ZONE_TOLERANCE = 0.32

CROP_PADDING_RATIO_V = 0.12
CROP_PADDING_RATIO_H = 0.05
CROP_PADDING_MIN = 8
CROP_PADDING_MAX = 40
HEIGHT_OUTLIER_RATIO = 0.55
DIGIT_GAP_RATIO = 1.9
REPAIRED_SCORE_MULTIPLIER = 0.5

MIN_FRAMES_TO_SURVIVE = 3
MIN_CONF_TO_FIX = 0.60
SHOW_LIVE_LABEL = True
FRAMES_TO_FORGET = 45
SHOW_WAGON_COUNTER = False
SHOW_RECOGNIZED_LIST = True
RECOGNIZED_LIST_ON_RIGHT = True     # ro'yxat kadrning o'ng tomonida
RECOGNIZED_LIST_MARGIN_RIGHT = 28   # o'ng chetdan bo'shliq
RECOGNIZED_LIST_DIM = 0.25          # yozuv ortidagi fon shu darajada qoraytiriladi
RECOGNIZED_LIST_FONT_SCALE = 0.62
RECOGNIZED_LIST_LINE_HEIGHT = 26
RECOGNIZED_LIST_MARGIN_LEFT = 14
RECOGNIZED_LIST_MARGIN_TOP = 25
RECOGNIZED_LIST_MARGIN_BOTTOM = 20
# Yozuv o'lchovlari 1280 px kenglikka nisbatan beriladi va kadr kengligiga
# mutanosib o'zgaradi (ui_scale_of). Shunda 640x360 va 2688x1520 videolarda ham
# oynada bir xil ko'rinadi: oyna kadrni 1280 px ga keltiradi.
DISPLAY_MIN_WIDTH = 1280
DEBUG_RAW_DETECTIONS = False

ADAPTIVE_HEIGHT_RATIO = 0.55
MIN_SAMPLES_FOR_ADAPTIVE = 15
HEIGHT_HISTORY_MAXLEN = 200
PRINT_BOX_HEIGHT_DEBUG = False

ROW_Y_TOLERANCE_RATIO = 0.6

DIGIT_CHARS = set("0123456789")
NON_DIGIT_DISCARD_THRESHOLD = 1

MISSING_SIDE_GAP_RATIO = 0.55

ENABLE_MISSING_LAST_DIGIT_GUESS = False

MIN_PLATE_ASPECT_RATIO = 2.2

FAST_INSTANT_ACCEPT_CONF = 0.62
FAST_MIN_READ_FRAMES = 3
# Pozitsiya ovozi yetakchisi checksumdan o'tmasa, zaxira nomzod trekdagi 8 xonali
# o'qishlarning kamida shuncha ulushini olgan bo'lishi shart. Aks holda plastinka
# izchil ravishda boshqa narsa deb o'qilgan -- masalan lokomotivdagi "ТЭМ2-059"
# (harflar raqam deb o'qiladi: 73112059), va 6% tasodifiy o'qish vagon bo'lib ketardi.
FALLBACK_MIN_SHARE = 0.30
POSITION_VOTE_MIN_FRAMES_STRONG = 3

SPACING_UNIFORMITY_MAX_RATIO = 2.2
# Ba'zi vagonlarda (masalan yopiq vagonlar) raqam ikki guruhda yoziladi: "2358 8106".
# Bunday raqamda 4- va 5-raqam orasida bitta katta bo'shliq bo'ladi.
SPLIT_NUMBER_GROUPS = True
# 9 ta raqamdan qaysi biri ortiqcha -- bir nechta to'g'ri variant bo'lsa, eng tekis
# qadamli variant keyingisidan shuncha barobar tekis bo'lishi shart.
EXTRA_DIGIT_GEOMETRY_MARGIN = 1.25
# Och rangli vagonlarda (sisterna, kumush/sariq kuzov) raqam QORA bo'yoqda yoziladi, raqam
# modeli esa asosan to'q fondagi oq raqamlarda o'qitilgan. Oddiy o'qish 8 ta raqam
# bermasa, kesim invertlanib (oq<->qora) qayta o'qiladi. Normal plastinkada qo'shimcha
# xarajat yo'q -- faqat o'qish muvaffaqiyatsiz bo'lganda ishlaydi.
DIGIT_TRY_INVERTED = True
SPLIT_GROUP_HEIGHT_TOL = 0.25   # ikki guruh balandliklari shu ulushdan ko'p farq qilmasin

DIAGNOSTIC_LOG_ENABLED = False

# Sessiyaning birinchi N treki hisobga olinmaydi ("odatda lokomotiv").
# 1 dan 0 ga tushirildi: real sinovda bu filtr HAQIQIY vagonni yo'q qildi
# (IMG_2723 dagi 62624739). Sababi -- sessiya noto'g'ri joyda yopilib, keyingi
# vagon "yangi sessiyaning birinchi treki" bo'lib qolgan edi. Lokomotivdan
# himoya uchun checksum va ko'p-freymli ovoz berish allaqachon yetarli:
# lokomotiv korpusida 8 xonali UIC raqami yo'q. Zarur bo'lsa 1 ga qaytariladi.
IGNORE_FIRST_N_TRACKS_PER_SESSION = 0

KNOWN_FALSE_POSITIVE_NUMBERS = {
    # "XXXXXXXX",
}

ENABLE_POSITION_OUTLIER_FILTER = True
POSITION_OUTLIER_THRESHOLD = 0.18
MIN_WAGONS_FOR_POSITION_MEDIAN = 3

MIN_INTER_WAGON_SECONDS = 2.0
FRAGMENTATION_STRONG_VOTE_MIN = 3

TRAINS_OUTPUT_DIR = os.path.join(BASE_DIR, "trains")
TRAIN_SESSION_GAP_SECONDS = 300   # oxirgi vagon ko'rinmay qolgandan shuncha soniya o'tsa -- poyezd ketdi
# Video fayl yozib olingan real vaqt (kamera ekranidagi soatdan): "2025-12-10 16:05:00" yoki --vaqt.
# Berilmasa -- dastur ishga tushgan vaqtdan hisoblanadi. Jonli kamerada har doim real soat.
VIDEO_START_TIME = None
# Poyezd to'xtab turib, raqam ko'rinib tursa ham -- shuncha vaqt yangi vagon qo'shilmasa sessiya yopiladi
# (kamera oldida qoldirib ketilgan vagon keyingi poyezdni o'ziga qo'shib olmasin).
TRAIN_MAX_STOP_SECONDS = 3600
DUPLICATE_NUMBER_COOLDOWN_SECONDS = 120

CROP_MIN_HEIGHT_FOR_UPSCALE = 70
CROP_UPSCALE_FACTOR = 2.2


def sharpen_crop(image):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1.8, blurred, -0.8, 0)
    return sharpened


def maybe_upscale_crop(crop):
    """Kichik (uzoqdagi vagon) croplarni klassifikatordan oldin kattalashtiradi."""
    h = crop.shape[0]
    if h > 0 and h < CROP_MIN_HEIGHT_FOR_UPSCALE:
        new_crop = cv2.resize(crop, None, fx=CROP_UPSCALE_FACTOR, fy=CROP_UPSCALE_FACTOR,
                               interpolation=cv2.INTER_CUBIC)
        return new_crop, CROP_UPSCALE_FACTOR
    return crop, 1.0


# =====================================================================
# ====================  t / v  YORDAMCHI FUNKSIYALARI  ================
# =====================================================================

def ui_scale_of(width):
    """Yozuv va chiziqlar o'lchovi uchun koeffitsient: 1280 px kenglik -- birlik."""
    return max(0.35, width / 1280.0)


def check_gui_support():
    """
    cv2.imshow ishlay oladimi -- oldindan tekshiradi.
    opencv-python-headless o'rnatilgan bo'lsa oyna umuman ochilmaydi,
    shuning uchun buni BOSHIDA aniqlab, foydalanuvchiga aytamiz.
    """
    if not SHOW_WINDOW:
        return False
    if os.name != "nt" and sys.platform != "darwin" and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        # Ekransiz server/servis: Qt oynasi ochilmoqchi bo'lsa jarayonni butunlay o'ldiradi.
        print("ℹ️  Ekran (DISPLAY) yo'q -- oynasiz rejimda ishlaydi, natijalar saqlanaveradi.")
        return False
    try:
        info = cv2.getBuildInformation()
        has_gui = ("GUI: " in info and "NONE" not in info.split("GUI:")[1][:60])
        if not has_gui:
            print("ℹ️  OpenCV GUIsiz (headless) o'rnatilgan -- oynasiz rejimda ishlaydi.")
            return False
        test = np.zeros((10, 10, 3), np.uint8)
        cv2.imshow("__gui_test__", test)
        cv2.waitKey(1)
        cv2.destroyWindow("__gui_test__")
        return True
    except Exception as e:
        print("\n" + "!" * 68)
        print("⚠️  REAL-TIME OYNA OCHILMAYDI -- cv2.imshow ishlamayapti.")
        print(f"   Sabab: {e}")
        print("   Eng ko'p uchraydigan sabab: 'opencv-python-headless' o'rnatilgan")
        print("   (u serverlar uchun, GUIsiz versiya). Tuzatish:")
        print("       pip uninstall opencv-python-headless opencv-python -y")
        print("       pip install opencv-python")
        print("   Vaqtinchalik yechim: yuqorida SAVE_RESULT = True qiling --")
        print("   natija video fayl sifatida saqlanadi.")
        print("!" * 68 + "\n")
        return False


def tv_upscale(crop):
    """t/v kropini aniq balandlikka cho'zadi (juda kichik bo'lgani uchun)."""
    h = crop.shape[0]
    if h <= 0:
        return crop, 1.0
    if h < TV_CROP_TARGET_HEIGHT:
        f = TV_CROP_TARGET_HEIGHT / h
        return cv2.resize(crop, None, fx=f, fy=f,
                          interpolation=cv2.INTER_CUBIC), f
    return crop, 1.0


def tv_suppress_inside_unit(items):
    """
    Birlik glifi ("t" / "m3") ichidan chiqqan soxta raqamlarni olib tashlaydi.

    Model "92 m3" plastinkasida muntazam ravishda kirill "М" ning chap vertikal
    shtrixini "1" deb o'qiydi (ishonchi 0.86 gacha chiqadi -- conf chegarasi
    yordam bermaydi) va qiymat 92 o'rniga 921 bo'lib ketadi.

    Birlik qutisi esa o'sha shtrixdan KEYIN boshlanadi, shuning uchun quti
    tekshiruvdan oldin chapga kengaytiriladi -- aks holda shtrix tashqarida
    qolib ketadi. Haqiqiy raqamlar birlikdan sezilarli bo'shliq bilan ajralgan,
    shuning uchun kengaytirilgan hudud ularga tegmaydi.
    """
    unit_boxes = [it for it in items
                  if it[4] in TV_UNIT_T_NAMES or it[4] in TV_UNIT_M3_NAMES]
    if not unit_boxes:
        return items

    zones = []
    for ux1, uy1, ux2, uy2, _, _ in unit_boxes:
        zones.append((ux1 - (ux2 - ux1) * TV_UNIT_LEFT_MARGIN_RATIO, uy1, ux2, uy2))

    kept = []
    for it in items:
        x1, y1, x2, y2, nm, _ = it
        if not (len(nm) == 1 and nm.isdigit()):
            kept.append(it)
            continue
        area = max(1.0, (x2 - x1) * (y2 - y1))
        swallowed = False
        for ux1, uy1, ux2, uy2 in zones:
            iw = min(x2, ux2) - max(x1, ux1)
            ih = min(y2, uy2) - max(y1, uy1)
            if iw > 0 and ih > 0 and (iw * ih) / area >= TV_UNIT_OVERLAP_DISCARD:
                swallowed = True
                break
        if not swallowed:
            kept.append(it)
    return kept


def tv_dot_position(digit_boxes):
    """Vergul nechta raqamdan keyin turganini belgilar geometriyasidan topadi.
    Belgi modelida vergul klassi yo'q, lekin vergul turgan joyda kengroq bo'shliq
    qoladi va o'nlik qism ko'pincha kichikroq yoziladi. Topilmasa -- None."""
    if len(digit_boxes) < 3:
        return None
    ws = sorted(b[2] - b[0] for b in digit_boxes)
    mw = ws[len(ws) // 2]
    if mw <= 0:
        return None
    gaps = [(digit_boxes[i + 1][0] - digit_boxes[i][2]) / mw
            for i in range(len(digit_boxes) - 1)]
    i = max(range(len(gaps)), key=lambda k: gaps[k])
    others = [g for k, g in enumerate(gaps) if k != i]
    wide = gaps[i] >= TV_DOT_GAP_MIN and (not others or gaps[i] - max(others) >= TV_DOT_GAP_MARGIN)
    hs = [b[3] - b[1] for b in digit_boxes]
    head, tail = hs[:i + 1], hs[i + 1:]
    head_h = sorted(head)[len(head) // 2]
    tail_h = sorted(tail)[len(tail) // 2]
    small = head_h > 0 and tail_h / head_h <= TV_DOT_HEIGHT_MAX
    return i + 1 if (wide or small) else None


def tv_value_group(items, prefer=None):
    """Birlik belgisidan chapga qarab, bir-biriga yopishgan raqamlar guruhini qaytaradi
    (birlik bilan birga). Birlik topilmasa -- o'zgarishsiz."""
    is_unit = lambda nm: nm in TV_UNIT_T_NAMES or nm in TV_UNIT_M3_NAMES
    is_digit = lambda nm: len(nm) == 1 and nm.isdigit()
    units = [i for i, it in enumerate(items) if is_unit(it[4])]
    if not units:
        return items
    digit_w = sorted(it[2] - it[0] for it in items if is_digit(it[4]))
    if not digit_w:
        return items
    mw = max(1.0, digit_w[len(digit_w) // 2])
    if prefer in ("t", "v"):
        # Tonna va hajm bitta qatorda yonma-yon yozilgan bo'lsa ("69т 74м³") ikkala
        # plastinka qutisi ham butun qatorni oladi -- detektor aytgan turdagi birlik olinadi.
        names = TV_UNIT_T_NAMES if prefer == "t" else TV_UNIT_M3_NAMES
        same = [u for u in units if items[u][4] in names]
        if same:
            units = same
    best = None
    for u in units:
        group, right_x1 = [], items[u][0]
        limit = TV_UNIT_GAP_MAX
        for it in reversed(items[:u]):
            if not (is_digit(it[4]) or it[4] in TV_DOT_NAMES):
                break
            if (right_x1 - it[2]) / mw > limit:
                break
            group.append(it)
            right_x1, limit = it[0], TV_GROUP_GAP_MAX
        if group and (best is None or len(group) > len(best[0])):
            best = (group, items[u])
    if best is None:
        return items
    return list(reversed(best[0])) + [best[1]]


def tv_read_chars(tv_classifier, crop, prefer=None):
    """
    t/v plastinkasi kropini o'qiydi -- HECH QANDAY QIYMAT CHEGARASI YO'Q.
    Model nimani ko'rsa, chapdan o'ngga tartiblab qaytaradi.

    Qaytaradi: (belgilar_ro'yxati, birlik, o'rtacha_ishonch, vergul_o'rni)
      (["7","5","0"], "t", 0.83, 2)    (["9","2"], "m3", 0.79, None)
    Hech narsa o'qilmasa: (None, None, 0.0, None)
    """
    if crop is None or crop.size == 0:
        return None, None, 0.0, None

    work, _ = tv_upscale(crop)
    try:
        res = tv_classifier.predict(work, conf=TV_CLS_CONF, iou=TV_CLS_IOU,
                                    imgsz=TV_CLS_IMGSZ, agnostic_nms=TV_CLS_AGNOSTIC,
                                    verbose=False, device=dev())[0]
    except Exception:
        return None, None, 0.0, None

    if res.boxes is None or len(res.boxes) == 0:
        return None, None, 0.0, None

    names = tv_classifier.names
    items = []
    for (bx1, by1, bx2, by2), ci, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                            res.boxes.cls.cpu().numpy().astype(int),
                                            res.boxes.conf.cpu().numpy()):
        nm = names[int(ci)] if isinstance(names, list) else names.get(int(ci), str(int(ci)))
        items.append((float(bx1), float(by1), float(bx2), float(by2),
                      str(nm).strip().lower(), float(cf)))

    if not items:
        return None, None, 0.0, None

    if TV_SUPPRESS_INSIDE_UNIT:
        items = tv_suppress_inside_unit(items)
    items.sort(key=lambda it: (it[0] + it[2]) / 2.0)
    items = tv_value_group(items, prefer)
    dot_pos = tv_dot_position([it[:4] for it in items
                               if len(it[4]) == 1 and it[4].isdigit()])
    digits, unit, confs = [], None, []
    for _, _, _, _, nm, cf in items:
        if nm in TV_DOT_NAMES:
            digits.append(".")
            confs.append(cf)
        elif nm in TV_UNIT_T_NAMES:
            unit = "t"
            confs.append(cf)
        elif nm in TV_UNIT_M3_NAMES:
            unit = "m3"
            confs.append(cf)
        elif len(nm) == 1 and nm.isdigit():
            digits.append(nm)
            confs.append(cf)
        else:
            # noma'lum belgi -- shunchaki e'tiborsiz qoldiriladi,
            # butun o'qish BEKOR QILINMAYDI
            continue

    if not any(ch.isdigit() for ch in digits):
        return None, None, 0.0, None

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return digits, unit, avg_conf, dot_pos


def load_tv_detector():
    """t/v detektorini yuklaydi: iloji bo'lsa OpenVINO (Intel iGPU), bo'lmasa PyTorch.
    Qaytaradi: (model, predict uchun device yoki None)."""
    if TV_DET_OPENVINO_DEVICE:
        ov_dir = os.path.splitext(TV_DETECTION)[0] + "_openvino_model"
        try:
            if not os.path.isdir(ov_dir) or os.path.getmtime(TV_DETECTION) > os.path.getmtime(ov_dir):
                print("   OpenVINO formatiga o'tkazilmoqda (bir marta)...")
                YOLO(TV_DETECTION).export(format="openvino", imgsz=TV_DET_IMGSZ, verbose=False)
            model = YOLO(ov_dir, task="detect")
            model.predict(np.zeros((64, 64, 3), np.uint8), imgsz=TV_DET_IMGSZ,
                          device=TV_DET_OPENVINO_DEVICE, verbose=False)
            print(f"   t/v detektori OpenVINO ({TV_DET_OPENVINO_DEVICE}) da ishlaydi")
            return model, TV_DET_OPENVINO_DEVICE
        except Exception as e:
            print(f"   OpenVINO ishlamadi ({e}) -- PyTorch CPU ishlatiladi")
    return YOLO(TV_DETECTION), None


def tv_detect_plates(tv_detector, frame, device=None):
    """Butun freymdan barcha t/v plastinkalarini topadi."""
    try:
        kw = {"device": device if device else dev()}
        res = tv_detector.predict(frame, conf=TV_DET_CONF, iou=0.5,
                                  imgsz=TV_DET_IMGSZ, verbose=False, **kw)[0]
    except Exception:
        return []

    if res.boxes is None or len(res.boxes) == 0:
        return []

    names = tv_detector.names
    out = []
    for (x1, y1, x2, y2), ci, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                        res.boxes.cls.cpu().numpy().astype(int),
                                        res.boxes.conf.cpu().numpy()):
        nm = names[int(ci)] if isinstance(names, list) else names.get(int(ci), str(int(ci)))
        nm = str(nm).strip().lower()
        kind = "t" if nm in ("t", "т") else ("v" if nm in ("v", "м3", "m3") else None)
        if kind is None:
            continue
        out.append({
            "kind": kind,
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
            "conf": float(cf),
        })
    return out


def tv_assign_to_tracks(tv_plates, boxes, track_ids):
    """
    Har bir t/v plastinkasini O'Z vagonining trekiga biriktiradi.

    Oyna butun vagon bo'ylab: plastinka har doim raqamning yonida turmaydi --
    vagonning tepasida yoki eng pastida ham bo'lishi mumkin. Shuning uchun
    VERTIKAL oyna keng (kadrda bir vaqtda bitta vagon bo'lgani uchun tepada yoki
    pastda raqib vagon yo'q), GORIZONTAL oyna esa cheklangan: aks holda qo'shni
    vagonning plastinkasi yopishib, qiymatlar aralashib ketadi.

    O'lchov (7 525 kadrlik yozuv): haqiqiy plastinkalar raqamdan 0.9..2.5 quti
    kengligi o'ngda, 32% i esa raqam markazidan tepada turgan.
    """
    assigned = defaultdict(list)
    if not tv_plates or not boxes:
        return assigned

    centers = []
    for (bx1, by1, bx2, by2) in boxes:
        centers.append(((bx1 + bx2) / 2.0, (by1 + by2) / 2.0,
                        max(1.0, bx2 - bx1), max(1.0, by2 - by1)))

    for p in tv_plates:
        best_i, best_d = None, None
        for i, (cx, cy, w, h) in enumerate(centers):
            dx = (p["cx"] - cx) / w
            dy = abs(p["cy"] - cy) / h
            if not (TV_ASSOC_MIN_DX <= dx <= TV_ASSOC_MAX_DX) or dy > TV_ASSOC_MAX_DY:
                continue
            if best_d is None or abs(dx) < best_d:
                best_d, best_i = abs(dx), i
        if best_i is not None:
            assigned[track_ids[best_i]].append(p)
    return assigned


def draw_plate_label(frame, text, x1, y1, color, ui=2.1):
    """t/v qiymatini o'z plastinkasining ustiga yozadi (vagon raqami kabi).
    `ui` -- kadr kengligiga mutanosib koeffitsient (ui_scale_of)."""
    scale = 0.43 * ui
    thick = max(1, round(0.95 * ui))
    pad = max(2, int(3.3 * ui))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    ty = max(th + pad, y1 - 8)
    bx1, by1 = x1 - pad, ty - th - pad
    bx2, by2 = x1 + tw + pad, ty + base
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
    cv2.putText(frame, text, (x1, ty), cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thick, cv2.LINE_AA)


def tv_apply_decimal(value, kind, dot_pos=None):
    """
    Belgi modelida vergul klassi yo'q: plastinkadagi o'nlik qism oddiy raqamlar
    bo'lib o'qiladi ("69,5т" -> "695", "85,6м³" -> "856"). Qiymat ikki yo'l bilan
    tiklanadi.

    1) ARIFMETIKA -- haqiqiy vagon chegaralaridan. Eng katta yuk ko'tarish 125 t,
       eng katta kuzov hajmi ~250 m3; shundan oshgan o'qish vergulli bo'lishi shart.
       Belgilar datasetidagi 2 145 ta asl plastinkada o'nlik qism doim 2 xonali
       butun qismdan keyin kelgan.

           t:  69 -> 69     102 -> 102    695 -> 69.5   266 -> 26.6   2500 -> 25.00
           m3: 92 -> 92     138 -> 138    250 -> 250    856 -> 85.6   7238 -> 72.38

    2) GEOMETRIYA (`dot_pos`) -- faqat arifmetika ikkiga bo'lina olmaydigan tor
       oraliqda: t 100..125, m3 200..265. Masalan "250" o'qishi ham haqiqiy 250 m3,
       ham "25,0" bo'lishi mumkin -- bunda vergul o'rnidagi bo'shliq hal qiladi.
    """
    if not value.isdigit() or len(value) not in (3, 4) or value[0] == "0":
        return value
    max_whole = TV_TONNAGE_MAX_WHOLE if kind == "t" else TV_VOLUME_MAX_WHOLE
    head = int(value[:3])
    if head > max_whole:                        # arifmetika o'zi hal qiladi
        return value[:2] + "." + value[2:]
    if len(value) == 4:                         # 1385 -> 138.5,  2500 -> 250.0
        return value[:3] + "." + value[3]
    if dot_pos and 0 < dot_pos < len(value) and head >= TV_AMBIGUOUS_MIN.get(kind, 10 ** 9):
        return value[:dot_pos] + "." + value[dot_pos:]
    return value


def tv_plausible(value, kind):
    """Qiymat haqiqiy vagonnikiga o'xshaydimi (TV_VALUE_RANGE ga qarang)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    lo, hi = TV_VALUE_RANGE.get(kind, (0.0, float("inf")))
    return lo <= v <= hi


def make_tv_slot():
    """Bitta plastinka turi (t yoki v) uchun ovoz idishi."""
    return {"len": defaultdict(float), "cnt": defaultdict(int), "best": defaultdict(float),
            "dot": defaultdict(float), "pos": {}, "crop": None, "best_conf": 0.0,
            # aynan o'qilgan satrlar: ball, soni va plastinkaning eng katta balandligi
            "str": defaultdict(float), "strn": defaultdict(int), "strh": defaultdict(int)}


def tv_add_vote(slot, chars, score, crop, dot_pos=None):
    """
    Butun satr o'rniga HAR BIR POZITSIYA alohida ovoz to'playdi
    (vagon raqamidagi POSITION-VOTE bilan bir xil mantiq).

    "65", "69", "66" kabi o'qishlar endi bir-birini kuchaytiradi:
    0-pozitsiya 6,6,6 -> "6", 1-pozitsiya 5,9,6 -> ko'pchilik yutadi.
    Uzunligi boshqacha (tushib qolgan raqamli) o'qishlar aralashib ketmaydi.
    """
    n = len(chars)
    if n == 0:
        return
    slot["len"][n] += score
    slot["cnt"][n] += 1
    slot["best"][n] = max(slot["best"][n], score)
    slot["dot"][(n, dot_pos)] += score
    text = "".join(chars)
    slot["str"][text] += score
    slot["strn"][text] += 1
    if crop is not None and crop.size:
        slot["strh"][text] = max(slot["strh"][text], crop.shape[0])
    pv = slot["pos"].setdefault(n, [defaultdict(float) for _ in range(n)])
    for i, ch in enumerate(chars):
        pv[i][ch] += score
    if crop is not None and crop.size > 0:
        # Hisobot rasmiga tushadigan krop faqat ishonch bo'yicha tanlansa, harakatdan
        # xiralashgan kadr tushib qolardi. Tiniqlik (Laplasian dispersiyasi) va
        # plastinkaning kattaligi ham hisobga olinadi.
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        quality = (score * min(1.0, cv2.Laplacian(g, cv2.CV_64F).var() / 150.0)
                   * min(1.0, crop.shape[0] / 40.0))
        if quality > slot["best_conf"]:
            slot["best_conf"] = quality
            slot["crop"] = crop.copy()


def tv_pick_winner(slot):
    """Eng ko'p ovoz olgan uzunlikni tanlab, o'sha uzunlikdagi o'qishlar
    bo'yicha har bir pozitsiyaning g'olib belgisini yig'adi."""
    if not slot or not slot["len"]:
        return None, None, None

    def voted(n):
        pv = slot["pos"].get(n)
        if not pv or any(not d for d in pv):
            return None
        return "".join(max(d.items(), key=lambda kv: kv[1])[0] for d in pv)

    # Kamida TV_MIN_READS marta o'qilgan, YOKI bir marta bo'lsa ham yuqori ballik
    # uzunliklar. Shunda tez o'tgan vagonning yagona, lekin ishonchli o'qishi yo'qolmaydi,
    # past ballik tasodifiy o'qish esa o'tmaydi.
    enough = {n: sc for n, sc in slot["len"].items()
              if slot["cnt"].get(n, 0) >= TV_MIN_READS
              or slot["best"].get(n, 0.0) >= TV_SINGLE_READ_MIN_SCORE}
    if not enough:
        return None, None, None
    best_n = max(enough.items(), key=lambda kv: kv[1])[0]
    val = voted(best_n)
    if val is None:
        return None, None, None

    # Uzoqdagi mayda plastinkada oxirgi raqam tez-tez tushib qoladi ("52" -> "5").
    # Qisqa o'qish uzunrog'ining aynan boshlanishi bo'lsa -- u kesilgan variant,
    # shuning uchun yetarlicha ovoz to'plagan uzunrog'i afzal ko'riladi.
    for n in sorted(enough, reverse=True):
        if n <= best_n:
            break
        longer = voted(n)
        if (longer and longer.startswith(val)
                and enough[n] >= enough[best_n] * TV_LONGER_PREFIX_MIN_SHARE):
            val = longer
            break

    # G'olib qiymat haqiqatan shunday o'qilganmi va o'z uzunligidagi o'qishlar
    # ichida ustunmi -- aks holda bu bir-biriga zid o'qishlardan yig'ilgan qiymat.
    same_len = {t: sc for t, sc in slot["str"].items() if len(t) == len(val)}
    if sum(same_len.values()) > 0 and same_len.get(val, 0.0) / sum(same_len.values()) < TV_MIN_SHARE:
        return None, None, None
    if slot["strn"].get(val, 0) and slot["str"].get(val, 0.0) / slot["strn"][val] < TV_MIN_AVG_SCORE:
        return None, None, None
    if slot["strn"].get(val, 0) < TV_MIN_SAME_READS:
        big_and_sure = (slot["strh"].get(val, 0) >= TV_BIG_PLATE_H
                        and slot["best"].get(len(val), 0.0) >= TV_SINGLE_READ_MIN_SCORE)
        if not big_and_sure:
            return None, None, None

    # vergul o'rni ham xuddi belgilar kabi ovoz bilan tanlanadi
    dots = {dp: sc for (n, dp), sc in slot["dot"].items() if n == len(val)}
    dot_pos = max(dots.items(), key=lambda kv: kv[1])[0] if dots else None
    if dot_pos is not None and dots[dot_pos] < 0.5 * sum(dots.values()):
        dot_pos = None
    return val, slot["crop"], dot_pos
# =====================================================================


def combine_wagon_image(number_text, raw_view_crop,
                         max_width=COMBINED_IMAGE_MAX_WIDTH, min_width=400,
                         extra_rows=None):
    """
    Qatorlar, hammasi bir xil kenglikda:
        [ 50889386 ]        <- model aniqlagan RAQAM (matn)
        [ kamera kropi ]    <- vagondagi ASL raqam hududi
        [ TONNA: 75.0 t ]   <- ixtiyoriy
        [ t kropi ]
        [ HAJM: 92 m3 ]     <- ixtiyoriy
        [ v kropi ]

    extra_rows = None bo'lsa -- rasm AYNAN eski (2 qatorli) ko'rinishda.
    """
    def resize_to_width(img, target_w):
        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            return img
        scale = target_w / w
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        return cv2.resize(img, (target_w, new_h), interpolation=interp)

    def text_band(text, target_w, band_h, font_scale_limit=0.90):
        band = np.full((band_h, target_w, 3), 255, dtype=np.uint8)
        if not text:
            return band
        (bw, bh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        scale_w = (target_w * font_scale_limit) / bw if bw > 0 else 1.0
        scale_h = (band_h * 0.65) / bh if bh > 0 else 1.0
        font_scale = max(0.4, min(scale_w, scale_h))
        thickness = max(2, int(round(font_scale * 1.8)))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        tx = max(0, (target_w - tw) // 2)
        ty = (band_h + th) // 2
        cv2.putText(band, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return band

    raw_view_crop = raw_view_crop.copy()
    if len(raw_view_crop.shape) == 2:
        raw_view_crop = cv2.cvtColor(raw_view_crop, cv2.COLOR_GRAY2BGR)
    target_w = min(max_width, max(raw_view_crop.shape[1], min_width))
    bottom = resize_to_width(raw_view_crop, target_w)
    top_h = max(60, bottom.shape[0])

    top = text_band(number_text, target_w, top_h)
    divider = np.full((4, target_w, 3), 255, dtype=np.uint8)
    parts = [top, divider, bottom]

    if extra_rows:
        sep = np.full((3, target_w, 3), 150, dtype=np.uint8)
        for label_text, crop_img in extra_rows:
            if crop_img is None or crop_img.size == 0:
                continue
            if len(crop_img.shape) == 2:
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
            row_img = resize_to_width(crop_img, target_w)
            band_h = max(44, min(80, row_img.shape[0]))
            parts.append(sep)
            parts.append(text_band(label_text, target_w, band_h, font_scale_limit=0.80))
            parts.append(np.full((3, target_w, 3), 255, dtype=np.uint8))
            parts.append(row_img)

    combined = np.vstack(parts)
    return cv2.copyMakeBorder(combined, 2, 2, 2, 2,
                              cv2.BORDER_CONSTANT, value=(120, 120, 120))


def filter_height_outliers(detections):
    if len(detections) < 3:
        return detections
    heights = sorted([d[3] for d in detections])
    n = len(heights)
    median_h = heights[n // 2] if n % 2 == 1 else (heights[n // 2 - 1] + heights[n // 2]) / 2
    filtered = []
    for d in detections:
        h = d[3]
        if median_h > 0 and abs(h - median_h) / median_h <= HEIGHT_OUTLIER_RATIO:
            filtered.append(d)
    return filtered


def filter_same_row(detections):
    if len(detections) < 3:
        return detections
    ys = sorted(d[4] for d in detections)
    n = len(ys)
    median_y = ys[n // 2] if n % 2 == 1 else (ys[n // 2 - 1] + ys[n // 2]) / 2
    avg_h = sum(d[3] for d in detections) / len(detections)
    tolerance = max(avg_h * ROW_Y_TOLERANCE_RATIO, 5)
    filtered = [d for d in detections if abs(d[4] - median_y) <= tolerance]
    if DEBUG_RAW_DETECTIONS and len(filtered) != len(detections):
        removed = len(detections) - len(filtered)
        print(f"[ROW-FILTER] {removed} ta boshqa qatordagi begona raqam chiqarib tashlandi")
    return filtered if filtered else detections


def cluster_digits_by_gap(detections, target_center_x, gap_ratio=DIGIT_GAP_RATIO):
    if len(detections) < 2:
        return detections
    dets = sorted(detections, key=lambda d: d[0])
    avg_h = sum(d[3] for d in dets) / len(dets)
    max_gap = max(avg_h * gap_ratio, 10)
    clusters = [[dets[0]]]
    for prev, curr in zip(dets, dets[1:]):
        gap = curr[0] - prev[0]
        if gap > max_gap:
            clusters.append([curr])
        else:
            clusters[-1].append(curr)
    if len(clusters) == 1:
        return clusters[0]

    def cluster_key(c):
        count_penalty = abs(8 - len(c))
        cx = sum(d[0] for d in c) / len(c)
        dist_penalty = abs(cx - target_center_x)
        return (count_penalty, dist_penalty)

    best_cluster = min(clusters, key=cluster_key)
    if SPLIT_NUMBER_GROUPS and len(best_cluster) != 8:
        # 8 xonali guruh yo'q -- yonma-yon ikki guruh birgalikda 8 ta bo'lsa va bir xil
        # balandlikda yozilgan bo'lsa, bu ikki guruhda yozilgan bitta raqam.
        def med_h(c):
            hs = sorted(d[3] for d in c)
            return hs[len(hs) // 2]
        pairs = [a + b for a, b in zip(clusters, clusters[1:])
                 if len(a) + len(b) == 8 and min(len(a), len(b)) >= 3
                 and abs(med_h(a) - med_h(b)) <= SPLIT_GROUP_HEIGHT_TOL * max(med_h(a), med_h(b))]
        if len(pairs) == 1:
            best_cluster = pairs[0]
    if DEBUG_RAW_DETECTIONS and len(clusters) > 1:
        dropped = sum(len(c) for c in clusters) - len(best_cluster)
        print(f"[CLUSTER] {len(clusters)} ta guruh topildi, {dropped} ta belgi begona guruh(lar)dan chiqarib tashlandi")
    return best_cluster


def check_spacing_uniform(detected_digits):
    if len(detected_digits) < 3:
        return True, 0.0
    xs = sorted(d[0] for d in detected_digits)
    gaps = [b - a for a, b in zip(xs, xs[1:]) if (b - a) > 0]
    if not gaps:
        return True, 0.0
    sorted_gaps = sorted(gaps)
    median_gap = sorted_gaps[len(sorted_gaps) // 2]
    if median_gap <= 0:
        return True, 0.0
    max_gap = max(gaps)
    ratio = max_gap / median_gap
    is_uniform = ratio <= SPACING_UNIFORMITY_MAX_RATIO
    if not is_uniform and SPLIT_NUMBER_GROUPS and len(detected_digits) == 8 and len(gaps) == 7:
        # "XXXX XXXX": yagona katta bo'shliq aynan 4- va 5-raqam orasida bo'lsa --
        # qolgan bo'shliqlar tekis bo'lishi sharti bilan -- bu ikki guruhli raqam.
        others = gaps[:3] + gaps[4:]
        om = sorted(others)[len(others) // 2]
        if om > 0 and max(others) / om <= SPACING_UNIFORMITY_MAX_RATIO and gaps[3] == max_gap:
            is_uniform, ratio = True, max(others) / om
    if DEBUG_RAW_DETECTIONS and not is_uniform:
        print(f"[SPACING-REJECT] Bo'shliqlar notekis (max/median={ratio:.2f} > {SPACING_UNIFORMITY_MAX_RATIO})")
    return is_uniform, ratio


def calculate_checksum(number_str_7):
    if len(number_str_7) != 7: return -1
    digits = [int(d) for d in number_str_7]
    weights = [2, 1, 2, 1, 2, 1, 2]
    total_sum = 0
    for d, w in zip(digits, weights):
        res = d * w
        total_sum += (res // 10) + (res % 10)
    next_ten = (total_sum + 9) // 10 * 10
    return next_ten - total_sum


def validate_wagon_number(number_str):
    if len(number_str) != 8 or not number_str.isdigit(): return False
    target = int(number_str[-1])
    calc = calculate_checksum(number_str[:7])
    return target == calc


def guess_missing_side(detected_digits, box_x1, box_x2):
    if not detected_digits:
        return None
    xs = [d[0] for d in detected_digits]
    avg_width = sum(d[3] for d in detected_digits) / len(detected_digits)
    if avg_width <= 0:
        return None
    leftmost_x = min(xs)
    rightmost_x = max(xs)
    left_gap = leftmost_x - box_x1
    right_gap = box_x2 - (rightmost_x + avg_width)
    left_is_empty = left_gap > (avg_width * MISSING_SIDE_GAP_RATIO)
    right_is_empty = right_gap > (avg_width * MISSING_SIDE_GAP_RATIO)
    if left_is_empty and not right_is_empty:
        return "left"
    if right_is_empty and not left_is_empty:
        return "right"
    return None


def repair_number(detected_digits, box_x1=None, box_x2=None):
    detected_digits.sort(key=lambda x: x[0])

    if len(detected_digits) == 7:
        side = guess_missing_side(detected_digits, box_x1, box_x2) if (box_x1 is not None and box_x2 is not None) else None

        if side == "right" and ENABLE_MISSING_LAST_DIGIT_GUESS:
            prefix_str = "".join([d[1] for d in detected_digits])
            guessed_last = calculate_checksum(prefix_str)
            if guessed_last != -1:
                candidate = prefix_str + str(guessed_last)
                return candidate, True, f"Completed missing LAST digit (geometriya): {guessed_last}"

        elif side == "left":
            rest_str = "".join([d[1] for d in detected_digits])
            known_last = rest_str[-1]
            middle6 = rest_str[:-1]
            candidates_found = []
            for d in range(10):
                trial_prefix7 = str(d) + middle6
                if str(calculate_checksum(trial_prefix7)) == known_last:
                    candidates_found.append(str(d) + rest_str)
            if len(candidates_found) == 1:
                return candidates_found[0], True, f"Completed missing FIRST digit (geometriya): {candidates_found[0][0]}"

        original_str = "".join([d[1] for d in detected_digits])
        return original_str, False, "7 digits - missing side noaniq"

    original_str = "".join([d[1] for d in detected_digits])
    if len(original_str) != 8: return original_str, False, "Len!=8"
    if validate_wagon_number(original_str): return original_str, False, "Valid"

    min_conf_val = 1.0
    min_conf_idx = -1
    for i in range(8):
        if detected_digits[i][2] < min_conf_val:
            min_conf_val = detected_digits[i][2]
            min_conf_idx = i

    if min_conf_val < MIN_CONF_TO_FIX and min_conf_idx != -1:
        candidates_found = []
        for d in range(10):
            temp = list(original_str)
            temp[min_conf_idx] = str(d)
            temp_str = "".join(temp)
            if validate_wagon_number(temp_str):
                candidates_found.append(temp_str)
        if len(candidates_found) == 1:
            fixed = candidates_found[0]
            return fixed, True, f"Fixed pos {min_conf_idx} (yagona yechim): {original_str[min_conf_idx]}->{fixed[min_conf_idx]}"

    return original_str, False, "No Fix"


def suppress_contained(predictions):
    """Boshqa (ishonchliroq) qutining ichida qolgan qutini olib tashlaydi: "7" ning
    vertikal chizig'i alohida "1" bo'lib topilishi kabi. Qo'shni haqiqiy raqamlar
    bunchalik ustma-ust tushmaydi."""
    kept = []
    for pr in sorted(predictions, key=lambda r: -r[4]):
        x1, y1, x2, y2 = pr[:4]
        area = max(1e-6, (x2 - x1) * (y2 - y1))
        inside = False
        for k in kept:
            iw = min(x2, k[2]) - max(x1, k[0])
            ih = min(y2, k[3]) - max(y1, k[1])
            if iw > 0 and ih > 0 and (iw * ih) / area >= DIGIT_CONTAIN_RATIO:
                inside = True
                break
        if not inside:
            kept.append(pr)
    return kept


def drop_extra_digit(dets):
    """9 ta raqam o'qilgan bo'lsa (chetdagi begona belgi yoki qo'shaloq quti),
    bittasini olib tashlab checksum bo'yicha tekshiradi. Faqat YAGONA to'g'ri raqam
    chiqsa qabul qilinadi: haqiqiy raqam har doim nomzodlar ichida bo'lgani uchun
    yagona yechim -- aynan u."""
    if len(dets) != 9:
        return dets
    dets = sorted(dets, key=lambda d: d[0])
    found = {}
    for i in range(9):
        rest = dets[:i] + dets[i + 1:]
        s = "".join(d[1] for d in rest)
        if validate_wagon_number(s):
            found.setdefault(s, rest)
    if len(found) == 1:
        return next(iter(found.values()))
    if len(found) > 1:
        # Bir nechta to'g'ri raqam chiqdi. Haqiqiy raqamlar bir tekis qadamda yozilgan,
        # chetdagi begona belgi esa qadamni buzadi -- qolgan 8 tasi eng tekis bo'lgan
        # variant boshqalardan aniq ustun bo'lsagina tanlanadi.
        def unevenness(rest):
            xs = [d[0] for d in rest]
            g = [b - a for a, b in zip(xs, xs[1:])]
            m = sorted(g)[len(g) // 2]
            if m <= 0 or min(g) <= 0:
                return float('inf')
            return max(max(g) / m, m / min(g))
        scored = sorted((unevenness(r), num) for num, r in found.items())
        if (scored[0][0] <= SPACING_UNIFORMITY_MAX_RATIO
                and scored[1][0] >= scored[0][0] * EXTRA_DIGIT_GEOMETRY_MARGIN):
            return found[scored[0][1]]
    return dets


def detect_digits_in_crop(classifier, crop_img, x_offset, y_offset, box_center_x, track_id=None, scale_factor=1.0):
    results_v8 = classifier.predict(crop_img, conf=DIGIT_CONF, iou=0.45,
                                    agnostic_nms=DIGIT_AGNOSTIC_NMS, verbose=False, device=dev())
    predictions = suppress_contained(results_v8[0].boxes.data.cpu().numpy())
    raw_detections = []

    for pred in predictions:
        dx1, dy1, dx2, dy2, d_conf, d_cls = pred
        dx1, dy1, dx2, dy2 = dx1 / scale_factor, dy1 / scale_factor, dx2 / scale_factor, dy2 / scale_factor
        class_id = int(d_cls)
        if hasattr(classifier, 'names'):
            names = classifier.names
            digit_str = names[class_id] if isinstance(names, list) else names.get(class_id, str(class_id))
        else:
            digit_str = str(class_id)
        digit_height = float(dy2 - dy1)
        digit_y_center = float(y_offset + (dy1 + dy2) / 2)
        raw_detections.append((int(x_offset + dx1), digit_str, float(d_conf), digit_height, digit_y_center))

    any_digit_seen = bool(raw_detections)

    non_digit = [d for d in raw_detections if d[1] not in DIGIT_CHARS]
    if len(non_digit) >= NON_DIGIT_DISCARD_THRESHOLD:
        if DEBUG_RAW_DETECTIONS:
            bad = ",".join(d[1] for d in non_digit)
            print(f"[LETTER-DISCARD] ID:{track_id} | {len(non_digit)} ta begona belgi ({bad})")
        raw_detections = []

    height_filtered = filter_height_outliers(raw_detections)
    row_filtered = filter_same_row(height_filtered)
    detected_full = cluster_digits_by_gap(row_filtered, box_center_x)
    detected_full = drop_extra_digit(detected_full)

    is_uniform, spacing_ratio = check_spacing_uniform(detected_full)
    if DIAGNOSTIC_LOG_ENABLED and len(detected_full) in (7, 8):
        confs = [d[2] for d in detected_full]
        avg_c = sum(confs) / len(confs)
        min_c = min(confs)
        raw_str = "".join(d[1] for d in sorted(detected_full, key=lambda x: x[0]))
        print(f"[DIAG] ID:{track_id} | str={raw_str} | n={len(detected_full)} | "
              f"spacing_ratio={spacing_ratio:.2f} | avg_conf={avg_c:.2f} | min_conf={min_c:.2f} | "
              f"uniform={'HA' if is_uniform else 'YOQ'}")

    if not is_uniform:
        return [], any_digit_seen, None

    avg_y_center = sum(d[4] for d in detected_full) / len(detected_full) if detected_full else None
    detected_digits = [(x, s, c, h) for (x, s, c, h, y) in detected_full]

    return detected_digits, any_digit_seen, avg_y_center


def make_position_votes():
    return [defaultdict(float) for _ in range(8)]


def add_position_votes(position_votes, detected_digits):
    if len(detected_digits) != 8:
        return False
    sorted_digits = sorted(detected_digits, key=lambda d: d[0])
    for i, (x, digit_str, conf, h) in enumerate(sorted_digits):
        position_votes[i][digit_str] += conf
    return True


def build_voted_number(position_votes):
    if any(len(pv) == 0 for pv in position_votes):
        return None, 0.0
    digits = []
    agreement_scores = []
    for pv in position_votes:
        total = sum(pv.values())
        best_digit, best_weight = max(pv.items(), key=lambda kv: kv[1])
        digits.append(best_digit)
        agreement_scores.append(best_weight / total if total > 0 else 0.0)
    avg_agreement = sum(agreement_scores) / len(agreement_scores)
    return "".join(digits), avg_agreement


# =====================================================================
# ====================  VAQT MANBAI (MEDIA SOATI)  ====================
# =====================================================================
# Vaqtga bog'liq BARCHA qarorlar shu soatdan o'qiydi: takror vagon to'sig'i,
# poyezd sessiyasining yakunlanishi, vagonlar orasidagi eng kichik oraliq.
#
# Nega kerak. Fayldan video o'qilganda ishlov real vaqtdan bir necha barobar
# sekin ketadi (bu mashinada ~5x). Real soatga bog'langan filtr videoda
# shuncha barobar kuchsizlanadi: 120 soniyalik takror to'sig'i videoda atigi
# ~24 soniyaga aylanadi va to'xtab turgan poyezdda bitta vagon ikki marta
# yozilib qoladi; 300 soniyalik sessiya oralig'i esa bitta poyezdni bir necha
# "poyezd"ga bo'lib yuboradi. Jonli kamerada real soat aynan to'g'ri o'lchov,
# shuning uchun u yerda hech narsa o'zgarmaydi.
class MediaClock:
    """Jonli manbada -- real soat; fayldan o'qilganda -- video vaqti."""

    def __init__(self):
        self.live = True
        self.fps = 25.0
        self.base = datetime.datetime.now()
        self.frame = 0

    def configure(self, live, fps):
        self.live = live
        self.fps = fps if (fps and fps > 0) else 25.0
        self.base = datetime.datetime.now()
        if not live and VIDEO_START_TIME:
            self.base = datetime.datetime.strptime(str(VIDEO_START_TIME).strip(), "%Y-%m-%d %H:%M:%S")
        self.frame = 0
        if live:
            print("   Vaqt manbai: real soat (jonli manba)")
        else:
            print("   Vaqt manbai: video vaqti (%.2f fps), boshlanishi %s%s" % (
                self.fps, self.base.strftime("%Y-%m-%d %H:%M:%S"),
                "" if VIDEO_START_TIME else "  (real yozilgan vaqt uchun: --vaqt \"YYYY-MM-DD HH:MM:SS\")"))

    def tick(self, frame_index):
        self.frame = frame_index

    def now(self):
        if self.live:
            return datetime.datetime.now()
        return self.base + datetime.timedelta(seconds=self.frame / self.fps)


clock = MediaClock()


def fmt_duration(sec):
    """3725 -> '1 soat 2 daq 5 s'"""
    sec = int(round(max(0, sec)))
    h, rem = divmod(sec, 3600)
    m, s_ = divmod(rem, 60)
    parts = ([f"{h} soat"] if h else []) + ([f"{m} daq"] if (m or h) else []) + [f"{s_} s"]
    return " ".join(parts)


def _table_path(date_str):
    return os.path.join(TRAINS_OUTPUT_DIR, f"{date_str}_poyezdlar_jadvali.txt")


def write_trains_table(date_str, train_number, start_dt, end_dt, n_wagons, tv_found, gap_txt):
    """Kunlik poyezdlar jadvali: har poyezd -- bitta qator (ketma-ketlik va oraliqlar bir qarashda)."""
    path = _table_path(date_str)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(f"POYEZDLAR JADVALI -- {start_dt:%Y-%m-%d}\n")
            f.write(f"{'No':>3} | {'Kelgan':<19} | {'Ketgan':<19} | {'Turgan vaqti':<16} | "
                    f"{'Vagon':>5} | {'t/v':>7} | Oldingi poyezddan keyin\n")
            f.write("-" * 110 + "\n")
        f.write(f"{train_number:>3} | {start_dt:%Y-%m-%d %H:%M:%S} | {end_dt:%Y-%m-%d %H:%M:%S} | "
                f"{fmt_duration((end_dt - start_dt).total_seconds()):<16} | {n_wagons:>5} | "
                f"{str(tv_found) + '/' + str(n_wagons):>7} | {gap_txt or '--'}\n")


def last_train_end_from_tables(now):
    """Dastur qayta ishga tushsa ham oraliq to'g'ri bo'lsin: bugungi/kechagi jadvaldan oxirgi ketish vaqti."""
    best = None
    for d in (now, now - datetime.timedelta(days=1)):
        path = _table_path(d.strftime("%Y.%m.%d"))
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 3 and cols[0].isdigit():
                try:
                    t = datetime.datetime.strptime(cols[2], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                if t <= now and (best is None or t > best):
                    best = t
    return best


class TrainSessionManager:
    def __init__(self):
        self.session_start = None
        self.wagons = []
        self.last_number_time = {}
        self.last_event_time = None
        self.last_accept_time = None
        self.total_tv_found = 0
        self.first_activity = None      # poyezdning birinchi vagoni kamera oldida ko'ringan vaqt
        self.last_activity = None       # oxirgi marta vagon raqami ko'ringan vaqt
        self.prev_train_end = None      # oldingi poyezd ketgan vaqt (oraliqni hisoblash uchun)

    def activity(self, now):
        """Kadrda vagon raqami ko'rindi: poyezd shu yerda (to'xtab turgan bo'lsa ham)."""
        if self.first_activity is None:
            self.first_activity = now
        self.last_activity = now

    def add_wagon(self, number_str, rel_y=None, image=None, tv_info=None):
        # takror trek birlashtirilsa -- qaysi yozuvga borgani (kech qiymat uchun)
        self.last_merge_index = None
        if number_str in KNOWN_FALSE_POSITIVE_NUMBERS:
            print(f"       ↳ [BLOCKLIST-SKIP] {number_str} -- ma'lum lokomotiv/begona yozuv, hisoblanmadi")
            self.last_event_time = clock.now()
            return None

        now = clock.now()

        # Bitta poyezdda bir xil vagon raqami IKKI MARTA bo'lishi mumkin emas.
        # Ilgari bu vaqt bo'yicha tekshirilardi ("120 soniya ichida takrorlanmasin")
        # va to'xtab turgan poyezdda ishlamay qolardi: treker bitta vagonga bir
        # nechta ID ochadi (qutilarning IoU si ~0.98), uzun va qisqa treklar esa
        # to'siq muddatidan uzoqroq farq bilan yakunlanib, vagon ikki marta
        # yozilardi. Endi tekshiruv vaqtga emas, SESSIYA RO'YXATIGA tayanadi.
        existing_idx = None
        for i in range(len(self.wagons) - 1, -1, -1):
            if self.wagons[i][0] == number_str:
                existing_idx = i
                break

        if existing_idx is not None:
            # Raqam qisqa trekda tasdiqlanib, t/m3 qiymatlari uzunrog'ida to'planib
            # qolishi mumkin -- yetishmayotgan qiymatlar avvalgi yozuvga qo'shiladi.
            num, ts, ry, img, tv = self.wagons[existing_idx]
            merged = []
            new_tv = dict(tv)
            info = tv_info or {}
            if new_tv.get("_numcrop") is None and info.get("_numcrop") is not None:
                new_tv["_numcrop"] = info["_numcrop"]
            for k in ("t", "v"):
                v, sup = info.get(k), info.get("_n_" + k, 0)
                # yetishmasa -- qo'shiladi; bor bo'lsa -- faqat ko'proq o'qish bilan tasdiqlangan bo'lsa almashadi
                if v and (not new_tv.get(k) or (v != new_tv.get(k) and sup > new_tv.get("_n_" + k, 0))):
                    old = new_tv.get(k)
                    new_tv[k], new_tv["_n_" + k] = v, sup
                    if info.get("_crop_" + k) is not None:
                        new_tv["_crop_" + k] = info["_crop_" + k]
                    merged.append(f"{'TONNA' if k == 't' else 'HAJM'} {v}" + (f" (oldin {old})" if old else ""))
            if merged:
                image = self._rebuild_image(num, image if image is not None else img, new_tv)
            self.last_merge_index = existing_idx
            if merged or (img is None and image is not None):
                self.wagons[existing_idx] = (num, ts, ry if ry is not None else rel_y,
                                             image if (merged and image is not None) else img,
                                             new_tv)
            if merged:
                print(f"       ↳ [BIRLASHTIRILDI] {number_str} -- takror trekdan olindi: "
                      f"{'  '.join(merged)}")
            else:
                print(f"       ↳ [SKIP] {number_str} -- shu poyezdda "
                      f"{(now - ts).total_seconds():.0f}s oldin yozilgan, takror hisoblanmadi")
            self.last_event_time = now
            return None

        # Sessiya chegarasidan o'tgan takror (masalan manyovr qilib qaytgan vagon):
        # bu yerda vaqt to'sig'i o'z kuchida qoladi, endi u media soati bo'yicha.
        last_time = self.last_number_time.get(number_str)
        if last_time is not None and (now - last_time).total_seconds() < DUPLICATE_NUMBER_COOLDOWN_SECONDS:
            print(f"       ↳ [SKIP] {number_str} -- oldingi sessiyada "
                  f"{(now - last_time).total_seconds():.0f}s oldin yozilgan, takror hisoblanmadi")
            self.last_event_time = now
            return None
        if self.session_start is None:
            self.session_start = now
            print(f"\n🚂 YANGI POYEZD SESSIYASI BOSHLANDI: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        self.wagons.append((number_str, now, rel_y, image, tv_info or {}))
        self.last_number_time[number_str] = now
        self.last_event_time = now
        self.last_accept_time = now
        return len(self.wagons) - 1          # keyin qiymat qo'shish uchun yozuv indeksi

    def _rebuild_image(self, num, image, tv):
        """Hisobot rasmini joriy t/v qiymatlari bilan qayta chizadi (qiymat yangilanganda)."""
        if not SAVE_WAGON_IMAGES or tv.get("_numcrop") is None:
            return image
        rows = [(f"{title}: {tv[k]} {unit}", tv["_crop_" + k])
                for k, title, unit in (("t", "TONNA", "t"), ("v", "HAJM", "m3"))
                if tv.get(k) and tv.get("_crop_" + k) is not None]
        if not rows:
            return image
        try:
            return combine_wagon_image(num, tv["_numcrop"], extra_rows=rows)
        except Exception:
            return image

    def update_wagon_value(self, idx, kind, value, support=0, crop=None):
        """Vagon hisobotga yozilgandan KEYIN kelgan t/m3 qiymatini qo'shadi.
        Plastinka raqamdan o'ngroqda joylashgani uchun u ko'pincha vagon raqami
        kadrdan chiqib, trek yopilgandan keyin o'qiladi."""
        if not (0 <= idx < len(self.wagons)):
            return False
        num, ts, rel_y, image, tv = self.wagons[idx]
        old, old_sup = tv.get(kind), tv.get("_n_" + kind, 0)
        if old == value:
            if support > old_sup:
                tv = dict(tv)
                tv["_n_" + kind] = support
                self.wagons[idx] = (num, ts, rel_y, image, tv)
            return False
        if old and support <= old_sup:
            return False                     # mavjud qiymat kuchliroq tasdiqlangan
        tv = dict(tv)
        tv[kind], tv["_n_" + kind] = value, support
        if crop is not None:
            tv["_crop_" + kind] = crop
        image = self._rebuild_image(num, image, tv)
        self.wagons[idx] = (num, ts, rel_y, image, tv)
        what = f"{'TONNA' if kind == 't' else 'HAJM'} {value}"
        if old:
            print(f"       ↳ [YANGILANDI] {num} -- {what} (oldin {old}; {support} ta o'qish bilan tasdiqlandi)")
        else:
            print(f"       ↳ [KEYIN QO'SHILDI] {num} -- {what}")
        return True

    def should_close(self):
        marks = [t for t in (self.last_event_time, self.last_activity) if t is not None]
        if not marks:
            return False
        elapsed = (clock.now() - max(marks)).total_seconds()
        if self.session_start is None:
            # vagon yozilmagan faollik (lokomotiv, begona yozuv) -- uzoq vaqt o'tsa unutiladi
            if elapsed > TRAIN_SESSION_GAP_SECONDS:
                self.first_activity = self.last_activity = None
            return False
        # Poyezd kamera oldida turgan bo'lsa (raqam ko'rinib tursa) sessiya yopilmaydi -- yuk
        # tushirish uchun uzoq turgan poyezd ikkita "poyezd" bo'lib qolmasin.
        if (self.last_event_time is not None and
                (clock.now() - self.last_event_time).total_seconds() > TRAIN_MAX_STOP_SECONDS):
            return True
        return elapsed > TRAIN_SESSION_GAP_SECONDS

    def close(self):
        if self.session_start is None or not self.wagons:
            self._reset()
            return
        fa = self.first_activity
        if fa is not None and (self.session_start - fa).total_seconds() > TRAIN_SESSION_GAP_SECONDS:
            fa = None                     # bu faollik oldingi (qoldirilgan) vagondan -- kelish vaqti emas
        start_dt = min(t for t in (self.session_start, fa) if t is not None)
        last_seen = self.last_activity
        cap = self.wagons[-1][1] + datetime.timedelta(seconds=TRAIN_MAX_STOP_SECONDS)
        if last_seen is not None and last_seen > cap:
            last_seen = cap
        end_dt = max(t for t in (self.wagons[-1][1], last_seen) if t is not None)
        if self.prev_train_end is None:
            self.prev_train_end = last_train_end_from_tables(start_dt)
        gap_txt = (fmt_duration((start_dt - self.prev_train_end).total_seconds())
                   if self.prev_train_end is not None and self.prev_train_end <= start_dt else None)

        final_wagons = list(self.wagons)
        removed_as_outlier = []

        if ENABLE_POSITION_OUTLIER_FILTER:
            wagons_with_pos = [w for w in self.wagons if w[2] is not None]
            if len(wagons_with_pos) >= MIN_WAGONS_FOR_POSITION_MEDIAN:
                rel_ys = sorted(w[2] for w in wagons_with_pos)
                n = len(rel_ys)
                median_rel_y = rel_ys[n // 2] if n % 2 == 1 else (rel_ys[n // 2 - 1] + rel_ys[n // 2]) / 2
                kept = []
                for w in self.wagons:
                    number_str, ts, rel_y, image, tv_info = w
                    if rel_y is not None and abs(rel_y - median_rel_y) > POSITION_OUTLIER_THRESHOLD:
                        removed_as_outlier.append((number_str, ts, rel_y, median_rel_y))
                    else:
                        kept.append(w)
                final_wagons = kept
                if removed_as_outlier:
                    print(f"\n📍 POZITSIYA-OUTLIER FILTRI: median joylashuv={median_rel_y:.2f} "
                          f"(0=tepa,1=tag), {len(removed_as_outlier)} ta yozuv chetga chiqarib tashlandi:")
                    for number_str, ts, rel_y, med in removed_as_outlier:
                        print(f"     ✗ {number_str} (pozitsiya={rel_y:.2f}, mediandan farq={abs(rel_y-med):.2f}) "
                              f"[{ts.strftime('%H:%M:%S')}]")

        os.makedirs(TRAINS_OUTPUT_DIR, exist_ok=True)
        date_str = start_dt.strftime('%Y.%m.%d')
        filename = f"{date_str}_barcha_natijalar.txt"
        filepath = os.path.join(TRAINS_OUTPUT_DIR, filename)
        start_time_str = start_dt.strftime('%H.%M.%S')
        end_time_str = end_dt.strftime('%H.%M.%S')
        train_number = 1
        if os.path.exists(filepath):
            existing_content = open(filepath, encoding='utf-8').read()
            train_number = existing_content.count("-POYEZD") + 1

        tv_found = sum(1 for w in final_wagons if w[4].get("t") or w[4].get("v"))
        self.total_tv_found += tv_found

        with open(filepath, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"{train_number}-POYEZD\n")
            f.write(f"Video manba          : {VIDEO_SOURCE}\n")
            f.write(f"Poyezd kelgan vaqt    : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Poyezd ketgan vaqt    : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Oraliq                : {start_time_str} - {end_time_str}\n")
            f.write(f"Turgan vaqti          : {fmt_duration((end_dt - start_dt).total_seconds())}\n")
            if gap_txt:
                f.write(f"Oldingi poyezddan keyin: {gap_txt} "
                        f"(oldingisi {self.prev_train_end.strftime('%Y-%m-%d %H:%M:%S')} da ketgan)\n")
            else:
                f.write("Oldingi poyezddan keyin: -- (birinchi qayd etilgan poyezd)\n")
            f.write(f"Jami vagonlar (barchasi VALID) : {len(final_wagons)}\n")
            if TV_ENABLED:
                f.write(f"t/v o'qilgan vagonlar          : {tv_found}/{len(final_wagons)}\n")
            if removed_as_outlier:
                f.write(f"Pozitsiya-outlier (chetga chiqarilgan): {len(removed_as_outlier)}\n")
            f.write("-" * 70 + "\n")
            for i, (number, ts, rel_y, image, tv_info) in enumerate(final_wagons, 1):
                t_val = tv_info.get("t")
                v_val = tv_info.get("v")
                t_txt = f"{t_val} t" if t_val else "--"
                v_txt = f"{v_val} m3" if v_val else "--"
                if TV_ENABLED:
                    f.write(f"{i:>3}) {number}   [VALID]   TONNA: {t_txt:>10}   "
                            f"HAJM: {v_txt:>10}   {ts.strftime('%H:%M:%S')}\n")
                else:
                    f.write(f"{i:>3}) {number}   [VALID]   {ts.strftime('%H:%M:%S')}\n")
            if removed_as_outlier:
                f.write("-" * 70 + "\n")
                f.write("Chetga chiqarilgan (ehtimol begona yozuv, pozitsiya mos kelmadi):\n")
                for number, ts, rel_y, med in removed_as_outlier:
                    f.write(f"     {number}   pozitsiya={rel_y:.2f} (median={med:.2f})   {ts.strftime('%H:%M:%S')}\n")
        write_trains_table(date_str, train_number, start_dt, end_dt, len(final_wagons),
                           tv_found, gap_txt)
        print(f"\n🚂 {train_number}-POYEZD: kelgan {start_dt:%H:%M:%S}, ketgan {end_dt:%H:%M:%S}, "
              f"turgan vaqti {fmt_duration((end_dt - start_dt).total_seconds())}, {len(final_wagons)} vagon"
              + (f", oldingi poyezddan {gap_txt} keyin" if gap_txt else ""))
        self.prev_train_end = end_dt
        print(f"💾 Poyezd hisoboti qo'shildi (append): {filepath}")
        print(f"   Jami {len(final_wagons)} ta vagon (barchasi VALID)")
        if TV_ENABLED:
            print(f"   t/v o'qilgan: {tv_found}/{len(final_wagons)}\n")

        if SAVE_WAGON_IMAGES and final_wagons:
            images_root = os.path.join(TRAINS_OUTPUT_DIR, WAGON_IMAGES_SUBDIR)
            train_folder_name = f"{date_str}_{train_number}-poyezd"
            train_folder = os.path.join(images_root, train_folder_name)
            os.makedirs(train_folder, exist_ok=True)
            saved_count = 0
            for i, (number, ts, rel_y, image, tv_info) in enumerate(final_wagons, 1):
                if image is None or image.size == 0:
                    continue
                img_filename = f"{i:03d}_{number}_{ts.strftime('%H.%M.%S')}.jpg"
                cv2.imwrite(os.path.join(train_folder, img_filename), image)
                saved_count += 1
            print(f"🖼️  Vagon rasmlari saqlandi: {train_folder} ({saved_count}/{len(final_wagons)} ta)\n")

        self._reset()

    def _reset(self):
        cleanup_old_results()
        self.first_activity = None
        self.last_activity = None
        self.last_merge_index = None
        self.session_start = None
        self.wagons = []
        # last_number_time TOZALANMAYDI: sessiya chegarasidan sakrab o'tgan takrorni
        # to'sish uchun kerak. 24/7 ishlaganda cheksiz o'smasligi uchun to'siq
        # muddatidan eskirgan yozuvlar olib tashlanadi.
        if self.last_number_time:
            edge = clock.now() - datetime.timedelta(seconds=DUPLICATE_NUMBER_COOLDOWN_SECONDS)
            self.last_number_time = {k: v for k, v in self.last_number_time.items() if v > edge}
        self.last_event_time = None
        self.last_accept_time = None


train_session = TrainSessionManager()


def process_finished_track(track_id, scores_dict, candidate_frames, position_votes, position_vote_frames,
                            lifespan, stats, any_digit_seen, rel_positions, session_ordinal,
                            best_image=None, raw_number_crop=None, tv_votes=None):
    if lifespan < MIN_FRAMES_TO_SURVIVE:
        return

    if session_ordinal is not None and session_ordinal < IGNORE_FIRST_N_TRACKS_PER_SESSION:
        # Ilgari bu yerda jimgina `return` bor edi va yo'qotilgan vagon
        # hech qayerda ko'rinmasdi. Endi har bir tashlab yuborish yoziladi.
        print(f"       ↳ [LOKOMOTIV-SKIP] ID:{track_id} -- sessiyaning "
              f"{session_ordinal + 1}-treki, hisobga olinmadi")
        return

    winner = None
    source_tag = ""

    if position_vote_frames >= POSITION_VOTE_MIN_FRAMES_STRONG:
        voted_number, avg_agreement = build_voted_number(position_votes)
        if voted_number and validate_wagon_number(voted_number):
            winner = voted_number
            source_tag = f" [POSITION-VOTE x{position_vote_frames}, kelishuv={avg_agreement:.0%}]"

    if winner is None:
        if not scores_dict:
            return

        min_frames = max(FAST_MIN_READ_FRAMES, int(FALLBACK_MIN_SHARE * position_vote_frames))
        valid_repeated_candidates = {
            num: sc for num, sc in scores_dict.items()
            if validate_wagon_number(num) and candidate_frames.get(num, 0) >= min_frames
        }
        if not valid_repeated_candidates:
            return

        sorted_candidates = sorted(valid_repeated_candidates.items(), key=lambda item: item[1], reverse=True)
        winner, _ = sorted_candidates[0]
        source_tag = f" [repair-fallback x{candidate_frames[winner]}]"

    if winner is None:
        return

    if train_session.last_accept_time is not None:
        elapsed_since_last = (clock.now() - train_session.last_accept_time).total_seconds()
        if elapsed_since_last < MIN_INTER_WAGON_SECONDS:
            is_strong_position_vote = position_vote_frames >= FRAGMENTATION_STRONG_VOTE_MIN
            if not is_strong_position_vote:
                print(f"       ↳ [FRAGMENTATION-REJECT] ID:{track_id} | {winner} -- oldingi vagondan "
                      f"{elapsed_since_last:.1f}s dan keyin keldi (juda tez, ehtimol trek bo'linib ketgan "
                      f"bitta vagon) -- yetarli kuchli tasdiq yo'q, rad etildi")
                return

    median_rel_y = None
    if rel_positions:
        median_rel_y = sorted(rel_positions)[len(rel_positions) // 2]

    # ---- shu vagon uchun t/v yakuniy qiymatlari (chegarasiz) ----
    tv_info = {}
    extra_rows = []
    if TV_ENABLED and tv_votes:
        for kind, title, forced_unit in (("t", "TONNA", "t"), ("v", "HAJM", "m3")):
            val, crop, dot_pos = tv_pick_winner(tv_votes.get(kind))
            if val:
                support = tv_votes[kind]["strn"].get(val, 0)
                val = tv_apply_decimal(val, kind, dot_pos)
                if tv_plausible(val, kind):
                    tv_info[kind] = val
                    tv_info["_n_" + kind] = support      # qiymatni tasdiqlagan o'qishlar soni
                    tv_info["_crop_" + kind] = crop
                    extra_rows.append((f"{title}: {val} {forced_unit}", crop))
    if raw_number_crop is not None and raw_number_crop.size > 0:
        tv_info["_numcrop"] = raw_number_crop

    final_image = best_image
    if SAVE_WAGON_IMAGES and extra_rows and raw_number_crop is not None and raw_number_crop.size > 0:
        try:
            final_image = combine_wagon_image(winner, raw_number_crop, extra_rows=extra_rows)
        except Exception:
            final_image = best_image

    wagon_idx = train_session.add_wagon(winner, rel_y=median_rel_y,
                                        image=final_image, tv_info=tv_info)

    if wagon_idx is not None:
        tv_txt = ""
        if tv_info:
            parts = []
            if tv_info.get("t"):
                parts.append(f"TONNA {tv_info['t']}")
            if tv_info.get("v"):
                parts.append(f"HAJM {tv_info['v']}")
            tv_txt = (" | " + "  ".join(parts)) if parts else ""
        print(f"\n[EVENT] 🚆 Wagon ID:{track_id} | Number: {winner} | ✅ VALID{source_tag}{tv_txt}")
        stats['read_success'] += 1
        stats['read_valid'] += 1
        print("-" * 50)
    # takror trek bo'lsa ham, keyin kelgan qiymat o'sha yozuvga borsin
    return wagon_idx if wagon_idx is not None else getattr(train_session, "last_merge_index", None)


def setup_device():
    """Qurilmani tanlaydi va unga moslashadi. Qaytaradi: "cuda" yoki "cpu"."""
    global DEVICE, TV_DET_OPENVINO_DEVICE
    want = str(DEVICE).lower()
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  GPU so'ralgan, lekin CUDA mavjud emas -- CPU ishlatiladi")
        want = "cpu"
    DEVICE = "cuda" if want.startswith("cuda") else "cpu"
    if DEVICE == "cuda":
        try:
            # GPU haqiqatan ishlayaptimi (drayver xatosi, xotira yetmasligi) -- aks holda CPU ga o'tiladi
            _x = torch.randn(256, 256, device="cuda")
            assert torch.isfinite(_x @ _x).all()
            del _x
        except Exception as e:
            print(f"⚠️  GPU ishga tushmadi ({e}) -- CPU ga o'tildi")
            DEVICE = "cpu"
    if DEVICE == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        print(f"⚙️  Qurilma: GPU -- {name} ({mem:.0f} GB)")
    if DEVICE == "cpu":
        n = os.cpu_count() or 4
        torch.set_num_threads(n)          # ba'zi tizimlarda standart 1 ta ip bo'ladi
        cv2.setNumThreads(n)
        print(f"⚙️  Qurilma: CPU ({n} yadro, {platform.machine()}) -- GPU topilmadi. "
              f"Video fayl har bir kadri bilan tahlil qilinadi (aniqlik saqlanadi), lekin sekinroq.")
        if (TV_DET_OPENVINO_DEVICE is None and platform.machine().lower() in ("x86_64", "amd64")
                and importlib.util.find_spec("openvino") is not None):
            # Intel/AMD protsessorda eng og'ir model (t/v detektori) OpenVINO bilan 2-3 barobar tez.
            TV_DET_OPENVINO_DEVICE = "intel:cpu"
    return DEVICE


def dev():
    """predict(device=...) uchun: tanlangan qurilma (ultralytics o'zi tanlasa, CPU so'ralganda ham GPU oladi)."""
    return "cpu" if DEVICE == "cpu" else 0


def is_live_source(src):
    """Kamera indeksi, /dev/video*, RTSP/HTTP oqimi -- jonli; qolgani -- video fayl."""
    if isinstance(src, int) or FORCE_LIVE:
        return True
    s = str(src).strip().lower()
    return s.isdigit() or s.startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://",
                                        "udp://", "tcp://", "/dev/video"))


def mask_source(src):
    """Jurnalga parol chiqmasin: rtsp://admin:parol@ip -> rtsp://admin:***@ip"""
    s = str(src)
    if "://" in s and "@" in s:
        head, rest = s.split("://", 1)
        cred, host = rest.rsplit("@", 1)
        if ":" in cred:
            return f"{head}://{cred.split(':', 1)[0]}:***@{host}"
    return s


_ACTIVE_CAPTURES = []


class LiveCapture:
    """
    Jonli manba (IP-kamera, RTSP, USB-kamera) uchun ishonchli o'quvchi.

    * Kadrlar alohida oqimda uzluksiz o'qiladi, read() har doim ENG YANGI kadrni beradi:
      ishlov bir lahza sekinlashsa ham kechikish to'planmaydi (eski kadrlar tashlanadi).
    * Oqim uzilsa yoki qotsa -- o'zi qayta ulanadi; read() hech qachon "tugadi" demaydi.
    * Sinov uchun video faylni ham kamera kabi o'qiy oladi (--jonli): o'z tezligida,
      oxiriga yetganda "uzilish" deb qayta ochadi.
    """

    def __init__(self, source):
        self.source = int(source) if str(source).strip().isdigit() else source
        self.is_file = isinstance(self.source, str) and os.path.isfile(self.source)
        self.lock = threading.Condition()
        self.frame, self.seq, self.read_seq = None, 0, 0
        self.last_frame_time = time.time()
        self.reconnects = 0
        self.running = True
        self.cap = None
        self.width = self.height = 0
        self.fps = 25.0
        self.generation = 0
        _ACTIVE_CAPTURES.append(self)
        self._connect(first=True)
        self._start_thread()

    def _open(self):
        if isinstance(self.source, int):
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
            cap = cv2.VideoCapture(self.source, backend)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        if not self.is_file:
            # RTSP TCP orqali -- UDP da paketlar yo'qolib, kadrlar buziladi.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        params = []
        for name, val in (("CAP_PROP_OPEN_TIMEOUT_MSEC", LIVE_OPEN_TIMEOUT_MS),
                          ("CAP_PROP_READ_TIMEOUT_MSEC", LIVE_READ_TIMEOUT_MS)):
            if hasattr(cv2, name):
                params += [getattr(cv2, name), val]
        try:
            return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)
        except Exception:
            return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

    def _connect(self, first=False):
        wait = 1
        while self.running:
            try:
                cap = self._open()
                if cap is not None and cap.isOpened():
                    ok, f = cap.read()
                    if ok and f is not None:
                        self.cap = cap
                        self.height, self.width = f.shape[:2]
                        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        if not (1 <= self.fps <= 240):
                            self.fps = 25.0
                        with self.lock:
                            self.frame = f
                            self.seq += 1
                            self.last_frame_time = time.time()
                            self.lock.notify_all()
                        if not first:
                            print(f"✅ Manbaga qayta ulandi ({self.reconnects}-marta): {mask_source(self.source)}")
                        return True
                    cap.release()
            except Exception as e:
                print(f"⚠️  Manbani ochishda xato: {e}")
            print(f"⚠️  Manbaga ulanib bo'lmadi ({mask_source(self.source)}) -- {wait} s dan keyin qayta urinish")
            time.sleep(wait)
            wait = min(wait * 2, LIVE_RECONNECT_MAX_WAIT)
        return False

    def _start_thread(self):
        self.generation += 1
        th = threading.Thread(target=self._loop, args=(self.generation,), daemon=True)
        th.start()

    def _loop(self, gen):
        period = 1.0 / self.fps if self.is_file else 0.0
        next_t = time.perf_counter()
        while self.running and gen == self.generation:
            try:
                ok, f = self.cap.read()
            except Exception:
                ok, f = False, None
            if gen != self.generation:
                return                      # qotgan eski oqim -- o'rniga yangisi ishlayapti
            if ok and f is not None:
                with self.lock:
                    self.frame = f
                    self.seq += 1
                    self.last_frame_time = time.time()
                    self.lock.notify_all()
                if period:
                    next_t += period
                    time.sleep(max(0.0, next_t - time.perf_counter()))
                continue
            if not self.running:
                return
            print("⚠️  Oqim uzildi -- qayta ulanmoqda...")
            try:
                self.cap.release()
            except Exception:
                pass
            self.reconnects += 1
            self._connect()
            next_t = time.perf_counter()

    def read(self):
        """Yangi kadr kelguncha kutadi va uni qaytaradi. Oqim qotib qolsa -- qayta ulaydi."""
        with self.lock:
            while self.running and self.seq == self.read_seq:
                self.lock.wait(timeout=1.0)
                if self.running and time.time() - self.last_frame_time > LIVE_STALL_SECONDS:
                    print(f"⚠️  {LIVE_STALL_SECONDS} s dan beri yangi kadr yo'q -- oqim qotgan, qayta ulanmoqda")
                    self.last_frame_time = time.time()
                    self.reconnects += 1
                    threading.Thread(target=self._restart_stalled, daemon=True).start()
            if not self.running:
                return False, None
            self.read_seq = self.seq
            return True, self.frame

    def _restart_stalled(self):
        old = self.cap
        self.generation += 1                # eski (qotgan) oqim natijasi e'tiborsiz qoladi
        if self._connect():
            self._start_thread()
        try:
            old.release()
        except Exception:
            pass

    def grab(self):
        return False

    def isOpened(self):
        return self.cap is not None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0

    def set(self, *a):
        return False

    def release(self):
        self.running = False
        with self.lock:
            self.lock.notify_all()
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        if self in _ACTIVE_CAPTURES:
            _ACTIVE_CAPTURES.remove(self)


FORCE_LIVE = False
_RUN_STATE = {"quit": False}


def cleanup_old_results():
    """KEEP_DAYS dan eski natijalarni (rasm papkalari, hisobot fayllari) o'chiradi."""
    if not KEEP_DAYS or KEEP_DAYS <= 0:
        return
    edge = time.time() - KEEP_DAYS * 86400
    roots = [TRAINS_OUTPUT_DIR, os.path.join(TRAINS_OUTPUT_DIR, WAGON_IMAGES_SUBDIR)]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            path = os.path.join(root, name)
            try:
                if path == roots[1] or os.path.getmtime(path) > edge:
                    continue
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif name.endswith(("_barcha_natijalar.txt", "_poyezdlar_jadvali.txt")):
                    os.remove(path)
            except OSError:
                pass


def run_realtime():
    setup_device()
    print("📡 Loading detector model (YOLOv8)...")
    try:
        detector = YOLO(VAGON_NUMBER_DETECTION)
    except Exception as e:
        print(f"Error loading detector: {e}")
        return

    print("⏳ Loading digit detector (YOLOv8)...")
    try:
        classifier = YOLO(VAGON_NUMBER_CLASSIFICATION)
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return

    tv_detector = tv_classifier = tv_det_device = None
    tv_active = False
    if TV_ENABLED:
        try:
            print("🔤 Loading t/v plate detector...")
            tv_detector, tv_det_device = load_tv_detector()
            print("🔤 Loading t/v character reader...")
            tv_classifier = YOLO(TV_CLASSIFICATION)
            tv_active = True
            print(f"   t/v detektor klasslari : {tv_detector.names}")
            print(f"   t/v belgi klasslari    : {tv_classifier.names}")
        except Exception as e:
            print(f"⚠️  t/v modellari yuklanmadi ({e}) -- t/v o'qish O'CHIRILDI, "
                  f"raqam aniqlash odatdagidek davom etadi.")
            tv_active = False

    gui_available = check_gui_support()

    _live_source = is_live_source(VIDEO_SOURCE)
    if _live_source:
        print(f"📷 Jonli manbaga ulanmoqda: {mask_source(VIDEO_SOURCE)}")
        cap = LiveCapture(VIDEO_SOURCE)
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(VIDEO_SOURCE)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print(f"❌ Video manbasini ochib bo'lmadi: {VIDEO_SOURCE}")
        if not _live_source and not os.path.exists(str(VIDEO_SOURCE)):
            print(f"   Fayl topilmadi: {VIDEO_SOURCE}")
            print("   Ishga tushirish: python logic_N_T_V.py <video_fayl yoki kamera manzili>")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"✅ Camera started: {width}x{height} @ {fps} FPS")

    # Jonli manba (kamera, RTSP/HTTP oqimi) -- real soat; fayl -- video vaqti.
    clock.configure(_live_source, fps)

    ui_scale = ui_scale_of(width)
    list_font_scale = RECOGNIZED_LIST_FONT_SCALE * ui_scale
    list_line_height = max(11, int(RECOGNIZED_LIST_LINE_HEIGHT * ui_scale))
    list_margin_left = max(4, int(RECOGNIZED_LIST_MARGIN_LEFT * ui_scale))
    list_margin_top = max(8, int(RECOGNIZED_LIST_MARGIN_TOP * ui_scale))
    list_margin_bottom = max(4, int(RECOGNIZED_LIST_MARGIN_BOTTOM * ui_scale))
    list_margin_right = max(6, int(RECOGNIZED_LIST_MARGIN_RIGHT * ui_scale))
    list_thickness = max(1, round(1.3 * ui_scale))
    list_shadow_thickness = list_thickness + 1
    line_thickness = max(1, round(1.4 * ui_scale))      # ramkalar va zona chiziqlari

    if SAVE_RESIZE_WIDTH and SAVE_RESIZE_WIDTH < width:
        save_width = SAVE_RESIZE_WIDTH
        save_height = int(height * (SAVE_RESIZE_WIDTH / width))
    else:
        save_width, save_height = width, height

    out = None
    if SAVE_RESULT:
        os.makedirs(TRAINS_OUTPUT_DIR, exist_ok=True)
        base_name, ext = os.path.splitext(OUTPUT_FILENAME)
        timestamp_str = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
        FINAL_OUTPUT_PATH = os.path.join(TRAINS_OUTPUT_DIR, f"{timestamp_str}_{base_name}{ext}")
        fourcc_h264 = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(FINAL_OUTPUT_PATH, fourcc_h264, fps, (save_width, save_height))
        if not out.isOpened():
            print("⚠️  H.264 (avc1) kodek topilmadi, mp4v kodekka qaytilmoqda")
            out = cv2.VideoWriter(FINAL_OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (save_width, save_height))
        else:
            print("✅ H.264 (avc1) kodek bilan saqlanmoqda")
        print(f"🎥 Video shu manzilga saqlanadi: {FINAL_OUTPUT_PATH}")

    frame_center_x = width / 2
    zone_pixel_width = width * ZONE_TOLERANCE
    zone_left = int(frame_center_x - zone_pixel_width)
    zone_right = int(frame_center_x + zone_pixel_width)
    track_scores = defaultdict(lambda: defaultdict(float))
    track_candidate_frames = defaultdict(lambda: defaultdict(int))
    track_position_votes = defaultdict(make_position_votes)
    track_position_vote_frames = defaultdict(int)
    track_lifespan = defaultdict(int)
    track_any_digit_seen = defaultdict(bool)
    track_read_frames = defaultdict(int)
    track_rel_positions = defaultdict(list)
    track_session_ordinal = {}
    last_seen_frame = {}
    frame_count = 0

    track_best_conf = defaultdict(float)
    track_best_image = {}
    track_best_raw_crop = {}

    # {track_id: {"t": <ovoz idishi>, "v": <ovoz idishi>}}
    track_tv_votes = defaultdict(lambda: {"t": make_tv_slot(), "v": make_tv_slot()})
    # jonli ekranda ko'rsatish uchun
    track_tv_live = defaultdict(dict)
    track_last_box = {}          # {track_id: ([x1,y1,x2,y2], kadr raqami)}
    track_vx = {}                # {track_id: gorizontal tezlik, px/kadr}

    session_track_counter = [0]

    height_history = deque(maxlen=HEIGHT_HISTORY_MAXLEN)

    seen_stable_ids = set()
    finalized_wagon = {}         # {track_id: hisobotdagi vagon indeksi} -- kech kelgan qiymat uchun
    stats = {'read_success': 0, 'read_valid': 0}

    pace = REALTIME_PLAYBACK and not _live_source and not (DEVICE == "cpu" and CPU_PROCESS_EVERY_FRAME)
    play_start = None            # birinchi kadr ishlangan payt -- jadval shundan boshlanadi
    dropped_frames = 0
    max_late = 0.0

    hb_time, hb_frames = time.time(), 0

    print("🚀 Started. Press 'q' to exit.")
    while True:
        if _live_source and time.time() - hb_time >= HEARTBEAT_SECONDS:
            el = time.time() - hb_time
            print(f"💓 {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ishlayapti: "
                  f"{(frame_count - hb_frames) / el:.1f} kadr/s ishlandi, "
                  f"qayta ulanishlar {getattr(cap, 'reconnects', 0)}, "
                  f"joriy poyezdda {len(train_session.wagons)} vagon", flush=True)
            hb_time, hb_frames = time.time(), frame_count
        if pace and play_start is not None:
            # Orqada qolgan bo'lsak -- jonli kameradagi kabi eskirgan kadrlar tashlanadi.
            # frame_count ular uchun ham oshadi: video vaqti va trek tezligi to'g'ri qoladi.
            behind = int((time.perf_counter() - play_start) * fps) - frame_count
            while behind > 0 and cap.grab():
                frame_count += 1
                dropped_frames += 1
                behind -= 1
        success, frame = cap.read()
        if not success:
            print("⚠️ Stream interrupted or camera disconnected.")
            break
        frame_count += 1
        clock.tick(frame_count)
        # Kroplar SHU nusxadan olinadi: `frame` ga zona chiziqlari, ramkalar va yozuvlar
        # chiziladi, ular plastinka yoki raqam ustidan o'tib o'qishni buzishi mumkin.
        clean_frame = frame.copy()
        results = detector.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml",
                                 conf=WAGON_DETECTOR_CONF, agnostic_nms=WAGON_DETECTOR_AGNOSTIC_NMS,
                                 device=dev())
        result = results[0]
        cv2.line(frame, (zone_left, 0), (zone_left, height), (200, 200, 200), line_thickness)
        cv2.line(frame, (zone_right, 0), (zone_right, height), (200, 200, 200), line_thickness)

        track_ids, boxes = [], []
        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            for tid, b in zip(track_ids, boxes):
                prev = track_last_box.get(tid)
                if prev:
                    df = frame_count - prev[1]
                    if 0 < df <= 5:
                        v = ((b[0] + b[2]) - (prev[0][0] + prev[0][2])) / 2.0 / df
                        track_vx[tid] = v if tid not in track_vx else 0.6 * track_vx[tid] + 0.4 * v
                track_last_box[tid] = (list(b), frame_count)

        # ---------- t/v plastinkalari: HAR KADRDA ----------
        # Ilgari bu blok "raqam qutisi topilgan bo'lsa" ichida edi. Plastinka eng katta va
        # aniq ko'ringan paytda vagon raqami ko'pincha kadrdan chiqib ketgan bo'ladi --
        # aynan eng yaxshi kadrlar umuman qidirilmay qolardi.
        tv_assigned = {}
        if tv_active and frame_count % TV_FRAME_STRIDE == 0:
            tv_plates = tv_detect_plates(tv_detector, clean_frame, tv_det_device)
            # Chegara zonasi PLASTINKAGA qo'yiladi, raqamga emas: chekkadagi
            # plastinka kesilgan va xira bo'ladi, bundan tashqari uzoqdagi
            # plastinka qo'shni vagonning trekiga yopishib, qiymatlarni
            # aralashtirib yuboradi.
            centered = [p for p in tv_plates if zone_left < p["cx"] < zone_right]
            # Avval KADRDA KO'RINIB turgan vagonlarga biriktiriladi.
            tv_assigned = tv_assign_to_tracks(centered, boxes, track_ids)
            # Egasi topilmagan plastinkalar uchun zaxira: raqami kadrdan chiqib ketgan
            # treklarning oxirgi qutisi tezlik bo'yicha oldinga siljitiladi (poyezd bir
            # tekis harakatlanadi). Bashorat faqat ZAXIRA -- aks holda eski trekning
            # taxminiy qutisi plastinkani ko'rinib turgan vagondan tortib oladi.
            taken = {id(p) for ps in tv_assigned.values() for p in ps}
            rest = [p for p in centered if id(p) not in taken]
            if rest:
                pred_boxes, pred_ids = [], []
                for tid, (lb, lf) in track_last_box.items():
                    gap = frame_count - lf
                    if tid in track_ids or not (0 < gap <= TV_PREDICT_MAX_FRAMES) or tid not in track_vx:
                        continue
                    shift = track_vx[tid] * gap
                    pred_boxes.append([lb[0] + shift, lb[1], lb[2] + shift, lb[3]])
                    pred_ids.append(tid)
                if pred_boxes:
                    for tid, ps in tv_assign_to_tracks(rest, pred_boxes, pred_ids).items():
                        tv_assigned.setdefault(tid, []).extend(ps)
            for p in tv_plates:
                col = (255, 200, 0) if p["kind"] == "t" else (0, 200, 255)
                if p not in centered:
                    col = (110, 110, 110)
                cv2.rectangle(frame, (p["x1"], p["y1"]), (p["x2"], p["y2"]), col, line_thickness)

            for tid, plates in tv_assigned.items():
                for p in plates:
                    if p["y2"] - p["y1"] < TV_READ_MIN_H:
                        continue
                    if sum(track_tv_votes[tid][p["kind"]]["cnt"].values()) >= TV_ENOUGH_READS:
                        continue
                    tv_crop = clean_frame[p["y1"]:p["y2"], p["x1"]:p["x2"]]
                    if tv_crop.size == 0:
                        continue
                    chars, unit, cf, dot_pos = tv_read_chars(tv_classifier, tv_crop, prefer=p["kind"])
                    if TV_DEBUG and chars:
                        print(f"[TV] ID:{tid} {p['kind']} -> '{''.join(chars)}' birlik={unit} "
                              f"cls_conf={cf:.2f} det_conf={p['conf']:.2f} "
                              f"crop={tv_crop.shape[1]}x{tv_crop.shape[0]}")
                    if not chars or (TV_REQUIRE_UNIT and not unit):
                        continue
                    # Plastinka turi BELGI O'QISHIDAN olinadi: belgilar modeli birlikni
                    # 419 ta haqiqiy kesimda 99% to'g'ri aniqladi, t/v detektori esa xira
                    # kadrda adashadi -- bitta plastinka ikkala slotga tushib, vagonda
                    # "68 t / 68 m3" kabi qiymat chiqardi.
                    kind = ("t" if unit == "t" else "v") if unit else p["kind"]
                    # Haqiqiy vagonnikiga o'xshamaydigan o'qish (bir xonali "7" -- kesilgan
                    # plastinka yoki logotip, "881120" -- ikki plastinka qo'shilib ketgan)
                    # ovozga kirmaydi: aks holda u to'g'ri qiymatning ovozini bosib ketadi.
                    if not tv_plausible(tv_apply_decimal("".join(chars), kind, dot_pos), kind):
                        continue
                    tv_add_vote(track_tv_votes[tid][kind], chars, cf * p["conf"], tv_crop, dot_pos)
                    raw_val, live_crop, live_dot = tv_pick_winner(track_tv_votes[tid][kind])
                    live_val = tv_apply_decimal(raw_val, kind, live_dot) if raw_val else None
                    if live_val and tv_plausible(live_val, kind):
                        track_tv_live[tid][kind] = live_val
                        if tid in finalized_wagon:      # vagon allaqachon yozilgan bo'lsa
                            train_session.update_wagon_value(
                                finalized_wagon[tid], kind, live_val,
                                track_tv_votes[tid][kind]["strn"].get(raw_val, 0), live_crop)
                        draw_plate_label(frame, f"{live_val}{'t' if kind == 't' else 'm3'}",
                                         p["x1"], p["y1"],
                                         (255, 200, 0) if kind == "t" else (0, 200, 255),
                                         ui=ui_scale)

        if track_ids:
            for box, track_id in zip(boxes, track_ids):
                track_lifespan[track_id] += 1
                last_seen_frame[track_id] = frame_count
                x1_orig, y1_orig, x2_orig, y2_orig = map(int, box)
                box_center_x = (x1_orig + x2_orig) / 2
                box_h = y2_orig - y1_orig
                is_stable = track_lifespan[track_id] > 2

                if is_stable and track_id not in seen_stable_ids:
                    seen_stable_ids.add(track_id)
                    if track_id not in track_session_ordinal:
                        track_session_ordinal[track_id] = session_track_counter[0]
                        session_track_counter[0] += 1

                is_in_x_zone = zone_left < box_center_x < zone_right

                is_main_track = True
                median_h_debug = None
                if is_in_x_zone:
                    height_history.append(box_h)
                    if len(height_history) >= MIN_SAMPLES_FOR_ADAPTIVE:
                        sorted_heights = sorted(height_history)
                        n = len(sorted_heights)
                        median_h_debug = sorted_heights[n // 2] if n % 2 == 1 else \
                            (sorted_heights[n // 2 - 1] + sorted_heights[n // 2]) / 2
                        is_main_track = box_h >= (median_h_debug * ADAPTIVE_HEIGHT_RATIO)

                is_in_zone = is_in_x_zone and is_main_track
                color = (0, 0, 255)

                if is_in_zone:
                    color = (0, 255, 0)

                    pad_v = int(box_h * CROP_PADDING_RATIO_V)
                    pad_v = max(CROP_PADDING_MIN, min(CROP_PADDING_MAX, pad_v))
                    pad_h = int(box_h * CROP_PADDING_RATIO_H)
                    pad_h = max(4, min(CROP_PADDING_MAX, pad_h))

                    x1_pad = max(0, x1_orig - pad_h)
                    y1_pad = max(0, y1_orig - pad_v)
                    x2_pad = min(width, x2_orig + pad_h)
                    y2_pad = min(height, y2_orig + pad_v)
                    crop = clean_frame[y1_pad:y2_pad, x1_pad:x2_pad]

                    box_aspect_ratio = (x2_orig - x1_orig) / max(1, box_h)
                    is_plausible_plate_shape = box_aspect_ratio >= MIN_PLATE_ASPECT_RATIO

                    if crop.size > 0 and is_plausible_plate_shape:
                        crop_upscaled, applied_scale = maybe_upscale_crop(crop)
                        crop_sharp = sharpen_crop(crop_upscaled)

                        detected_digits, any_digit, avg_y_center = detect_digits_in_crop(
                            classifier, crop_sharp, x1_pad, y1_pad, box_center_x, track_id,
                            scale_factor=applied_scale)

                        if any_digit:
                            track_any_digit_seen[track_id] = True

                        if SPLIT_NUMBER_GROUPS and 3 <= len(detected_digits) <= 5:
                            # Raqam ikki guruhda yozilgan va detektor har yarmini alohida
                            # quti qilib topgan ("2434" | "4863"): yonidagi, bir xil
                            # balandlikdagi qutini qo'shib, birgalikda o'qiladi.
                            for ob in boxes:
                                ox1, oy1, ox2, oy2 = map(int, ob)
                                oh = oy2 - oy1
                                if (ox1, oy1, ox2, oy2) == (x1_orig, y1_orig, x2_orig, y2_orig) or oh <= 0:
                                    continue
                                if not (0.75 <= oh / max(1, box_h) <= 1.33):
                                    continue
                                if min(y2_orig, oy2) - max(y1_orig, oy1) < 0.6 * min(box_h, oh):
                                    continue
                                gap = max(ox1 - x2_orig, x1_orig - ox2)
                                if not (-0.3 * box_h <= gap <= 1.5 * box_h):
                                    continue
                                ux1, uy1 = min(x1_orig, ox1), min(y1_orig, oy1)
                                ux2, uy2 = max(x2_orig, ox2), max(y2_orig, oy2)
                                ux1p, uy1p = max(0, ux1 - pad_h), max(0, uy1 - pad_v)
                                ux2p, uy2p = min(width, ux2 + pad_h), min(height, uy2 + pad_v)
                                crop_u = clean_frame[uy1p:uy2p, ux1p:ux2p]
                                if crop_u.size == 0:
                                    continue
                                up_u, sc_u = maybe_upscale_crop(crop_u)
                                dd_u, _any_u, ay_u = detect_digits_in_crop(
                                    classifier, sharpen_crop(up_u), ux1p, uy1p, (ux1 + ux2) / 2,
                                    track_id, scale_factor=sc_u)
                                if len(dd_u) == 8:
                                    detected_digits, avg_y_center, crop = dd_u, ay_u, crop_u
                                    x1_pad, x2_pad = ux1p, ux2p
                                    break

                        if DIGIT_TRY_INVERTED and len(detected_digits) not in (8, 9):
                            dd_i, _any_i, ay_i = detect_digits_in_crop(
                                classifier, 255 - crop_sharp, x1_pad, y1_pad, box_center_x, track_id,
                                scale_factor=applied_scale)
                            if len(dd_i) in (8, 9):
                                detected_digits, avg_y_center = dd_i, ay_i

                        if len(detected_digits) >= 6:
                            train_session.activity(clock.now())   # vagon raqami ko'rinib turibdi

                        if len(detected_digits) == 9:
                            # Ortiqcha belgini geometriya hal qila olmadi -- nomzodlardan biri
                            # shu trekning boshqa kadrlarda to'plagan yetakchi raqami bo'lsa, o'sha.
                            leader, _agr = build_voted_number(track_position_votes[track_id])
                            if leader and validate_wagon_number(leader):
                                srt = sorted(detected_digits, key=lambda d: d[0])
                                for i in range(9):
                                    rest = srt[:i] + srt[i + 1:]
                                    if "".join(d[1] for d in rest) == leader:
                                        detected_digits = rest
                                        break

                        added_to_votes = add_position_votes(track_position_votes[track_id], detected_digits)
                        if added_to_votes:
                            track_position_vote_frames[track_id] += 1

                        final_number_str = None
                        is_repaired = False
                        avg_conf = 0.0

                        if detected_digits:
                            final_number_str, is_repaired, log = repair_number(
                                detected_digits, box_x1=x1_pad, box_x2=x2_pad)
                            avg_conf = sum([d[2] for d in detected_digits]) / len(detected_digits)

                            if len(final_number_str) == 8 and avg_conf > 0.5:
                                is_valid = validate_wagon_number(final_number_str)

                                if is_valid:
                                    track_read_frames[track_id] += 1
                                    track_candidate_frames[track_id][final_number_str] += 1
                                    score_boost = 3.0 * REPAIRED_SCORE_MULTIPLIER if is_repaired else 3.0
                                    text_color = (0, 255, 0)
                                    track_scores[track_id][final_number_str] += (avg_conf * score_boost)

                                    wagon_box_h = max(1, y2_orig - y1_orig)
                                    if avg_y_center is not None:
                                        rel_y = (avg_y_center - y1_orig) / wagon_box_h
                                        track_rel_positions[track_id].append(rel_y)

                                    if (SAVE_WAGON_IMAGES and avg_conf > track_best_conf[track_id]
                                            and final_number_str.isdigit() and len(final_number_str) == 8):
                                        raw_view_img = crop
                                        if raw_view_img.size > 0:
                                            track_best_conf[track_id] = avg_conf
                                            display_number = final_number_str + ("*" if is_repaired else "")
                                            track_best_image[track_id] = combine_wagon_image(
                                                display_number, raw_view_img)
                                            track_best_raw_crop[track_id] = raw_view_img.copy()

                                    if SHOW_LIVE_LABEL:
                                        label = final_number_str
                                        if is_repaired: label += "*"
                                        # t/v qiymatlari bu yerda emas -- ular
                                        # o'z plastinkasining ustiga yoziladi.
                                        font_scale = 0.62 * ui_scale
                                        thickness = max(1, round(1.4 * ui_scale))
                                        padding = max(3, int(5.7 * ui_scale))
                                        border_thickness = max(1, round(ui_scale))
                                        (text_w, text_h), baseline = cv2.getTextSize(
                                            label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
                                        text_x = x1_orig
                                        text_y = max(text_h + padding, y1_orig - 20)
                                        box_x1 = text_x - padding
                                        box_y1 = text_y - text_h - padding
                                        box_x2 = text_x + text_w + padding
                                        box_y2 = text_y + baseline + padding // 2
                                        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
                                        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), text_color, border_thickness)
                                        cv2.putText(frame, label, (text_x, text_y),
                                                    cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

                if is_stable:
                    cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, line_thickness)
                    # ID raqam yorlig'idan TEPADA turadi: ilgari ular ustma-ust tushib,
                    # o'qilmay qolardi.
                    id_txt = f"ID:{track_id}"
                    id_scale = 0.38 * ui_scale
                    id_thick = max(1, round(ui_scale))
                    (iw, ih), ib = cv2.getTextSize(id_txt, cv2.FONT_HERSHEY_SIMPLEX, id_scale, id_thick)
                    iy = max(ih + 4, y1_orig - int(37 * ui_scale))
                    cv2.rectangle(frame, (x1_orig - 4, iy - ih - 4), (x1_orig + iw + 4, iy + ib + 2),
                                  (0, 0, 0), -1)
                    cv2.putText(frame, id_txt, (x1_orig, iy),
                                cv2.FONT_HERSHEY_SIMPLEX, id_scale, color, id_thick, cv2.LINE_AA)

        dead_tracks = []
        for tid, last_frame in last_seen_frame.items():
            if frame_count - last_frame > FRAMES_TO_FORGET:
                _idx = process_finished_track(tid, track_scores.get(tid), track_candidate_frames.get(tid, {}),
                                        track_position_votes.get(tid),
                                        track_position_vote_frames.get(tid, 0),
                                        track_lifespan.get(tid), stats,
                                        track_any_digit_seen.get(tid, False),
                                        track_rel_positions.get(tid), track_session_ordinal.get(tid),
                                        best_image=track_best_image.get(tid),
                                        raw_number_crop=track_best_raw_crop.get(tid),
                                        tv_votes=track_tv_votes.get(tid))
                if _idx is not None:
                    finalized_wagon[tid] = _idx
                dead_tracks.append(tid)

        for tid in dead_tracks:
            seen_stable_ids.discard(tid)
            for d in (last_seen_frame, track_scores, track_candidate_frames, track_rel_positions,
                      track_position_votes, track_position_vote_frames, track_lifespan,
                      track_any_digit_seen, track_read_frames, track_session_ordinal,
                      track_best_conf, track_best_image, track_best_raw_crop):
                if tid in d:
                    del d[tid]
        # track_last_box / track_vx / track_tv_votes saqlanadi: plastinka vagon raqami
        # kadrdan chiqqandan keyin o'qilishi mumkin. Eskirganlari tozalanadi.
        for tid in [t for t, (_b, f0) in track_last_box.items()
                    if frame_count - f0 > TV_LATE_VALUE_FRAMES]:
            for d in (track_last_box, track_vx, track_tv_votes, track_tv_live, finalized_wagon):
                if tid in d:
                    del d[tid]

        if train_session.should_close():
            train_session.close()
            height_history.clear()
            session_track_counter[0] = 0

        if SHOW_RECOGNIZED_LIST and train_session.wagons:
            available_height = height - list_margin_top - list_margin_bottom
            max_rows_that_fit = max(1, available_height // list_line_height)
            visible_wagons = train_session.wagons[-max_rows_that_fit:]
            # Ustunlar tekis tursin: kengliklar eng uzun bo'lishi mumkin qiymatlardan
            # olinadi (raqam 8 xona, tonna "125.5t", hajm "138.5m3").
            font = cv2.FONT_HERSHEY_SIMPLEX
            w_num = cv2.getTextSize("00000000", font, list_font_scale, list_thickness)[0][0]
            w_t = cv2.getTextSize("125.5t", font, list_font_scale, list_thickness)[0][0]
            w_v = cv2.getTextSize("138.5m3", font, list_font_scale, list_thickness)[0][0]
            col_gap = max(10, int(12 * ui_scale))
            block_w = w_num + col_gap + w_t + col_gap + w_v
            block_x = (width - block_w - list_margin_right
                       if RECOGNIZED_LIST_ON_RIGHT else list_margin_left)
            (_, th), tb = cv2.getTextSize("0", font, list_font_scale, list_thickness)
            if visible_wagons:                  # butun ro'yxat ortiga bitta fon
                y0 = list_margin_top - th - 6
                y1 = list_margin_top + (len(visible_wagons) - 1) * list_line_height + tb + 6
                bg = frame[max(0, y0):min(height, y1), max(0, block_x - 10):min(width, block_x + block_w + 10)]
                if bg.size:
                    bg[:] = (bg * RECOGNIZED_LIST_DIM).astype(bg.dtype)
            for idx, wagon_rec in enumerate(visible_wagons):
                number = wagon_rec[0]
                tv_info = wagon_rec[4] if len(wagon_rec) > 4 else {}
                t_txt = f"{tv_info['t']}t" if tv_info.get("t") else ""
                v_txt = f"{tv_info['v']}m3" if tv_info.get("v") else ""
                list_y = list_margin_top + idx * list_line_height
                for text, x, w_col in ((number, block_x, w_num),
                                       (t_txt, block_x + w_num + col_gap, w_t),
                                       (v_txt, block_x + w_num + col_gap + w_t + col_gap, w_v)):
                    if not text:
                        continue
                    tw = cv2.getTextSize(text, font, list_font_scale, list_thickness)[0][0]
                    tx = x + w_col - tw          # ustun ichida o'ngga tekislanadi
                    cv2.putText(frame, text, (tx + 1, list_y + 1), font, list_font_scale,
                                (0, 0, 0), list_shadow_thickness, cv2.LINE_AA)
                    cv2.putText(frame, text, (tx, list_y), font, list_font_scale,
                                (0, 255, 0), list_thickness, cv2.LINE_AA)

        if pace:
            now_t = time.perf_counter()
            if play_start is None:
                play_start = now_t - (frame_count - 1) / fps
            late = now_t - (play_start + (frame_count - 1) / fps)
            if late < 0:
                time.sleep(-late)
            elif late > max_late:
                max_late = late

        if gui_available:
            try:
                display_frame = frame
                if width > 1920:
                    display_frame = cv2.resize(frame, (1280, 720))
                elif width < DISPLAY_MIN_WIDTH:
                    scale_up = DISPLAY_MIN_WIDTH / width
                    display_frame = cv2.resize(
                        frame, (int(width * scale_up), int(height * scale_up)),
                        interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Wagon Number + TONNA/HAJM RealTime", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    _RUN_STATE["quit"] = True
                    break
            except cv2.error as e:
                print(f"ℹ️  GUI/resize xatosi ({e}) -- oyna o'chirildi, ishlov berish davom etmoqda.")
                gui_available = False

        if out:
            if SAVE_RESIZE_WIDTH and SAVE_RESIZE_WIDTH < width:
                out.write(cv2.resize(frame, (save_width, save_height)))
            else:
                out.write(frame)

    for tid in list(last_seen_frame.keys()):
        process_finished_track(tid, track_scores.get(tid), track_candidate_frames.get(tid, {}),
                                track_position_votes.get(tid),
                                track_position_vote_frames.get(tid, 0),
                                track_lifespan.get(tid), stats,
                                track_any_digit_seen.get(tid, False),
                                track_rel_positions.get(tid), track_session_ordinal.get(tid),
                                best_image=track_best_image.get(tid),
                                raw_number_crop=track_best_raw_crop.get(tid),
                                tv_votes=track_tv_votes.get(tid))

    train_session.close()

    if out: out.release()
    cap.release()
    if gui_available:
        cv2.destroyAllWindows()
        for _ in range(3):
            cv2.waitKey(1)

    print("\n" + "=" * 50)
    print("📊 YAKUNIY HISOBOT")
    print(f"✅ Checksum bo'yicha to'g'ri (VALID) chiqqan vagonlar: {stats['read_valid']}")
    if TV_ENABLED:
        print(f"🔤 TONNA/HAJM o'qilgan vagonlar: {train_session.total_tv_found}")
    if pace and play_start is not None:
        played = time.perf_counter() - play_start
        print(f"⏱️  Real vaqt: video {frame_count / fps:.1f} s -- ko'rsatildi {played:.1f} s, "
              f"tashlangan kadrlar {dropped_frames}/{frame_count} "
              f"({100 * dropped_frames / max(frame_count, 1):.1f}%), eng katta kechikish {max_late * 1000:.0f} ms")
    print(f"📁 Barcha poyezd fayllari: {TRAINS_OUTPUT_DIR}")
    if SAVE_WAGON_IMAGES:
        print(f"🖼️  Vagon rasmlari: {os.path.join(TRAINS_OUTPUT_DIR, WAGON_IMAGES_SUBDIR)}")
    print("=" * 50)
    print("🛑 Work completed.")


if __name__ == "__main__":
    # python logic_N_T_V.py                      -- yuqoridagi VIDEO_SOURCE
    # python logic_N_T_V.py video.mp4            -- boshqa video
    # python logic_N_T_V.py video.mp4 --oynasiz  -- oynasiz (ketma-ket sinov uchun)
    # python logic_N_T_V.py video.mp4 --natijalar <papka>  -- hisobotni boshqa papkaga
    # python logic_N_T_V.py video.mp4 --tez      -- real vaqtni kutmasdan, imkon qadar tez
    # python logic_N_T_V.py video.mp4 --cpu      -- GPU bo'lsa ham CPU da (sinov uchun)
    # python logic_N_T_V.py video.mp4 --vaqt "2025-12-10 16:05:00"  -- video yozilgan real vaqt
    _args = sys.argv[1:]
    if "--oynasiz" in _args or "--no-window" in _args:
        SHOW_WINDOW = False
        _args = [a for a in _args if a not in ("--oynasiz", "--no-window")]
    if "--tez" in _args:
        REALTIME_PLAYBACK = False
        _args = [a for a in _args if a != "--tez"]
    if "--natijalar" in _args:
        _i = _args.index("--natijalar")
        if _i + 1 < len(_args):
            TRAINS_OUTPUT_DIR = _args[_i + 1]
            _args = _args[:_i] + _args[_i + 2:]
        else:
            print("⚠️  --natijalar parametri uchun papka ko'rsatilmadi.")
            _args = _args[:_i]
    if "--vaqt" in _args:
        _i = _args.index("--vaqt")
        if _i + 1 < len(_args):
            VIDEO_START_TIME = _args[_i + 1]
            try:
                datetime.datetime.strptime(VIDEO_START_TIME, "%Y-%m-%d %H:%M:%S")   # format tekshiruvi
            except ValueError:
                print(f"⚠️  --vaqt formati noto'g'ri (YYYY-MM-DD HH:MM:SS kerak): {VIDEO_START_TIME}")
                VIDEO_START_TIME = None
            _args = _args[:_i] + _args[_i + 2:]
        else:
            print("⚠️  --vaqt parametri uchun vaqt ko'rsatilmadi.")
            _args = _args[:_i]
    if "--cpu" in _args:
        DEVICE = "cpu"
        _args = [a for a in _args if a != "--cpu"]
    if "--jonli" in _args:
        # sinov: video faylni jonli kamera kabi (o'z tezligida, oxirida qayta ulanib) o'qish
        FORCE_LIVE = True
        _args = [a for a in _args if a != "--jonli"]
    if _args:
        VIDEO_SOURCE = int(_args[0]) if _args[0].isdigit() else _args[0]

    def _on_term(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _on_term)   # servis to'xtatilganda ham hisobot saqlanadi
    except (ValueError, AttributeError):
        pass

    if is_live_source(VIDEO_SOURCE):
        # 24/7: kutilmagan xato bo'lsa ham to'xtamaydi -- RESTART_DELAY_SECONDS dan keyin davom etadi.
        # Poyezd sessiyasi (train_session) saqlanib qoladi, yozilgan vagonlar yo'qolmaydi.
        while True:
            try:
                run_realtime()
                if _RUN_STATE["quit"]:
                    break
                print(f"↻ Ishlov to'xtadi -- {RESTART_DELAY_SECONDS} s dan keyin qayta ishga tushadi", flush=True)
            except KeyboardInterrupt:
                break
            except Exception:
                traceback.print_exc()
                print(f"↻ Kutilmagan xato -- {RESTART_DELAY_SECONDS} s dan keyin qayta ishga tushadi", flush=True)
            finally:
                for _c in list(_ACTIVE_CAPTURES):
                    _c.release()
            try:
                time.sleep(RESTART_DELAY_SECONDS)
            except KeyboardInterrupt:
                break
        print("\n🛑 To'xtatildi -- joriy poyezd hisoboti saqlanmoqda...", flush=True)
        train_session.close()
    else:
        try:
            run_realtime()
        except KeyboardInterrupt:
            print("\n🛑 To'xtatildi -- joriy poyezd hisoboti saqlanmoqda...", flush=True)
            train_session.close()
    sys.stdout.flush()
    sys.stderr.flush()
    # Qt/CUDA yopilish tartibidagi segfault oldini olish: barcha fayllar allaqachon yozilgan.
    os._exit(0)
