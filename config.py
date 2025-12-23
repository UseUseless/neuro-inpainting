"""
Глобальная конфигурация проекта.
"""

import os
import torch
from pathlib import Path

# ==============================================================================
# 📂 1. ПУТИ И ИСТОЧНИКИ
# ==============================================================================
BASE_DIR = Path(__file__).parent

# Рабочие папки
INPUT_DIR = BASE_DIR / "images_input"       # Твои фото для обработки
OUTPUT_DIR = BASE_DIR / "images_cleaned"    # Результат
LOG_DIR = BASE_DIR / "logs"                 # Логи
MODELS_DIR = BASE_DIR / "models"            # Веса моделей (ТУТ ЛЕЖАТ ФАЙЛЫ)
TEMP_DIR = BASE_DIR / "temp_download"       # Временная папка

# Обучающие данные
TRAIN_DATASET_DIR = BASE_DIR / "train_dataset"
BACKGROUNDS_DIR = BASE_DIR / "backgrounds"

# === 🔗 URL ИСТОЧНИКИ (Для авто-скачивания) ===
BACKGROUNDS_URL = "http://images.cocodataset.org/zips/val2017.zip"

# LaMa (Big-Lama.pt) - Зеркало HuggingFace
LAMA_MODEL_URL = "https://huggingface.co/fashn-ai/LaMa/resolve/main/big-lama.pt"
LAMA_MODEL_PATH = MODELS_DIR / "big-lama.pt"

# YOLOv11s-seg - Официальный репозиторий
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt"
YOLO_MODEL_PATH = MODELS_DIR / "best.pt"    # Путь к ТВОЕЙ обученной модели

# Базовая модель для старта обучения
YOLO_BASE_MODEL_PATH = MODELS_DIR / "yolo11s-seg.pt"

# Файл ватермарки
WATERMARK_SOURCE = BASE_DIR / "watermark.png"

# ==============================================================================
# 🎨 2. НАСТРОЙКИ ГЕНЕРАТОРА (SYNTHETIC DATA)
# ==============================================================================
GEN_EXTRA_SOLID_PERCENT = 0.20  # +20% градиентов
GEN_TRAIN_RATIO = 0.8           # 80% train / 20% val

# Вероятности
GEN_PROB_NEGATIVE = 0.05        # 5% пустых фото
GEN_PROB_EASY_MODE = 0.20       # 20% легких примеров
GEN_PROB_EDGE_CORRUPTION = 0.50 # 50% рваные края

# Геометрия
GEN_SCALE_RANGE = (0.15, 0.85)  # Размер ватермарки
GEN_ROTATION_PROB = 0.5         # Поворот

# Искажения
GEN_OPACITY_RANGE = (0.05, 0.95)
GEN_INVERT_PROB = 0.3
GEN_BLUR_PROB = 0.5
GEN_COLOR_PROB = 0.3

GEN_SOURCE_REPAIR_DILATION = 2

# ==============================================================================
# 🏋️ 3. НАСТРОЙКИ ОБУЧЕНИЯ (YOLO Training)
# ==============================================================================
TRAIN_EPOCHS = 200
TRAIN_IMG_SIZE = 640
TRAIN_PATIENCE = 15
TRAIN_BATCH = 8

# 🔥 АВТО-ОПРЕДЕЛЕНИЕ OS ДЛЯ WORKERS
# Windows не умеет workers > 0 в spawn-режиме без танцев с бубном.
# Linux (Docker) умеет и нуждается в этом для скорости.
if os.name == 'nt':
    TRAIN_WORKERS = 0
else:
    TRAIN_WORKERS = 8  # Для Docker/Linux ставим побольше

# Аугментации
TRAIN_MOSAIC = 1.0
TRAIN_HSV_H = 0.015
TRAIN_HSV_S = 0.7
TRAIN_HSV_V = 0.4
TRAIN_SCALE = 0.5

# ==============================================================================
# 🕵️ 4. ДЕТЕКТОР И ПАЙПЛАЙН
# ==============================================================================
YOLO_CONFIDENCE = 0.25      # Порог уверенности

# Очистка (LaMa)
CLEANER_MASK_DILATION = 6   # Расширение маски на 6px перед удалением

# ==============================================================================
# ⚙️ СИСТЕМА
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_directories():
    for path in [INPUT_DIR, OUTPUT_DIR, LOG_DIR, MODELS_DIR, TRAIN_DATASET_DIR, BACKGROUNDS_DIR, TEMP_DIR]:
        path.mkdir(parents=True, exist_ok=True)

setup_directories()