"""
Скрипт обучения YOLOv11-SEG.
"""

import os
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import config
from core.utils import ensure_model

# Фикс для ошибки OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

RUNS_DIR = Path("runs/segment")
DATA_YAML = config.TRAIN_DATASET_DIR / "data.yaml"

# Путь к базовой модели (с которой начинаем обучение)
BASE_MODEL_PATH = config.MODELS_DIR / "yolo11s-seg.pt"

def train():
    print(f"🚀 ЗАПУСК ОБУЧЕНИЯ")
    print("=" * 40)

    # 1. Проверки
    if not DATA_YAML.exists():
        print(f"❌ Ошибка: Не найден {DATA_YAML}")
        print("   👉 Сначала запусти 1_image_generator.py!")
        return

    # 2. Гарантируем наличие базовой модели
    print(f"⏳ Проверка базовой модели...")
    try:
        ensure_model(BASE_MODEL_PATH, config.YOLO_MODEL_URL)
    except Exception as e:
        print(f"❌ Не могу скачать базовую модель: {e}")
        return

    print(f"🔥 Старт обучения (Epochs={config.TRAIN_EPOCHS}, Batch={config.TRAIN_BATCH})...")

    try:
        # Загружаем модель
        model = YOLO(BASE_MODEL_PATH)

        results = model.train(
            data=str(DATA_YAML),
            epochs=config.TRAIN_EPOCHS,
            imgsz=config.TRAIN_IMG_SIZE,
            patience=config.TRAIN_PATIENCE,
            batch=config.TRAIN_BATCH,
            workers=config.TRAIN_WORKERS,
            mosaic=config.TRAIN_MOSAIC,
            hsv_h=config.TRAIN_HSV_H,
            hsv_s=config.TRAIN_HSV_S,
            hsv_v=config.TRAIN_HSV_V,
            scale=config.TRAIN_SCALE,
            project="runs/segment",
            name="train_seg_run",
            exist_ok=True,
            save=True,
            val=True,
            plots=False,
            device=0 if config.DEVICE == 'cuda' else 'cpu'
        )

        print("\n🏁 Обучение завершено!")
        
        # Просто сообщаем путь, ничего не копируем
        best_weight = RUNS_DIR / "train_seg_run" / "weights" / "best.pt"
        print(f"👉 Лучшие веса здесь: {best_weight}")
        print(f"👉 Если результат устраивает — скопируй этот файл в 'models/best.pt'")

    except Exception as e:
        print(f"\n❌ Критическая ошибка обучения: {e}")

def plot_training_results():
    """
    Рисует дашборд (Losses + Metrics).
    """
    csv_path = RUNS_DIR / "train_seg_run" / "results.csv"

    if not csv_path.exists():
        return

    print(f"📊 Генерация графиков...")
    
    data = {k: [] for k in ['epoch', 'seg_loss_train', 'seg_loss_val', 'map50', 'precision', 'recall']}
    
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            
            for row in reader:
                try:
                    data['epoch'].append(int(row['epoch']))
                    data['seg_loss_train'].append(float(row['train/seg_loss']))
                    data['seg_loss_val'].append(float(row['val/seg_loss']))
                    data['map50'].append(float(row['metrics/mAP50(M)']))
                    data['precision'].append(float(row['metrics/precision(M)']))
                    data['recall'].append(float(row['metrics/recall(M)']))
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"⚠️ Ошибка чтения CSV: {e}")
        return

    if not data['epoch']:
        print("⚠️ Данных в CSV нет.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    epochs = data['epoch']

    # 1. Loss
    axs[0].plot(epochs, data['seg_loss_train'], label='Train Loss', color='red', linestyle='--')
    axs[0].plot(epochs, data['seg_loss_val'], label='Val Loss', color='darkred', linewidth=2)
    axs[0].set_title('Segmentation Loss')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # 2. mAP
    axs[1].plot(epochs, data['map50'], label='mAP 50%', color='green')
    axs[1].set_title('Accuracy (mAP 50%)')
    axs[1].grid(True, alpha=0.3)

    # 3. P vs R
    axs[2].plot(epochs, data['precision'], label='Precision', color='purple')
    axs[2].plot(epochs, data['recall'], label='Recall', color='cyan')
    axs[2].set_title('Precision & Recall')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    output_img = "training_dashboard.png"
    plt.savefig(output_img)
    print(f"✅ График сохранен: {output_img}")

if __name__ == "__main__":
    train()
    plot_training_results()