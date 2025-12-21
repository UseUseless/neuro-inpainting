"""
Скрипт обучения YOLOv11-SEG.
"""

import os
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
RUNS_DIR = Path("runs/segment")
DATA_YAML = config.TRAIN_DATASET_DIR / "data.yaml"

def train():
    print(f"🚀 Запуск обучения...")

    if not DATA_YAML.exists():
        print(f"❌ Не найден {DATA_YAML}. Запусти генератор!")
        return

    print(f"⏳ Загрузка: {config.YOLO_MODEL_NAME}...")
    try:
        model = YOLO(config.YOLO_MODEL_NAME)
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    print(f"🔥 Старт (Epochs={config.TRAIN_EPOCHS}, Batch={config.TRAIN_BATCH})...")

    try:
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

        print("\n🏁 Готово!")
        best_weight = RUNS_DIR / "train_seg_run" / "weights" / "best.pt"
        print(f"👉 Скопируй этот файл в models/: {best_weight}")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")

def plot_training_results():
    """
    Рисует расширенный дашборд из 6 графиков.
    """
    csv_path = RUNS_DIR / "train_seg_run" / "results.csv"

    if not csv_path.exists():
        print(f"⚠️ Файл {csv_path} не найден.")
        return

    print(f"📊 Генерация дашборда обучения...")

    data = {
        'epoch': [],
        'box_loss_train': [], 'box_loss_val': [],
        'seg_loss_train': [], 'seg_loss_val': [],
        'cls_loss_train': [], 'cls_loss_val': [],
        'map50_mask': [], 'map95_mask': [],
        'precision_mask': [], 'recall_mask': []
    }

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

            for row in reader:
                try:
                    data['epoch'].append(int(row['epoch']))

                    # Losses
                    data['box_loss_train'].append(float(row['train/box_loss']))
                    data['box_loss_val'].append(float(row['val/box_loss']))
                    data['seg_loss_train'].append(float(row['train/seg_loss']))
                    data['seg_loss_val'].append(float(row['val/seg_loss']))
                    data['cls_loss_train'].append(float(row['train/cls_loss']))
                    data['cls_loss_val'].append(float(row['val/cls_loss']))

                    # Metrics (Mask)
                    data['map50_mask'].append(float(row['metrics/mAP50(M)']))
                    data['map95_mask'].append(float(row['metrics/mAP50-95(M)']))
                    data['precision_mask'].append(float(row['metrics/precision(M)']))
                    data['recall_mask'].append(float(row['metrics/recall(M)']))
                except ValueError:
                    continue
    except Exception as e:
        print(f"❌ Ошибка CSV: {e}")
        return

    # Настройка графиков (2 строки, 3 колонки)
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('YOLOv11 Segmentation Training Dashboard', fontsize=16)

    epochs = data['epoch']

    # 1. SEGMENTATION LOSS (Самое важное)
    axs[0, 0].plot(epochs, data['seg_loss_train'], label='Train', color='red', linestyle='--')
    axs[0, 0].plot(epochs, data['seg_loss_val'], label='Val', color='darkred', linewidth=2)
    axs[0, 0].set_title('Ошибка Маски (Seg Loss)')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 2. BOX LOSS (Геометрия)
    axs[0, 1].plot(epochs, data['box_loss_train'], label='Train', color='blue', linestyle='--')
    axs[0, 1].plot(epochs, data['box_loss_val'], label='Val', color='darkblue')
    axs[0, 1].set_title('Ошибка Рамки (Box Loss)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # 3. CLASS LOSS (Узнаваемость)
    axs[0, 2].plot(epochs, data['cls_loss_train'], label='Train', color='orange', linestyle='--')
    axs[0, 2].plot(epochs, data['cls_loss_val'], label='Val', color='darkorange')
    axs[0, 2].set_title('Ошибка Класса (Is it watermark?)')
    axs[0, 2].legend()
    axs[0, 2].grid(True, alpha=0.3)

    # 4. mAP (Точность общая)
    axs[1, 0].plot(epochs, data['map50_mask'], label='mAP 50%', color='green', linewidth=2)
    axs[1, 0].plot(epochs, data['map95_mask'], label='mAP 50-95%', color='lightgreen')
    axs[1, 0].set_title('Точность Маски (mAP)')
    axs[1, 0].set_ylabel('Score (0-1)')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # 5. Precision & Recall (Баланс)
    axs[1, 1].plot(epochs, data['precision_mask'], label='Precision (Меткость)', color='purple')
    axs[1, 1].plot(epochs, data['recall_mask'], label='Recall (Охват)', color='cyan')
    axs[1, 1].set_title('Precision vs Recall')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    # 6. Сравнение переобучения (Seg Train vs Val)
    # Показывает разрыв (Gap) между обучением и тестом
    gap = [v - t for t, v in zip(data['seg_loss_train'], data['seg_loss_val'])]
    axs[1, 2].plot(epochs, gap, label='Val - Train Gap', color='gray')
    axs[1, 2].axhline(0, color='black', linestyle='--')
    axs[1, 2].set_title('Переобучение (Разрыв Loss)')
    axs[1, 2].legend()
    axs[1, 2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_img = "training_dashboard.png"
    plt.savefig(output_img)
    print(f"✅ Дашборд сохранен: {output_img}")

if __name__ == "__main__":
    train()
    plot_training_results()