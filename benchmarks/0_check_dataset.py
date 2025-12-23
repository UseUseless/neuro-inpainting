"""
Скрипт для проверки качества разметки (Visualize YOLO Polygons).
"""

import sys
from pathlib import Path

# Добавляем корень проекта
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import shutil
import cv2
import numpy as np
import random
import config

IMG_DIR = config.TRAIN_DATASET_DIR / "images" / "train"
LABEL_DIR = config.TRAIN_DATASET_DIR / "labels" / "train"
TEST_OUTPUT_DIR = Path("bench_tests/step0_train_labels_check")

def check_dataset():
    if not IMG_DIR.exists():
        print(f"❌ Папка {IMG_DIR} не найдена. Сначала запусти 1_image_generator.py.")
        return

    if TEST_OUTPUT_DIR.exists(): shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True)

    all_images = list(IMG_DIR.glob("*.jpg"))
    if not all_images:
        print("❌ Нет картинок.")
        return

    # Берем 15 случайных (побольше, чтобы поймать разные)
    sample_images = random.sample(all_images, min(15, len(all_images)))

    print(f"🕵️ Проверяем {len(sample_images)} фото из датасета...")

    for img_path in sample_images:
        label_path = LABEL_DIR / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        has_label = False

        if label_path.exists():
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                if not line.strip(): continue # 🔥 ФИКС: Пропуск пустых строк
                
                parts = list(map(float, line.strip().split()))
                if len(parts) < 3: continue # 🔥 ФИКС: Защита от битых данных

                coords = parts[1:]
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i + 1] * h)
                    points.append([x, y])

                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Полупрозрачная заливка
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                has_label = True

        # Пишем на картинке статус
        status_text = "WATERMARK" if has_label else "CLEAN (Negative)"
        color = (0, 255, 0) if has_label else (255, 0, 0)
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        save_path = TEST_OUTPUT_DIR / f"check_{img_path.name}"
        cv2.imwrite(str(save_path), img)
        print(f"✅ {save_path.name} ({status_text})")

    print(f"\n📂 Результаты в папке: {TEST_OUTPUT_DIR}")

if __name__ == "__main__":
    check_dataset()