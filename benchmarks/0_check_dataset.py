"""
Скрипт для проверки качества разметки (Visualize YOLO Polygons).
Берет фото и txt из train_dataset и рисует контуры.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь, чтобы видеть модули core и config
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import shutil
import cv2
import numpy as np
import random
from pathlib import Path
import config

# Откуда берем (берем из тренировочной части)
IMG_DIR = config.TRAIN_DATASET_DIR / "images" / "train"
LABEL_DIR = config.TRAIN_DATASET_DIR / "labels" / "train"

# Куда сохраним примеры
TEST_OUTPUT_DIR = Path("bench_tests/step0_train_labels_check.py")


def check_dataset():
    if not IMG_DIR.exists():
        print(f"❌ Папка {IMG_DIR} не найдена. Сначала запусти генератор.")
        return

    # Чистим папку вывода
    if TEST_OUTPUT_DIR.exists(): shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True)

    # Берем 10 случайных файлов
    all_images = list(IMG_DIR.glob("*.jpg"))
    if not all_images:
        print("❌ Нет картинок.")
        return

    sample_images = random.sample(all_images, min(10, len(all_images)))

    print(f"🕵️ Проверяем 10 случайных фото из датасета...")

    for img_path in sample_images:
        # Ищем пару .txt
        label_path = LABEL_DIR / (img_path.stem + ".txt")

        if not label_path.exists():
            print(f"⚠️ Нет лейбла для {img_path.name}")
            continue

        # Загружаем картинку
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Читаем координаты
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Рисуем каждый полигон
        for line in lines:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])  # Первый элемент - класс
            coords = parts[1:]  # Остальное - координаты x y x y...

            # Превращаем нормализованные (0..1) в пиксели
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                points.append([x, y])

            # Переводим в формат, понятный OpenCV
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # 1. Рисуем ЗЕЛЕНЫЙ контур (толщина 2)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # 2. Рисуем полупрозрачную заливку (чтобы видеть площадь)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Сохраняем результат
        save_path = TEST_OUTPUT_DIR / f"check_{img_path.name}"
        cv2.imwrite(str(save_path), img)
        print(f"✅ Сохранено: {save_path}")

    print(f"\n📂 Открой папку '{TEST_OUTPUT_DIR}' и посмотри глазами!")


if __name__ == "__main__":
    check_dataset()