"""
Запуск обучения YOLO

Скачивает базовую YOLO11 и запускает процесс обучения на данных.

Вход: Подготовленный датасет.
Выход: Файл best.pt (дообученная модель).
"""

from ultralytics import YOLO
from pathlib import Path
import os
import config
import csv
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
RUNS_DIR = Path("runs/detect")

def train():
    # 1. Загружаем модель
    # yolo11n.pt - самая быстрая (nano). Скачается сама.
    print("⏳ Загрузка модели...")
    model = YOLO(config.YOLO_MODEL_NAME)

    # Путь к yaml файлу, который создал предыдущий скрипт
    data_yaml = Path("datasets/prepared/data.yaml")

    print("🚀 Начинаем обучение...")

    # 2. Запуск обучения
    # epochs=100 -> Нейросеть посмотрит все фото 100 раз.
    # imgsz=640 -> Размер картинки для анализа.
    # device=0 -> Использовать первую видеокарту (NVIDIA).
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=100,
            imgsz=640,
            device=0,

            # === ПАРАМЕТРЫ БЕЗОПАСНОСТИ ===
            batch=2,  # Ставим 2! Если заработает, потом попробуем 4 или 8.
            workers=0,  # ОБЯЗАТЕЛЬНО 0.
            amp=False,  # Отключаем ускорение AMP (бывает нестабильно на 40xx Mobile)
            plots=False,  # Не рисовать графики (экономим память)
            save=True,  # Сохранять веса
            val=True,  # Валидация включена
            name="your_class",  # Имя папки с результатами
            patience=20  # Если 20 эпох нет улучшений - стоп (Early Stopping)
            )

        print("🏁 Обучение завершено!")
        print(f"Лучшая модель сохранена здесь: runs/detect/your_class/weights/best.pt")

    except Exception as e:
        print(f"\n❌ ПРОИЗОШЛА ОШИБКА Python: {e}")
        import traceback
        traceback.print_exc()


def plot_training_results():
    if not RUNS_DIR.exists():
        print(f"❌ Папка {RUNS_DIR} не найдена.")
        return

    # 1. Ищем самую свежую папку с тренировкой
    # Сортируем папки по времени изменения (последняя - самая новая)
    all_runs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
    if not all_runs:
        print("Нет папок с тренировками.")
        return

    latest_run = max(all_runs, key=lambda d: d.stat().st_mtime)
    csv_path = latest_run / "results.csv"

    if not csv_path.exists():
        print(f"❌ В папке {latest_run} нет файла results.csv")
        return

    print(f"📊 Анализируем файл: {csv_path}")

    # 2. Читаем данные
    epochs = []
    box_loss = []  # Ошибка рамки (Train)
    map50 = []  # Точность (Validation)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        # Чистим названия колонок от лишних пробелов (YOLO иногда ставит пробелы)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for row in reader:
            try:
                # Собираем данные по эпохам
                epochs.append(int(row['epoch']))
                box_loss.append(float(row['train/box_loss']))
                map50.append(float(row['metrics/mAP50(B)']))
            except ValueError:
                continue

    # 3. Рисуем Графики
    plt.figure(figsize=(12, 5))

    # График 1: Ошибка (Loss) -> Должен падать
    plt.subplot(1, 2, 1)
    plt.plot(epochs, box_loss, label='Train Box Loss', color='red')
    plt.title('Ошибка (Box Loss) - Чем ниже, тем лучше')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # График 2: Точность (mAP) -> Должен расти
    plt.subplot(1, 2, 2)
    plt.plot(epochs, map50, label='mAP 50%', color='green')
    plt.title('Точность (mAP 50%) - Чем выше, тем лучше')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность (0.0 - 1.0)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Сохраняем картинку
    output_img = "my_training_results.png"
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"✅ Графики сохранены в файл: {output_img}")
    print("Открой этот файл в PyCharm или проводнике.")

    # Показать на экране (если работает GUI)
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    train()
    plot_training_results()