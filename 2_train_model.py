"""
Запуск обучения YOLO

Этот скрипт берет подготовленные данные и учит нейросеть находить твои объекты.
Он автоматически подхватит количество классов из файла data.yaml.

Вход: datasets/prepared/data.yaml
Выход: runs/detect/train_run/weights/best.pt
"""

import os
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import config

# Фикс для частой ошибки на Windows (OMP: Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Папка, куда YOLO будет складывать результаты
RUNS_DIR = Path("runs/detect")
DATA_YAML = Path("datasets/prepared/data.yaml")

def train():
    print(f"🚀 Запуск скрипта обучения...")

    # 1. Проверка наличия данных
    if not DATA_YAML.exists():
        print(f"❌ ОШИБКА: Не найден файл конфигурации: {DATA_YAML}")
        print("   -> Сначала запусти '1_prepare_dataset.py'!")
        return

    # 2. Загружаем модель
    # yolo11n.pt - самая легкая и быстрая (nano).
    # Если качество будет низким, можно поменять на yolo11s.pt (small) в config.py
    print(f"⏳ Загрузка базовой модели: {config.YOLO_MODEL_NAME}...")
    model = YOLO(config.YOLO_MODEL_NAME)

    print("🔥 Начинаем процесс обучения. Это может занять время...")
    print(f"   Устройство: {config.DEVICE}")

    # 3. Запуск обучения
    try:
        results = model.train(
            data=str(DATA_YAML),

            # === ГЛАВНЫЕ ПАРАМЕТРЫ ===
            epochs=100,         # 100 эпох обычно достаточно для простой задачи
            imgsz=640,          # Размер картинки (стандарт YOLO)
            patience=15,        # Если 15 эпох нет улучшений - стоп (Early Stopping)
            batch=2,            # Сколько картинок за раз (если вылетает OutOfMemory, ставь 2 или 1)

            # === ТЕХНИЧЕСКИЕ НАСТРОЙКИ ===
            device=0 if config.DEVICE == 'cuda' else 'cpu',
            workers=0,          # Для Windows лучше 0, чтобы не было ошибок мультипроцессинга
            project="runs/detect",
            name="train_run",   # Имя папки с результатами
            exist_ok=True,      # Перезаписывать папку, если существует (чтобы не плодить train_run2, train_run3)

            # === ГРАФИКА И ЭКОНОМИЯ ===
            plots=False,        # Не рисовать графики встроенными средствами (мы нарисуем свои, легче)
            save=True,          # Сохранять веса (best.pt)
            val=True,           # Проверять качество на валидации
            amp=False           # Отключаем Mixed Precision (иногда глючит на мобильных RTX картах)
        )

        print("\n🏁 Обучение завершено успешно!")

        # Путь к результату
        best_weight = RUNS_DIR / "train_run" / "weights" / "best.pt"
        print(f"💎 ЛУЧШАЯ МОДЕЛЬ: {best_weight}")
        print("👉 НЕ ЗАБУДЬ: Скопируй этот файл в папку models/ перед запуском пайплайна!")

    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА ОБУЧЕНИЯ: {e}")
        import traceback
        traceback.print_exc()


def plot_training_results():
    """
    Рисует простые и понятные графики прогресса обучения.
    """
    csv_path = RUNS_DIR / "train_run" / "results.csv"

    if not csv_path.exists():
        print(f"⚠️ Нет файла статистики: {csv_path}. Графики не построены.")
        return

    print(f"\n📊 Строим графики обучения...")

    epochs = []
    box_loss = []   # Ошибка предсказания рамки
    map50 = []      # Точность (mAP 50%)

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            # Чистим пробелы в названиях колонок YOLO (они любят писать " train/box_loss")
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

            for row in reader:
                epochs.append(int(row['epoch']))
                box_loss.append(float(row['train/box_loss']))
                map50.append(float(row['metrics/mAP50(B)']))
    except Exception as e:
        print(f"Ошибка чтения CSV: {e}")
        return

    # Рисуем
    plt.figure(figsize=(12, 6))

    # 1. График Ошибки (должен падать)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, box_loss, label='Box Loss', color='red', linewidth=2)
    plt.title('Ошибка (Loss) -> Должна падать')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. График Точности (должен расти)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, map50, label='mAP 50%', color='green', linewidth=2)
    plt.title('Точность (Accuracy) -> Должна расти')
    plt.xlabel('Эпохи')
    plt.ylabel('mAP (0.0 - 1.0)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_img = "training_report.png"
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"✅ График сохранен: {output_img}")

    # Пытаемся показать (если есть GUI)
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    train()
    plot_training_results()