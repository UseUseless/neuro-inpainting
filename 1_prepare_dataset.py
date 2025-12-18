"""
Подготовка данных для обучения

Берет папку dataset_raw (где всё в куче) и правильно раскладывает её для нейросети:
создает папки train (для учебы) и val (для проверки),
перемешивает файлы и создает конфиг data.yaml.

Вход: Папка с картинками и txt (dataset_raw).
Выход: Структурированная папка datasets/prepared.
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# === Настройки ===
# Откуда берем (твоя папка с разметкой)
SOURCE_DIR = Path("dataset_raw")
# Куда положим подготовленные данные для YOLO
DEST_DIR = Path("datasets/prepared")
# Процент данных для обучения (остальное на проверку)
TRAIN_RATIO = 0.8
# Папка, где лежат подготовленные лейблы (и train, и val)
BASE_DIR = Path("datasets/prepared/labels")
# Папки с картинками для обучения
TARGET_DIRS = [
    Path("datasets/prepared/images/train"),
    Path("datasets/prepared/images/val")
]


def prepare_data():
    if not SOURCE_DIR.exists():
        print(f"❌ Ошибка: Папка {SOURCE_DIR} не найдена. Сначала разметь фото!")
        return

    # 1. Создаем структуру папок, которую требует YOLO
    # datasets/prepared/images/train
    # datasets/prepared/labels/train ...
    for type_ in ['images', 'labels']:
        for split in ['train', 'val']:
            (DEST_DIR / type_ / split).mkdir(parents=True, exist_ok=True)

    # 2. Собираем все пары (jpg + txt)
    # Поддерживаем jpg, jpeg, png, webp
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    files = [f for f in SOURCE_DIR.iterdir() if f.suffix.lower() in extensions]

    # Фильтруем только те, у которых есть пара .txt (разметка)
    valid_files = []
    for img_path in files:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            valid_files.append(img_path)

    if not valid_files:
        print("❌ Не найдено размеченных пар (фото + txt).")
        return

    # 3. Перемешиваем и делим
    random.shuffle(valid_files)
    split_idx = int(len(valid_files) * TRAIN_RATIO)
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]

    print(f"Найдено {len(valid_files)} фото. Обучение: {len(train_files)}, Проверка: {len(val_files)}")

    # 4. Копируем файлы по папкам
    def copy_set(files_list, split_name):
        for img_path in files_list:
            txt_path = img_path.with_suffix(".txt")

            # Копируем картинку
            shutil.copy(img_path, DEST_DIR / "images" / split_name / img_path.name)
            # Копируем txt
            shutil.copy(txt_path, DEST_DIR / "labels" / split_name / txt_path.name)

    copy_set(train_files, "train")
    copy_set(val_files, "val")

    # 5. Создаем файл data.yaml (Инструкция для YOLO)
    yaml_content = f"""
    path: {DEST_DIR.absolute().as_posix()} 
    train: images/train
    val: images/val

    nc: 1
    names: ['your_class']
        """

    with open(DEST_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("✅ Данные успешно подготовлены в папке 'datasets/prepared'")
    print("Теперь можно запускать обучение!")


def fix_labels():
    if not BASE_DIR.exists():
        print(f"❌ Папка {BASE_DIR} не найдена. Запусти сначала 1_prepare_dataset.py")
        return

    # Ищем все .txt файлы во всех подпапках (train и val)
    files = list(BASE_DIR.rglob("*.txt"))

    print(f"Найдено {len(files)} файлов разметки. Исправляем...")

    fixed_count = 0
    for txt_file in files:
        # Читаем содержимое
        with open(txt_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        changed = False

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            # Проверяем первый элемент (номер класса)
            class_id = parts[0]

            # Если это не 0, меняем на 0
            if class_id != "0":
                parts[0] = "0"
                new_line = " ".join(parts) + "\n"
                new_lines.append(new_line)
                changed = True
            else:
                new_lines.append(line)

        # Если были изменения, перезаписываем файл
        if changed:
            with open(txt_file, "w") as f:
                f.writelines(new_lines)
            fixed_count += 1

    print(f"✅ Готово! Исправлено файлов: {fixed_count}")


def clean_images():
    print("🧹 Начинаем чистку изображений...")

    for folder in TARGET_DIRS:
        if not folder.exists():
            continue

        files = list(folder.glob("*.*"))
        for img_path in tqdm(files, desc=f"Cleaning {folder.name}"):
            try:
                # Читаем картинку
                with Image.open(img_path) as img:
                    # Конвертируем в RGB (убирает лишние каналы и битые заголовки)
                    img = img.convert("RGB")
                    # Перезаписываем поверх старой
                    img.save(img_path, "JPEG", quality=100)
            except Exception as e:
                print(f"❌ Удаляем совсем битый файл: {img_path}")
                os.remove(img_path)

    print("✅ Все картинки пересохранены и вылечены.")


if __name__ == "__main__":
    prepare_data()
    fix_labels()
    clean_images()