"""
Подготовка данных для обучения YOLO (Multi-Class)
Функционал:
1. Читает classes.txt из dataset_raw (источник истины).
2. Лечит битые JPEGs (clean_images).
3. Проверяет корректность классов в файлах разметки.
4. Раскладывает на train/val.
5. Генерирует data.yaml.
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# === НАСТРОЙКИ ===
SOURCE_DIR = Path("dataset_raw")
DEST_DIR = Path("datasets/prepared")
TRAIN_RATIO = 0.8  # 80% на обучение, 20% на проверку

def get_classes_from_source():
    """
    Читает имена классов прямо из файла, который создал LabelImg.
    Это гарантирует, что порядок ID (0, 1, 2) совпадет с обучением.
    """
    classes_file = SOURCE_DIR / "classes.txt"
    if not classes_file.exists():
        print(f"❌ ОШИБКА: Файл {classes_file} не найден!")
        print("   -> Открой LabelImg, выбери папку dataset_raw и сохрани хотя бы одну картинку.")
        print("   -> LabelImg сам создаст classes.txt.")
        return []

    with open(classes_file, "r", encoding="utf-8") as f:
        # Читаем строки, убираем пустые
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    if not class_names:
        print("❌ ОШИБКА: Файл classes.txt пуст!")
        return []

    print(f"📋 Найдено классов: {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f"   [{i}] {name}")

    return class_names

def clean_images_in_source():
    """
    Проверяет и лечит картинки ПРЯМО В ИСХОДНИКЕ перед копированием.
    Предотвращает вылеты YOLO из-за битых заголовков или EXIF.
    """
    print("\n🧹 [1/3] Проверка и лечение изображений...")

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    # Собираем только картинки
    files = [f for f in SOURCE_DIR.iterdir() if f.suffix.lower() in valid_extensions]

    fixed_count = 0
    removed_count = 0

    for img_path in tqdm(files, desc="Checking images"):
        try:
            # 1. Быстрая проверка
            with Image.open(img_path) as img:
                img.verify()

            # 2. Полная перезапись (убирает лишние каналы, EXIF и мусор)
            with Image.open(img_path) as img:
                # Конвертируем в RGB (убирает альфа-канал, который YOLO не любит)
                rgb_img = img.convert("RGB")
                # Перезаписываем тот же файл
                rgb_img.save(img_path, quality=95)
                fixed_count += 1

        except Exception as e:
            print(f"❌ БИТЫЙ ФАЙЛ УДАЛЕН: {img_path.name} ({e})")
            os.remove(img_path)
            removed_count += 1
            # Если есть txt для него - тоже удаляем, чтобы не было сирот
            txt = img_path.with_suffix(".txt")
            if txt.exists(): os.remove(txt)

    print(f"✅ Проверено: {len(files)}. Пересохранено: {fixed_count}. Удалено битых: {removed_count}.")


def prepare_data():
    # 0. Получаем список классов
    class_names = get_classes_from_source()
    if not class_names:
        return

    print("\n📦 [2/3] Подготовка структуры Dataset...")

    if not SOURCE_DIR.exists():
        print(f"❌ Ошибка: Папка {SOURCE_DIR} не найдена.")
        return

    # Очистка старого датасета
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)

    # Создаем структуру папок YOLO
    for split in ['train', 'val']:
        (DEST_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DEST_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Собираем файлы (уже почищенные)
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = [f for f in SOURCE_DIR.iterdir() if f.suffix.lower() in valid_extensions]

    # Оставляем только те, у которых есть разметка (.txt)
    valid_pairs = []
    for img in files:
        txt = img.with_suffix('.txt')
        if txt.exists():
            valid_pairs.append(img)
        else:
            # Это нормально, если ты просто накидал фоток, но еще не успел разметить
            # print(f"⚠️ Пропуск (нет txt): {img.name}")
            pass

    if not valid_pairs:
        print("❌ Нет готовых пар (фото + txt). Проверь, что ты сохранил разметку в LabelImg.")
        return

    # Перемешиваем и делим на Train/Val
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * TRAIN_RATIO)
    train_files = valid_pairs[:split_idx]
    val_files = valid_pairs[split_idx:]

    print(f"Всего пар с разметкой: {len(valid_pairs)}. Train: {len(train_files)}, Val: {len(val_files)}")

    # Функция копирования с проверкой валидности классов
    def copy_and_validate(file_list, split_name):
        max_class_id = len(class_names) - 1

        for img_path in tqdm(file_list, desc=f"Copying to {split_name}"):
            txt_path = img_path.with_suffix('.txt')

            # 1. Читаем и валидируем лейблы
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            clean_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts: continue

                try:
                    cls_id = int(parts[0])
                    # Проверка: ID класса не должен быть больше, чем количество классов
                    if cls_id > max_class_id:
                        print(f"\n⚠️ ОШИБКА РАЗМЕТКИ в {txt_path.name}:")
                        print(f"   Нашел класс ID={cls_id}, а у нас всего {len(class_names)} классов (0-{max_class_id}).")
                        print(f"   -> Строка удалена. Проверь classes.txt в dataset_raw!")
                        continue

                    clean_lines.append(line)
                except ValueError:
                    continue

            # Если после чистки остались строки - копируем всё
            if clean_lines:
                # Копируем картинку
                shutil.copy(img_path, DEST_DIR / 'images' / split_name / img_path.name)

                # Записываем чистый txt
                dest_txt = DEST_DIR / 'labels' / split_name / txt_path.name
                with open(dest_txt, 'w') as f:
                    f.writelines(clean_lines)

    copy_and_validate(train_files, 'train')
    copy_and_validate(val_files, 'val')

    # Генерируем data.yaml
    # Этот файл YOLO будет читать при обучении
    yaml_content = f"""
path: {DEST_DIR.absolute().as_posix()} 
train: images/train
val: images/val

# КОЛИЧЕСТВО КЛАССОВ
nc: {len(class_names)}

# ИМЕНА КЛАССОВ
names: {class_names}
    """

    with open(DEST_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("\n✅ [3/3] Конфиг data.yaml создан.")
    print("Содержимое data.yaml:")
    print("-" * 20)
    print(yaml_content.strip())
    print("-" * 20)
    print(f"🎯 Готово! Можно запускать 2_train_model.py")

if __name__ == "__main__":
    # Сначала чистим картинки
    clean_images_in_source()
    # Потом готовим датасет
    prepare_data()