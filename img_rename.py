import os
import uuid
from pathlib import Path
from tqdm import tqdm

# === НАСТРОЙКИ ===
# Папка, в которой лежат картинки (например, dataset_raw)
TARGET_FOLDER = Path("dataset_raw")

# Желаемое имя (получится img_1.jpg, img_2.png и т.д.)
BASE_NAME = "img_"


def batch_rename():
    if not TARGET_FOLDER.exists():
        print(f"❌ Папка {TARGET_FOLDER} не найдена!")
        return

    # Список поддерживаемых форматов
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    # Собираем все файлы картинок
    files = [
        f for f in TARGET_FOLDER.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]

    if not files:
        print("Картинок не найдено.")
        return

    print(f"Найдено {len(files)} файлов. Начинаем переименование...")

    # --- ЭТАП 1: Переименование во временные уникальные имена ---
    # Это защищает от конфликтов (если файл img_1 уже существует)
    temp_files = []
    for file_path in tqdm(files, desc="Temp rename"):
        # Генерируем случайное имя: asd3-234f-df43.tmp
        temp_name = f"{uuid.uuid4()}{file_path.suffix}"
        temp_path = TARGET_FOLDER / temp_name

        try:
            file_path.rename(temp_path)
            temp_files.append(temp_path)
        except OSError as e:
            print(f"Ошибка при временном переименовании {file_path}: {e}")

    # --- ЭТАП 2: Переименование в итоговые красивые имена ---
    # img_1.jpg, img_2.png ...
    # Сортируем временные файлы, чтобы порядок был детерминированным (хотя UUID случайны)
    # Если важен старый порядок, нужно было сохранять его на этапе 1, но обычно это не важно.

    count = 1
    for temp_path in tqdm(temp_files, desc="Final rename"):
        # Формируем имя: img_1.jpg
        new_name = f"{BASE_NAME}{count}{temp_path.suffix}"
        final_path = TARGET_FOLDER / new_name

        try:
            temp_path.rename(final_path)
            count += 1
        except OSError as e:
            print(f"Ошибка при финальном переименовании {temp_path}: {e}")

    print(f"✅ Готово! Переименовано {count - 1} файлов.")


if __name__ == "__main__":
    # Запрос подтверждения, чтобы случайно не запустить
    print(f"ВНИМАНИЕ: Все картинки в папке '{TARGET_FOLDER}' будут переименованы!")
    user_input = input("Напиши 'yes' чтобы продолжить: ")

    if user_input.lower() == 'yes':
        batch_rename()
    else:
        print("Отмена.")