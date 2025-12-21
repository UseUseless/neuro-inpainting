import requests
import zipfile
import shutil
from tqdm import tqdm
import config

# Временный файл будет лежать внутри папки temp_download
ZIP_FILE = config.TEMP_DIR / "coco_backgrounds.zip"

def download_and_extract():
    # 1. Проверка: может уже скачано?
    if config.BACKGROUNDS_DIR.exists():
        count = len(list(config.BACKGROUNDS_DIR.glob("*.jpg")))
        if count > 4000:
            print(f"✅ Фоны уже на месте ({count} шт). Скачивать не нужно.")
            return

    # Убедимся, что временная папка чиста
    if config.TEMP_DIR.exists():
        shutil.rmtree(config.TEMP_DIR)
    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📥 Скачиваем архив в: {ZIP_FILE}")

    try:
        # Скачиваем
        response = requests.get(config.BACKGROUNDS_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(ZIP_FILE, "wb") as file, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

        print("📦 Распаковка архива во временную папку...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            # Распаковываем прямо в temp_download
            zip_ref.extractall(config.TEMP_DIR)

        print("📂 Перенос файлов в backgrounds...")
        # В архиве лежит папка "val2017". Ищем её внутри temp_download
        extracted_folder = config.TEMP_DIR / "val2017"

        if extracted_folder.exists():
            for file_path in extracted_folder.glob("*.jpg"):
                shutil.move(str(file_path), str(config.BACKGROUNDS_DIR / file_path.name))
        else:
            print("❌ Ошибка: Не нашел папку val2017 внутри архива!")
            return

        print("🧹 Очистка временных файлов...")
        # Удаляем всю папку temp_download целиком
        shutil.rmtree(config.TEMP_DIR)

        final_count = len(list(config.BACKGROUNDS_DIR.glob("*.jpg")))
        print(f"✅ Успешно! Скачано {final_count} фото.")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        # Если упали - не удаляем папку temp, чтобы можно было посмотреть, что скачалось
        print(f"   Временные файлы остались в {config.TEMP_DIR}")


if __name__ == "__main__":
    download_and_extract()