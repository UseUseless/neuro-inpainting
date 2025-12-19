"""
Запуск полного цикла обработки (Pipeline).

Цепочка действий:
1. Загрузка фото.
2. Detector (YOLO) -> Находит координаты и класс.
3. Segmenter (MaskRefiner) -> Выбирает стратегию (Box/Sam) и делает маску.
4. Cleaner (LaMa) -> Закрашивает маску.
5. Сохранение результата.

Особенности:
- Можно прервать (Ctrl+C) и продолжить позже.
- Сохраняет структуру папок.
- Логирует ошибки, но не падает.
"""

import time
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm  # Прогресс-бар

# Импорт наших модулей
import config
from core.pipeline_logger import setup_logger
from core.detector import YourClassDetector
from core.segmenter import MaskRefiner
from core.cleaner import ImageInpainter

# Настройка логирования
logger = setup_logger()

def main():
    # 1. ПРОВЕРКИ
    if not config.INPUT_DIR.exists():
        logger.error(f"❌ Входная папка не найдена: {config.INPUT_DIR}")
        print(f"Создай папку {config.INPUT_DIR} и положи туда фото!")
        return

    # 2. ЗАГРУЗКА МОДЕЛЕЙ (Самая тяжелая часть)
    logger.info("⏳ Загрузка нейросетей в память... (подожди 10-20 сек)")
    try:
        detector = YourClassDetector()  # YOLO
        segmenter = MaskRefiner()       # SAM / Logic
        cleaner = ImageInpainter()      # LaMa
        logger.info("✅ Все модели успешно загружены!")
    except Exception as e:
        logger.critical(f"❌ Не удалось загрузить модели: {e}")
        return

    # 3. ПОИСК ФАЙЛОВ
    # Ищем рекурсивно (rglob), чтобы поддерживать вложенные папки
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_files = [
        f for f in config.INPUT_DIR.rglob("*")
        if f.suffix.lower() in valid_extensions and f.is_file()
    ]

    total_files = len(all_files)
    logger.info(f"📁 Найдено изображений: {total_files}")

    if total_files == 0:
        logger.warning("Папка пуста. Нечего обрабатывать.")
        return

    # 4. ЗАПУСК КОНВЕЙЕРА
    start_time = time.time()
    processed_count = 0
    skipped_count = 0
    error_count = 0
    no_detection_count = 0

    print("\n🚀 Поехали! (Нажми Ctrl+C, чтобы остановить мягко)\n")

    try:
        # tqdm создает полоску прогресса
        for img_path in tqdm(all_files, desc="Processing", unit="img"):

            # --- Подготовка путей ---
            # Вычисляем относительный путь (например: subfolder/image.jpg)
            relative_path = img_path.relative_to(config.INPUT_DIR)
            # Итоговый путь сохранения
            save_path = config.OUTPUT_DIR / relative_path

            # Создаем папку назначения, если её нет
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # --- ПРОВЕРКА: УЖЕ СДЕЛАНО? ---
            if save_path.exists():
                skipped_count += 1
                continue

            try:
                # --- ШАГ 0: Открытие ---
                with Image.open(img_path) as img:
                    # Конвертируем в RGB (LaMa не любит CMYK и Transparency)
                    original_image = img.convert("RGB")

                # --- ШАГ 1: Детекция (YOLO) ---
                # detections = [(x1, y1, x2, y2, conf, cls_id), ...]
                detections = detector.detect(original_image)

                if not detections:
                    # Если ничего не нашли - просто сохраняем оригинал.
                    # Это важно, чтобы выходная папка была полной копией входной.
                    original_image.save(save_path, quality=95)
                    no_detection_count += 1
                    continue

                # --- ШАГ 2: Сегментация (Mask Creation) ---
                # Передаем список детекций, сегментер сам решит (Box или Sam)
                mask = segmenter.create_mask(original_image, detections)

                # --- ШАГ 3: Очистка (Inpainting) ---
                # Если маска пустая (бывает такое), клинер вернет оригинал
                result_image = cleaner.clean(original_image, mask)

                # --- ШАГ 4: Сохранение ---
                result_image.save(save_path, quality=95)
                processed_count += 1

            except Exception as e:
                logger.error(f"❌ Ошибка на файле {img_path.name}: {e}")
                error_count += 1
                # Записываем имя битого файла, чтобы потом разобраться
                with open(config.LOG_DIR / "failed_files.txt", "a") as f:
                    f.write(f"{img_path}\n")

    except KeyboardInterrupt:
        logger.warning("\n🛑 Процесс остановлен пользователем (Ctrl+C).")
        logger.warning("   Прогресс сохранен. Запусти скрипт снова, чтобы продолжить.")

    # 5. ИТОГИ
    elapsed = time.time() - start_time
    logger.info("=" * 40)
    logger.info("🏁 Работа завершена!")
    logger.info(f"⏱  Время: {elapsed:.2f} сек")
    logger.info(f"✅ Обработано успешно: {processed_count}")
    logger.info(f"👻 Без ватермарок (копии): {no_detection_count}")
    logger.info(f"⏭  Пропущено (было готово): {skipped_count}")
    logger.info(f"❌ Ошибок: {error_count}")

    if processed_count > 0:
        avg_speed = elapsed / processed_count
        logger.info(f"🚀 Средняя скорость: {avg_speed:.2f} сек/фото")

if __name__ == "__main__":
    main()