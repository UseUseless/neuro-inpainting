"""
Запуск обработки всего массива фото

Объединяет всю обработку после обучения

Сканирует папку со всем массивом фото.
Загрузит модели один раз в память.
Запускает цикл (использует tqdm для полоски прогресса).
Для каждого файла вызывает по очереди: Detector -> Segmenter -> Cleaner.
Сохраняет результат.
(Запустит конвейер: Нашел -> Выделил -> Стер -> Сохранил.)
Пишет лог, чтобы знать где остановились.

!!! Убедись, что на диске, где лежит проект, есть свободное место.
!!! Так как сохраняем копии (а не перезаписываем оригиналы для безопасности),
!!! нужно столько же места, сколько весят исходные фото.

!!! Прежде чем запускать пайплайн проверь работает ли видеокарта.
!!! Запусти gpu_check.py
"""

import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm  # Красивая полоска прогресса

# Импортируем наши модули
import config
from core.pipeline_logger import setup_logger
from core.detector import YourClassDetector
from core.segmenter import MaskRefiner
from core.cleaner import ImageInpainter

logger = setup_logger()


def main():
    # 1. Проверка папок
    if not config.INPUT_DIR.exists():
        logger.error(f"❌ Входная папка не найдена: {config.INPUT_DIR}")
        return

    # 2. Инициализация моделей (Самая долгая часть - загрузка в VRAM)
    logger.info("⏳ Загрузка нейросетей... (это займет время)")
    try:
        detector = YourClassDetector()
        segmenter = MaskRefiner()
        cleaner = ImageInpainter()
        logger.info("✅ Все модели успешно загружены!")
    except Exception as e:
        logger.critical(f"❌ Фатальная ошибка при загрузке моделей: {e}")
        return

    # 3. Сбор файлов
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_files = [
        f for f in config.INPUT_DIR.rglob("*")
        if f.suffix.lower() in extensions and f.is_file()
    ]

    total_files = len(all_files)
    logger.info(f"📁 Найдено изображений для обработки: {total_files}")

    if total_files == 0:
        logger.warning("Папка пуста. Нечего обрабатывать.")
        return

    # 4. Главный цикл (Processing Loop)
    start_time = time.time()
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # tqdm создает прогресс-бар в консоли
    for img_path in tqdm(all_files, desc="Processing", unit="img"):

        # Определяем путь для сохранения
        # Сохраняем ту же структуру папок, если она есть внутри input
        relative_path = img_path.relative_to(config.INPUT_DIR)
        save_path = config.OUTPUT_DIR / relative_path

        # Создаем подпапку, если нужно
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # === ПРОВЕРКА: Если файл уже есть, пропускаем (Idempotency) ===
        if save_path.exists():
            skipped_count += 1
            continue

        try:
            # === ШАГ 1: Загрузка ===
            # convert("RGB") важно, чтобы убрать альфа-канал, если он есть (LaMa любит RGB)
            with Image.open(img_path) as img:
                original_image = img.convert("RGB")

            # === ШАГ 2: Детекция (YOLO) ===
            boxes = detector.detect(original_image)

            if not boxes:
                # Если твоего класса нет - просто копируем оригинал
                # (или можно не сохранять, если нужно только очищенные. Но лучше сохранить)
                original_image.save(save_path)
                processed_count += 1
                continue

            # === ШАГ 3: Сегментация (SAM) ===
            mask = segmenter.create_mask(original_image, boxes)

            # === ШАГ 4: Очистка (LaMa) ===
            result_image = cleaner.clean(original_image, mask)

            # === ШАГ 5: Сохранение ===
            # Сохраняем с качеством 95, чтобы не плодить артефакты сжатия
            result_image.save(save_path, quality=95)
            processed_count += 1


        except KeyboardInterrupt:
            logger.warning("\n🛑 Процесс остановлен пользователем.")
            break


        except Exception as e:
            # Логируем ошибку в errors.log
            logger.error(f"❌ Ошибка на файле {img_path.name}: {e}")
            error_count += 1

            # Пишем путь в failed_files.txt
            failed_log_path = config.LOG_DIR / "failed_files.txt"
            try:
                with open(failed_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{img_path}\n")
            except Exception as log_err:
                logger.error(f"Не удалось записать в список ошибок: {log_err}")

    # 5. Итоги
    elapsed_time = time.time() - start_time
    logger.info("=" * 40)
    logger.info("🏁 Обработка завершена!")
    logger.info(f"⏱  Время выполнения: {elapsed_time:.2f} сек")
    logger.info(f"✅ Обработано: {processed_count}")
    logger.info(f"⏭  Пропущено: {skipped_count}")
    logger.info(f"❌ Ошибок: {error_count}")

    if processed_count > 0:
        avg_speed = elapsed_time / processed_count
        logger.info(f"🚀 Средняя скорость: {avg_speed:.2f} сек/фото")


if __name__ == "__main__":
    main()