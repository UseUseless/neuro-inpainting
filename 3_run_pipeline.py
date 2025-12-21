"""
Запуск полного цикла обработки (Pipeline).
Версия: YOLO-SEG (End-to-End).

Цепочка действий теперь максимально проста:
1. Загрузка фото.
2. Detector (YOLO-Seg) -> Сразу выдает готовую Ч/Б маску.
3. Cleaner (LaMa) -> Закрашивает маску.
4. Сохранение.

Больше нет отдельного шага сегментации (SAM/Box), всё делает YOLO.
"""

import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Импорты модулей
import config
from core.pipeline_logger import setup_logger
from core.detector import YourClassDetector
from core.cleaner import ImageInpainter

# Настройка логирования
logger = setup_logger()

def main():
    # 1. ПРОВЕРКИ
    if not config.INPUT_DIR.exists():
        logger.error(f"❌ Входная папка не найдена: {config.INPUT_DIR}")
        print(f"Создай папку {config.INPUT_DIR} и положи туда фото!")
        return

    # 2. ЗАГРУЗКА МОДЕЛЕЙ
    logger.info("⏳ Загрузка нейросетей (YOLO-Seg + LaMa)...")
    try:
        # Detector теперь умный: он сам делает маску
        detector = YourClassDetector()

        # Cleaner остался прежним
        cleaner = ImageInpainter()

        logger.info("✅ Модели загружены и готовы к бою!")
    except Exception as e:
        logger.critical(f"❌ Фатальная ошибка при загрузке: {e}")
        return

    # 3. ПОИСК ФАЙЛОВ
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_files = [
        f for f in config.INPUT_DIR.rglob("*")
        if f.suffix.lower() in valid_extensions and f.is_file()
    ]

    total_files = len(all_files)
    logger.info(f"📁 Найдено изображений: {total_files}")

    if total_files == 0:
        logger.warning("Папка пуста.")
        return

    # 4. ЗАПУСК ОБРАБОТКИ
    start_time = time.time()
    processed_count = 0
    skipped_count = 0
    error_count = 0
    empty_mask_count = 0 # Сколько раз YOLO ничего не нашла

    print("\n🚀 Поехали! (Ctrl+C для остановки)\n")

    try:
        for img_path in tqdm(all_files, desc="Processing", unit="img"):

            # --- Пути сохранения (зеркалируем структуру папок) ---
            relative_path = img_path.relative_to(config.INPUT_DIR)
            save_path = config.OUTPUT_DIR / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # --- Идемпотентность (Пропуск готового) ---
            if save_path.exists():
                skipped_count += 1
                continue

            try:
                # --- ШАГ 0: Открытие ---
                with Image.open(img_path) as img:
                    # LaMa требует RGB
                    original_image = img.convert("RGB")

                # --- ШАГ 1: Детекция + Сегментация (YOLO-Seg) ---
                # Теперь мы просим детектор сразу дать нам МАСКУ (PIL Image)
                # Он внутри себя прогонит нейросеть, соберет полигоны и нарисует Ч/Б картинку
                mask = detector.get_mask(original_image)

                # --- ШАГ 2: Очистка (LaMa) ---
                # Если маска черная (bbox is None), cleaner вернет оригинал моментально
                result_image = cleaner.clean(original_image, mask)

                # Статистика: нашла ли YOLO что-то?
                if not mask.getbbox():
                    empty_mask_count += 1

                # --- ШАГ 3: Сохранение ---
                result_image.save(save_path, quality=95)
                processed_count += 1

            except Exception as e:
                logger.error(f"❌ Ошибка на {img_path.name}: {e}")
                error_count += 1
                # Пишем в лог ошибок
                with open(config.LOG_DIR / "failed_files.txt", "a") as f:
                    f.write(f"{img_path}\n")

    except KeyboardInterrupt:
        logger.warning("\n🛑 Остановлено пользователем.")

    # 5. ИТОГИ
    elapsed = time.time() - start_time
    logger.info("=" * 40)
    logger.info(f"⏱  Время выполнения: {elapsed:.2f} сек")
    logger.info(f"✅ Готово: {processed_count}")
    logger.info(f"👻 Пустых (не найдено): {empty_mask_count}")
    logger.info(f"⏭  Пропущено (было): {skipped_count}")
    logger.info(f"❌ Ошибок: {error_count}")

    if processed_count > 0:
        logger.info(f"🚀 Скорость: {elapsed / processed_count:.3f} сек/фото")
        logger.info(f"🏎  FPS: {processed_count / elapsed:.1f}")

if __name__ == "__main__":
    main()