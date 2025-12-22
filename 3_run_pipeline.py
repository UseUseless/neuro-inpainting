"""
PRODUCTION PIPELINE

ВХОД:  images_input/
ВЫХОД: images_cleaned/
АРХИВ: images_input/processed/
"""

import time
import shutil
import logging
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, wait
from tqdm import tqdm

import config
from core.pipeline_logger import setup_logger
from core.detector import YourClassDetector
from core.cleaner import ImageInpainter

# === КОНФИГ ===
SLEEP_ON_EMPTY = 60      # Секунд сна
IO_THREADS = 4           # Потоки записи

# === ПУТИ ===
DIR_RESULT_CLEAN = config.OUTPUT_DIR
DIR_RESULT_SKIPPED = config.OUTPUT_DIR / "skipped"
DIR_SOURCE_ARCHIVE = config.INPUT_DIR / "processed"

logger = setup_logger()

def setup_structure():
    for d in [DIR_RESULT_CLEAN, DIR_RESULT_SKIPPED, DIR_SOURCE_ARCHIVE]:
        d.mkdir(parents=True, exist_ok=True)

def save_and_move_worker(img_result, save_path_result, img_source_path):
    """
    Фоновая задача: Сохранение + Перемещение.
    """
    try:
        # 1. Сохранение результата
        if img_result and save_path_result:
            img_result.save(save_path_result, quality=95, optimize=True)

        # 2. Перемещение исходника
        if img_source_path.exists():
            dest_source = DIR_SOURCE_ARCHIVE / img_source_path.name
            if dest_source.exists():
                dest_source.unlink() # Удаляем старый файл в архиве, если есть

            # Используем shutil.move с обработкой ошибок
            shutil.move(str(img_source_path), str(dest_source))
        else:
            # Если файла нет - скорее всего мы его уже переместили, это не страшно
            pass

    except Exception as e:
        # Логируем, но не крашим поток
        logger.error(f"⚠️ Ошибка I/O {img_source_path.name}: {e}")

def print_summary(start_time, total_count, skipped_count):
    """Красивый вывод итогов, как ты любишь"""
    elapsed = time.time() - start_time
    processed_count = total_count - skipped_count

    print("\n" + "=" * 40)
    print(f"⏱  Время пачки:   {elapsed:.2f} сек")
    print(f"✅ Обработано:    {processed_count}")
    print(f"⏭  Пропущено:     {skipped_count}")
    print(f"📦 Всего файлов:  {total_count}")

    if total_count > 0:
        fps = total_count / elapsed
        print(f"🚀 Скорость:      {elapsed / total_count:.3f} сек/фото")
        print(f"🏎  FPS:           {fps:.1f}")
    print("=" * 40 + "\n")

def main():
    setup_structure()

    print("\n🚀 ЗАПУСК WATCHDOG PIPELINE")
    print(f"📂 Слежу за папкой: {config.INPUT_DIR}")
    print("⏳ Загрузка нейросетей... (подожди пару секунд)")

    try:
        detector = YourClassDetector()
        cleaner = ImageInpainter()
        logger.info("✅ Модели загружены и готовы.")
    except Exception as e:
        logger.critical(f"🔥 Ошибка запуска: {e}")
        return

    io_executor = ThreadPoolExecutor(max_workers=IO_THREADS)

    # Флаг, чтобы не спамить "Waiting..." каждую секунду
    is_waiting_message_shown = False

    try:
        while True:
            # 1. Поиск файлов (Скан)
            candidates = []
            try:
                # Фильтруем файлы, игнорируем папки
                candidates = [
                    f for f in config.INPUT_DIR.iterdir()
                    if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                ]
            except Exception:
                pass # Если папка заблокирована виндой, пробуем позже

            # Если пусто
            if not candidates:
                if not is_waiting_message_shown:
                    print(f"💤 Папка пуста. Жду новые фото... (Проверка каждые {SLEEP_ON_EMPTY}с)")
                    is_waiting_message_shown = True

                time.sleep(SLEEP_ON_EMPTY)
                continue

            # Если нашли файлы - сбрасываем флаг ожидания
            is_waiting_message_shown = False

            # Сортировка и Старт батча
            candidates.sort()
            batch_total = len(candidates)
            logger.info(f"⚡ Новая пачка: {batch_total} фото.")

            batch_start_time = time.time()
            skipped_in_batch = 0

            # Список задач для I/O (чтобы дождаться их завершения перед итогами)
            io_futures = []

            # === PROGRESS BAR (tqdm) ===
            # desc="Processing" - текст слева
            # unit="img" - ед. измерения
            # leave=False - полоска исчезнет после завершения (чтобы не засорять консоль)
            pbar = tqdm(candidates, desc="Processing", unit="img", leave=True)

            for img_path in pbar:
                try:
                    # ШАГ 1: Загрузка
                    with Image.open(img_path) as img:
                        original = img.convert("RGB")
                        original.load()

                    # ШАГ 2: GPU Inference
                    mask = detector.get_mask(original)

                    if mask.getbbox():
                        # Нашли -> Чистим
                        result = cleaner.clean(original, mask)
                        save_to = DIR_RESULT_CLEAN / img_path.name
                    else:
                        # Пусто -> Скип
                        result = original
                        save_to = DIR_RESULT_SKIPPED / img_path.name
                        skipped_in_batch += 1

                    # ШАГ 3: Async Save
                    future = io_executor.submit(
                        save_and_move_worker,
                        result.copy(),
                        save_to,
                        img_path
                    )
                    io_futures.append(future)

                except Exception as e:
                    logger.error(f"❌ Ошибка {img_path.name}: {e}")
                    # При ошибке тоже пытаемся убрать файл, чтобы не виснуть
                    # (можно раскомментить перемещение в errors)

            pbar.close()

            # Ждем, пока все файлы реально допишутся на диск и переместятся
            # Это решит проблему с WinError и даст честное время выполнения
            wait(io_futures)

            # Выводим красивую табличку итогов
            print_summary(batch_start_time, batch_total, skipped_in_batch)

            # После таблички скрипт сразу пойдет искать новую пачку
            # Если там пусто - уйдет в сон

    except KeyboardInterrupt:
        print("\n🛑 Останавливаемся... Дописываем файлы...")
        io_executor.shutdown(wait=True)
        print("✅ Всё сохранено. Пока!")

if __name__ == "__main__":
    main()