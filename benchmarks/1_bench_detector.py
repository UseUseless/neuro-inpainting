import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import config
from core.detector import WatermarkDetector

# Папка для сохранения результатов теста
TEST_OUTPUT_DIR = Path("tests/step1_detection")


def test_detector():
    # 1. Подготовка папок
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True)

    print(f"🕵️ Загружаем детектор...")
    try:
        detector = WatermarkDetector()
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    # Берем фото из входной папки
    files = list(config.INPUT_DIR.glob("*.*"))
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    files = [f for f in files if f.suffix.lower() in valid_extensions]

    if not files:
        print(f"❌ Папка {config.INPUT_DIR} пуста! Положи туда 10-20 фото.")
        return

    print(f"📸 Найдено {len(files)} фото. Начинаем проверку...")

    for img_path in files:
        try:
            with Image.open(img_path) as img:
                original = img.convert("RGB")
                draw = ImageDraw.Draw(original)

                # === Детекция ===
                boxes = detector.detect(original)

                if not boxes:
                    print(f"⚠️ ПУСТО: {img_path.name} (Ватермарка не найдена)")
                    # Сохраняем с пометкой MISSED, чтобы ты обратил внимание
                    original.save(TEST_OUTPUT_DIR / f"MISSED_{img_path.name}")
                    continue

                img_w, img_h = original.size
                pad = config.BOX_PADDING

                for (x1, y1, x2, y2) in boxes:
                    # 1. СИНЯЯ РАМКА (То, что нашла YOLO)
                    draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)

                    # 2. КРАСНАЯ РАМКА (То, что увидит SAM - с отступом)
                    nx1 = max(0, x1 - pad)
                    ny1 = max(0, y1 - pad)
                    nx2 = min(img_w, x2 + pad)
                    ny2 = min(img_h, y2 + pad)

                    draw.rectangle([nx1, ny1, nx2, ny2], outline="red", width=3)

                # Сохраняем результат
                original.save(TEST_OUTPUT_DIR / f"checked_{img_path.name}")
                print(f"✅ OK: {img_path.name}")

        except Exception as e:
            print(f"❌ Ошибка на файле {img_path.name}: {e}")

    print(f"\n🏁 Тест завершен! Результаты здесь: {TEST_OUTPUT_DIR.absolute()}")
    print("Смотри на КРАСНЫЕ рамки. Они должны полностью охватывать ватермарку с запасом.")


if __name__ == "__main__":
    test_detector()