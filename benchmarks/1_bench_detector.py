import sys
from pathlib import Path

# Добавляем корень проекта в путь, чтобы видеть модули core и config
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import shutil
from PIL import Image
import config
from core.detector import YourClassDetector

TEST_OUTPUT_DIR = Path("bench_tests/step1_detector_check")

def run():
    # Чистим папку результатов
    if TEST_OUTPUT_DIR.exists(): shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True)

    print("🕵️ Загружаем YOLO-Seg...")
    try:
        detector = YourClassDetector()
    except Exception as e:
        print(f"❌ Не могу загрузить детектор: {e}")
        return

    # Берем первые 10 фото из входной папки
    files = list(config.INPUT_DIR.glob("*.*"))
    files = [f for f in files if f.suffix.lower() in {'.jpg', '.png', '.jpeg'}][:10]

    if not files:
        print("❌ Папка images_input пуста.")
        return

    print(f"📸 Обработка {len(files)} фото...")

    for img_path in files:
        try:
            with Image.open(img_path) as img:
                original = img.convert("RGB")
                w, h = original.size

                # 1. Получаем маску
                mask = detector.get_mask(original)

                # Если маска пустая - сохраняем как MISSED
                if not mask.getbbox():
                    print(f"⚠️ {img_path.name}: Не найдено!")
                    original.save(TEST_OUTPUT_DIR / f"MISSED_{img_path.name}")
                    continue

                # 2. Визуализация (Красная заливка)
                # Создаем красный слой
                red = Image.new("RGB", (w, h), (255, 0, 0))
                # Накладываем его там, где маска белая
                overlay = Image.composite(red, original, mask)
                # Смешиваем с оригиналом (50%), чтобы видеть фон
                final = Image.blend(original, overlay, 0.5)

                final.save(TEST_OUTPUT_DIR / f"OK_{img_path.name}")
                print(f"✅ {img_path.name}")

        except Exception as e:
            print(f"❌ Ошибка {img_path.name}: {e}")

    print(f"\n📂 Результаты в: {TEST_OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    run()