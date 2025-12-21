"""Комплексная проверка качества очистки (End-to-End Quality Check).

Скрипт запускает полный пайплайн обработки: Детекция -> Генерация маски -> Очистка (LaMa).
Результат сохраняется в виде коллажей-триптихов (Оригинал | Ч/Б Маска | Результат).
Используется для оценки финального качества восстановления фона и настройки параметра
расширения маски (dilation).
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import shutil
from PIL import Image
import config
from core.detector import YourClassDetector
from core.cleaner import ImageInpainter

TEST_OUTPUT_DIR = Path("bench_tests/step2_quality_check")


def run():
    if TEST_OUTPUT_DIR.exists(): shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True)

    print("⏳ Загрузка моделей...")
    detector = YourClassDetector()
    cleaner = ImageInpainter()

    files = list(config.INPUT_DIR.glob("*.*"))
    files = [f for f in files if f.suffix.lower() in {'.jpg', '.png'}][:10]

    print(f"🧼 Проверка качества на {len(files)} фото...")

    for img_path in files:
        try:
            with Image.open(img_path) as img:
                original = img.convert("RGB")
                w, h = original.size

                # 1. Маска
                mask = detector.get_mask(original)

                # 2. Очистка
                if mask.getbbox():
                    cleaned = cleaner.clean(original, mask)
                else:
                    cleaned = original  # Если не нашли, оставляем как есть

                # 3. Коллаж (Триптих)
                collage = Image.new("RGB", (w * 3, h))
                collage.paste(original, (0, 0))  # Слева: Было
                collage.paste(mask.convert("RGB"), (w, 0))  # Центр: Маска
                collage.paste(cleaned, (w * 2, 0))  # Справа: Стало

                collage.save(TEST_OUTPUT_DIR / f"RESULT_{img_path.name}")
                print(f"✅ {img_path.name}")

        except Exception as e:
            print(f"❌ {e}")

    print(f"\n📂 Смотри качество здесь: {TEST_OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    run()