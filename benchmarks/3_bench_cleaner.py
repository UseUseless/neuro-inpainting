import shutil
from pathlib import Path
from PIL import Image
import config
from core.detector import WatermarkDetector
from core.segmenter import MaskRefiner
from core.cleaner import ImageInpainter

TEST_OUTPUT_DIR = Path("tests/step3_cleaning")


def test_cleaner():
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True)

    print("⏳ Загрузка ВСЕХ моделей...")
    detector = WatermarkDetector()
    refiner = MaskRefiner()
    cleaner = ImageInpainter()

    files = [f for f in config.INPUT_DIR.glob("*.*") if f.suffix.lower() in {'.jpg', '.png', '.jpeg', '.webp'}]

    for img_path in files:
        try:
            with Image.open(img_path) as img:
                original = img.convert("RGB")

                # 1. Detect
                boxes = detector.detect(original)
                if not boxes:
                    continue

                # 2. Segment
                mask = refiner.create_mask(original, boxes)

                # 3. Clean
                cleaned = cleaner.clean(original, mask)

                # 4. Создаем коллаж (Триптих)
                # Ширина = 3 картинки, Высота = 1 картинка
                w, h = original.size
                collage = Image.new("RGB", (w * 3, h))

                # Вставляем оригинал
                collage.paste(original, (0, 0))

                # Вставляем маску (превращаем её в RGB, чтобы было видно)
                collage.paste(mask.convert("RGB"), (w, 0))

                # Вставляем результат
                collage.paste(cleaned, (w * 2, 0))

                collage.save(TEST_OUTPUT_DIR / f"result_{img_path.name}")
                print(f"✅ Processed: {img_path.name}")

        except Exception as e:
            print(f"❌ Error {img_path.name}: {e}")

    print(f"\n📂 Коллажи здесь: {TEST_OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    test_cleaner()