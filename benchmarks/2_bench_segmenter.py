import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import config
from core.detector import WatermarkDetector
from core.segmenter import MaskRefiner

TEST_OUTPUT_DIR = Path("tests/step2_segmentation")


def test_segmenter():
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True)

    print("⏳ Загрузка моделей (Detector + Segmenter)...")
    detector = WatermarkDetector()
    refiner = MaskRefiner()

    files = [f for f in config.INPUT_DIR.glob("*.*") if f.suffix.lower() in {'.jpg', '.png', '.jpeg', '.webp'}]
    print(f"📸 Обработка {len(files)} фото...")

    for img_path in files:
        try:
            with Image.open(img_path) as img:
                original = img.convert("RGB")

                # 1. Detect
                boxes = detector.detect(original)
                if not boxes:
                    print(f"⚠️ Skip: {img_path.name}")
                    continue

                # 2. Segment (получаем ч/б маску)
                mask = refiner.create_mask(original, boxes)

                # 3. Визуализация (Оверлей)
                # Создаем красную заливку
                red_layer = Image.new("RGB", original.size, (255, 0, 0))

                # Используем маску как альфа-канал для красного слоя
                # Там где маска белая -> будет красное. Где черная -> прозрачно.
                overlay = Image.composite(red_layer, original, mask)

                # Смешиваем оригинал и оверлей (50% прозрачности)
                # Но лучше сделать умнее: наложить красное ТОЛЬКО там где маска
                final_vis = original.convert("RGBA")
                mask_rgba = mask.convert("L")

                # Создаем полупрозрачный красный слой только для маски
                red_overlay = Image.new("RGBA", original.size, (255, 0, 0, 100))  # 100 = прозрачность
                final_vis.paste(red_overlay, (0, 0), mask_rgba)

                # Рисуем еще и рамку для наглядности
                draw = ImageDraw.Draw(final_vis)
                pad = config.BOX_PADDING
                w, h = original.size
                for x1, y1, x2, y2 in boxes:
                    nx1, ny1 = max(0, x1 - pad), max(0, y1 - pad)
                    nx2, ny2 = min(w, x2 + pad), min(h, y2 + pad)
                    draw.rectangle([nx1, ny1, nx2, ny2], outline="blue", width=2)

                final_vis.convert("RGB").save(TEST_OUTPUT_DIR / f"seg_{img_path.name}")
                print(f"✅ Saved: seg_{img_path.name}")

        except Exception as e:
            print(f"❌ Error {img_path.name}: {e}")

    print(f"\n📂 Результаты: {TEST_OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    test_segmenter()