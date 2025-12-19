import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import shutil
import numpy as np
from PIL import Image, ImageDraw
import config
from core.detector import YourClassDetector
from core.segmenter import MaskRefiner

TEST_OUTPUT_DIR = Path("bench_tests/step2_segmentation")


def test_segmenter_visual():
    if TEST_OUTPUT_DIR.exists(): shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True)

    print("⏳ Загрузка Detector + Segmenter...")
    detector = YourClassDetector()
    refiner = MaskRefiner()

    files = list(config.INPUT_DIR.glob("*.*"))
    files = [f for f in files if f.suffix.lower() in {'.jpg', '.png'}]

    print(f"📸 Генерация масок для {len(files)} фото...")

    for img_path in files:
        with Image.open(img_path) as img:
            original = img.convert("RGB")
            w, h = original.size

            # 1. Detect
            detections = detector.detect(original)
            if not detections: continue

            # 2. Segment
            mask = refiner.create_mask(original, detections)

            # === ВИЗУАЛИЗАЦИЯ ===

            # A. Оригинал с рамками (для сравнения)
            vis_box = original.copy()
            draw = ImageDraw.Draw(vis_box)
            for det in detections:
                x1, y1, x2, y2, _, cls_id = det
                # Цвет рамки зависит от класса
                color = "cyan" if cls_id == 0 else "magenta"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # B. Наложение маски (Красный полупрозрачный слой)
            vis_overlay = original.convert("RGBA")
            # Создаем красную картинку
            red_layer = Image.new("RGBA", (w, h), (255, 0, 0, 120))
            # Используем маску как альфа-канал для красного слоя
            mask_l = mask.convert("L")

            # Накладываем
            vis_overlay.paste(red_layer, (0, 0), mask_l)
            vis_overlay = vis_overlay.convert("RGB")

            # C. Сама маска (ЧБ)
            vis_bw = mask.convert("RGB")

            # Собираем коллаж: Рамки | Наложение | ЧБ Маска
            collage = Image.new("RGB", (w * 3, h))
            collage.paste(vis_box, (0, 0))
            collage.paste(vis_overlay, (w, 0))
            collage.paste(vis_bw, (w * 2, 0))

            collage.save(TEST_OUTPUT_DIR / f"mask_{img_path.name}")
            print(f"✅ {img_path.name}")

    print(f"\n📂 Открой папку и проверь маски: {TEST_OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    test_segmenter_visual()