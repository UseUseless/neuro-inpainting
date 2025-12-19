import sys
import gc
from pathlib import Path

# Добавляем путь к корню
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import shutil
import torch
from PIL import Image, ImageDraw, ImageFont
import config
from core.detector import YourClassDetector

TEST_OUTPUT_DIR = Path("bench_tests/step1_detection")

# === НАСТРОЙКА ЦВЕТОВ ===
# 0: Синий (обычно текст)
# 1: Красный (обычно лого)
COLOR_MAP = {
    0: "blue",
    1: "red"
}
DEFAULT_COLOR = "green"


def test_detector_final():
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True)

    print(f"🕵️ Загружаем детектор...", flush=True)
    try:
        detector = YourClassDetector()
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return

    files = list(config.INPUT_DIR.glob("*.*"))
    files = [f for f in files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]

    if not files:
        print(f"❌ Папка пуста!", flush=True)
        return

    try:
        font = ImageFont.load_default(size=18)
    except:
        font = ImageFont.load_default()

    print(f"📸 Проверка {len(files)} фото...", flush=True)

    for i, img_path in enumerate(files):
        # Чистим VRAM
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print(f"[{i + 1}/{len(files)}] {img_path.name}...", end=" ", flush=True)

        try:
            with Image.open(img_path) as img:
                original = img.convert("RGB")
                draw = ImageDraw.Draw(original)

                detections = detector.detect(original)

                if not detections:
                    print(f"⚠️ EMPTY", flush=True)
                    original.save(TEST_OUTPUT_DIR / f"MISSED_{img_path.name}")
                    continue

                img_w, img_h = original.size

                for det in detections:
                    x1, y1, x2, y2, conf, cls_id = det

                    # 1. Определяем ЦВЕТ по ID класса
                    color = COLOR_MAP.get(cls_id, DEFAULT_COLOR)

                    # 2. Получаем инфо из конфига (Имя и Стратегия)
                    params = config.CLASS_PARAMS.get(cls_id, config.DEFAULT_PARAMS)
                    class_name = params.get('name', f"id_{cls_id}")
                    strategy = params.get('strategy', 'UNK')
                    pad = params.get('padding', 0)

                    # --- ОТРИСОВКА ---

                    # Рамка YOLO (Жирная)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                    # Рамка PADDING (Тонкая, показывает зону захвата)
                    nx1 = max(0, x1 - pad)
                    ny1 = max(0, y1 - pad)
                    nx2 = min(img_w, x2 + pad)
                    ny2 = min(img_h, y2 + pad)
                    draw.rectangle([nx1, ny1, nx2, ny2], outline=color, width=1)

                    # ТЕКСТ: "text (BOX) 0.95"
                    label = f"{class_name} ({strategy}) {conf:.2f}"

                    # Подложка под текст
                    bbox = font.getbbox(label)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]

                    # Позиция текста (над рамкой)
                    bg_x1 = x1
                    bg_y1 = y1 - text_h - 6
                    bg_x2 = x1 + text_w + 8
                    bg_y2 = y1

                    # Если вылезает за верх - рисуем внутри
                    if bg_y1 < 0:
                        bg_y1 = y1
                        bg_y2 = y1 + text_h + 6

                    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)
                    draw.text((bg_x1 + 4, bg_y1 - 2), label, fill="white", font=font)

                original.save(TEST_OUTPUT_DIR / f"checked_{img_path.name}")
                print("✅ OK", flush=True)

        except Exception as e:
            print(f"\n❌ ОШИБКА: {e}", flush=True)

    print(f"\n🏁 Результаты: {TEST_OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    test_detector_final()