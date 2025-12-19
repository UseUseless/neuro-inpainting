import sys
from pathlib import Path
import time
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from PIL import Image
import config
from core.detector import YourClassDetector
from core.segmenter import MaskRefiner
from core.cleaner import ImageInpainter


def benchmark():
    print("⏳ Warming up models...")
    detector = YourClassDetector()
    refiner = MaskRefiner()
    cleaner = ImageInpainter()

    files = [f for f in config.INPUT_DIR.glob("*.*") if f.suffix.lower() in {'.jpg', '.png'}]
    if not files: return

    times = []
    print(f"🚀 Запуск теста на {len(files)} фото...")

    for i, img_path in enumerate(files):
        # Открываем вне таймера
        with Image.open(img_path) as img:
            original = img.convert("RGB")

        t0 = time.perf_counter()

        # Pipeline
        dets = detector.detect(original)
        if dets:
            mask = refiner.create_mask(original, dets)
            _ = cleaner.clean(original, mask)

        t1 = time.perf_counter()

        if dets:  # Считаем только те, где была работа
            dt = (t1 - t0) * 1000
            times.append(dt)
            print(f"Img {i + 1}: {dt:.0f} ms")

    if times:
        avg = np.mean(times)
        print("\n" + "=" * 30)
        print(f"⚡ Среднее время: {avg:.1f} ms")
        print(f"🚀 Скорость: {1000 / avg:.1f} FPS")
        print(f"📅 100k фото займет: {avg * 100000 / 1000 / 3600:.1f} часов")
        print("=" * 30)


if __name__ == "__main__":
    benchmark()