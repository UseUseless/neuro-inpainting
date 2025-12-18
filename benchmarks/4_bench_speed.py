import time
import numpy as np
from PIL import Image
import config
from core.detector import WatermarkDetector
from core.segmenter import MaskRefiner
from core.cleaner import ImageInpainter


def benchmark():
    print("⏳ Warming up models (Загрузка в память)...")
    # Замеряем время загрузки моделей
    t_load_start = time.perf_counter()
    detector = WatermarkDetector()
    refiner = MaskRefiner()
    cleaner = ImageInpainter()
    t_load_end = time.perf_counter()
    print(f"✅ Модели загружены за {t_load_end - t_load_start:.2f} сек.")

    files = [f for f in config.INPUT_DIR.glob("*.*") if f.suffix.lower() in {'.jpg', '.png'}]
    if not files:
        print("Нет фото для теста.")
        return

    print(f"🚀 Старт теста на {len(files)} изображениях...\n")

    # Списки для хранения времени (в миллисекундах)
    times_detect = []
    times_segment = []
    times_clean = []
    times_total = []

    for i, img_path in enumerate(files):
        # Открываем вне таймера (нас интересует скорость нейросетей, а не диска)
        with Image.open(img_path) as img:
            original = img.convert("RGB")

        # --- 1. Detect ---
        t0 = time.perf_counter()
        boxes = detector.detect(original)
        t1 = time.perf_counter()

        if not boxes:
            print(f"Img {i}: Skipped (No watermark)")
            continue

        # --- 2. Segment ---
        t2 = time.perf_counter()
        mask = refiner.create_mask(original, boxes)
        t3 = time.perf_counter()

        # --- 3. Clean ---
        t4 = time.perf_counter()
        _ = cleaner.clean(original, mask)  # Результат не сохраняем
        t5 = time.perf_counter()

        # Запись результатов (в мс)
        d_time = (t1 - t0) * 1000
        s_time = (t3 - t2) * 1000
        c_time = (t5 - t4) * 1000
        total = d_time + s_time + c_time

        times_detect.append(d_time)
        times_segment.append(s_time)
        times_clean.append(c_time)
        times_total.append(total)

        print(f"Img {i}: Det={d_time:.0f}ms | Seg={s_time:.0f}ms | Clean={c_time:.0f}ms | TOTAL={total:.0f}ms")

    # --- ИТОГИ ---
    print("\n" + "=" * 30)
    print("📊 СРЕДНИЕ ПОКАЗАТЕЛИ (Average Performance)")
    print("=" * 30)
    print(f"👁️  Detection (YOLO):  {np.mean(times_detect):.1f} ms")
    print(f"🎯 Segmentation (SAM): {np.mean(times_segment):.1f} ms")
    print(f"🧼 Cleaning (LaMa):    {np.mean(times_clean):.1f} ms")
    print("-" * 30)
    print(f"⚡ TOTAL PIPELINE:     {np.mean(times_total):.1f} ms  (~ {1000 / np.mean(times_total):.1f} FPS)")
    print("=" * 30)


if __name__ == "__main__":
    benchmark()