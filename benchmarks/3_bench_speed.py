"""
Бенчмарк производительности пайплайна.
"""

import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import config

def benchmark_speed():
    # Проверка перед импортом тяжелых либ
    if not config.YOLO_MODEL_PATH.exists():
        print("\n" + "!"*50)
        print(f"❌ ОШИБКА: Не найден файл модели {config.YOLO_MODEL_PATH.name}")
        print(f"   Путь: {config.YOLO_MODEL_PATH}")
        print("   👉 Сначала обучи модель (2_train_model.py)")
        print("   👉 И скопируй best.pt из runs/.../weights/ в models/best.pt")
        print("!"*50 + "\n")
        return

    # Импортируем core только если модель на месте
    try:
        from core.detector import YourClassDetector
        from core.cleaner import ImageInpainter
    except Exception as e:
        print(f"❌ Ошибка импорта Core: {e}")
        return

    TEST_OUTPUT_DIR = Path("bench_tests/step3_speed_test")
    REPORT_FILE = TEST_OUTPUT_DIR / "benchmark_report.txt"
    
    if TEST_OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True)

    def log(msg):
        print(msg)
        with open(REPORT_FILE, "a", encoding="utf-8-sig") as f:
            f.write(msg + "\n")

    log(f"🔥 ЗАПУСК БЕНЧМАРКА СКОРОСТИ")
    log(f"   Device: {config.DEVICE}")
    log("-" * 50)

    log("⏳ Инициализация моделей (Cold Start)...")
    try:
        detector = YourClassDetector()
        cleaner = ImageInpainter() # Сама скачает LaMa если нужно
    except Exception as e:
        log(f"❌ Ошибка инициализации: {e}")
        return

    # Warmup (Прогрев GPU)
    log("🌡️  Прогрев (Warmup run)...")
    dummy = Image.new("RGB", (640, 640), (128, 128, 128))
    for _ in range(3):
        m = detector.get_mask(dummy)
        if m.getbbox(): cleaner.clean(dummy, m)

    # Берем фото
    files = list(config.INPUT_DIR.glob("*.*"))
    files = [f for f in files if f.suffix.lower() in {'.jpg', '.png', '.jpeg', '.webp'}][:50]

    if not files:
        log("❌ Папка images_input пуста. Закинь пару фоток для теста.")
        return

    times_seg = []
    times_clean = []
    times_total = []

    log(f"\n🚀 Тестируем на {len(files)} фото...")
    log(f"{'FILE':<20} | {'SEG (ms)':<10} | {'LAMA (ms)':<10} | {'TOTAL':<10}")
    log("-" * 60)

    for img_path in files:
        try:
            with Image.open(img_path) as img:
                original = img.convert("RGB")
            
            # 1. Seg
            t0 = time.perf_counter()
            mask = detector.get_mask(original)
            t1 = time.perf_counter()
            dt_seg = (t1 - t0) * 1000

            # 2. Clean
            dt_clean = 0.0
            if mask.getbbox():
                t2 = time.perf_counter()
                _ = cleaner.clean(original, mask)
                t3 = time.perf_counter()
                dt_clean = (t3 - t2) * 1000

            dt_total = dt_seg + dt_clean

            times_seg.append(dt_seg)
            if dt_clean > 0: times_clean.append(dt_clean)
            times_total.append(dt_total)

            log(f"{img_path.name[:20]:<20} | {dt_seg:6.1f}     | {dt_clean:6.1f}      | {dt_total:6.1f}")

        except Exception as e:
            log(f"❌ Ошибка {img_path.name}: {e}")

    # Статистика
    if times_total:
        avg_total = np.mean(times_total)
        fps = 1000 / avg_total
        est_1k = (avg_total / 1000 * 1000) / 60 # Минут на 1000 фото

        log("\n" + "=" * 50)
        log(f"⚡ СРЕДНЕЕ: {avg_total:.1f} ms/фото")
        log(f"🏎  FPS:     {fps:.1f}")
        log("-" * 50)
        log(f"📅 Прогноз на 1,000 фото: ~{est_1k:.1f} минут")
        log("=" * 50)
        log(f"📄 Отчет: {REPORT_FILE}")

if __name__ == "__main__":
    benchmark_speed()