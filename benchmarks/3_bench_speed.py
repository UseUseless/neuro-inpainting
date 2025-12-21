import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Добавляем корень проекта
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import config
from core.detector import YourClassDetector
from core.cleaner import ImageInpainter

# Папка для отчета
TEST_OUTPUT_DIR = Path("bench_tests/step3_speed_test")
REPORT_FILE = TEST_OUTPUT_DIR / "benchmark_report.txt"


def benchmark_speed():
    # 1. Подготовка папки
    if TEST_OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True)

    # 2. Логгер в файл и в консоль
    def log(msg):
        print(msg)
        # !!! ИСПРАВЛЕНИЕ: utf-8-sig чтобы Windows открывала нормально !!!
        with open(REPORT_FILE, "a", encoding="utf-8-sig") as f:
            f.write(msg + "\n")

    log(f"🔥 ЗАПУСК БЕНЧМАРКА СКОРОСТИ")
    log(f"   Device: {config.DEVICE}")
    log("-" * 50)

    log("⏳ Загрузка и прогрев моделей...")
    try:
        detector = YourClassDetector()
        cleaner = ImageInpainter()

        # Прогрев (Warmup)
        dummy = Image.new("RGB", (640, 640), (0, 0, 0))
        detector.get_mask(dummy)
        log("✅ Модели прогреты.")
    except Exception as e:
        log(f"❌ Ошибка загрузки: {e}")
        return

    # Берем 50 фото
    files = list(config.INPUT_DIR.glob("*.*"))
    files = [f for f in files if f.suffix.lower() in {'.jpg', '.png', '.jpeg', '.webp'}][:50]

    if not files:
        log("❌ Нет фото для теста.")
        return

    # Списки для статистики
    times_seg = []
    times_clean = []
    times_total = []

    log(f"🚀 Тестируем на {len(files)} фото...\n")

    # Заголовок
    header = f"{'FILE':<20} | {'SEG (YOLO)':<12} | {'CLEAN (LaMa)':<12} | {'TOTAL':<12}"
    log(header)
    log("-" * 65)

    for i, img_path in enumerate(files):
        with Image.open(img_path) as img:
            original = img.convert("RGB")

        # --- ЗАМЕР YOLO ---
        t0 = time.perf_counter()
        mask = detector.get_mask(original)
        t1 = time.perf_counter()

        dt_seg = (t1 - t0) * 1000  # мс

        # --- ЗАМЕР LaMa ---
        dt_clean = 0.0
        if mask.getbbox():
            t2 = time.perf_counter()
            _ = cleaner.clean(original, mask)
            t3 = time.perf_counter()
            dt_clean = (t3 - t2) * 1000  # мс

        # --- ИТОГ ---
        dt_total = dt_seg + dt_clean

        times_seg.append(dt_seg)
        if dt_clean > 0:
            times_clean.append(dt_clean)
        times_total.append(dt_total)

        log(f"{img_path.name[:20]:<20} | {dt_seg:6.1f} ms   | {dt_clean:6.1f} ms   | {dt_total:6.1f} ms")

    # === ИТОГИ ===
    if times_total:
        avg_seg = np.mean(times_seg)
        avg_clean = np.mean(times_clean) if times_clean else 0.0
        avg_total = np.mean(times_total)

        fps = 1000 / avg_total if avg_total > 0 else 0

        est_hours = (avg_total / 1000 * 300000) / 3600
        est_days = est_hours / 24

        log("\n" + "=" * 50)
        log("📊 ИТОГОВЫЙ ОТЧЕТ")
        log("=" * 50)
        log(f"👁️  Сегментация (YOLO):  {avg_seg:.1f} ms  (вклад: {avg_seg / avg_total * 100:.1f}%)")
        log(f"🧼  Очистка (LaMa):      {avg_clean:.1f} ms  (вклад: {avg_clean / avg_total * 100:.1f}%)")
        log("-" * 50)
        log(f"⚡ СРЕДНЕЕ ВРЕМЯ:       {avg_total:.1f} ms / фото")
        log(f"🏎  FPS (Скорость):      {fps:.1f} кадров/сек")
        log("=" * 50)
        log(f"📅 Прогноз на 300,000 фото:")
        log(f"   ⏱  {est_hours:.1f} часов")
        log(f"   📆  {est_days:.1f} дней (non-stop)")
        log("=" * 50)
        log(f"📄 Отчет сохранен в: {REPORT_FILE}")


if __name__ == "__main__":
    benchmark_speed()