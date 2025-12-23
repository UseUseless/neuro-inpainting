"""
Стресс-тест API клиента.
Имитирует реальную нагрузку: 20 одновременных запросов.
"""

import asyncio
import aiohttp
import time
import os
from pathlib import Path
from tqdm.asyncio import tqdm

# === НАСТРОЙКИ ===
SERVER_URL = "http://localhost:8000/process"
INPUT_DIR = Path("images_input")
TEST_OUTPUT_DIR = Path("bench_tests/step4_api_check")
CONCURRENCY_LIMIT = 20  # Сколько запросов висит одновременно

async def process_file(session, file_path, semaphore):
    # Ограничиваем кол-во одновременных запросов семафором
    async with semaphore:
        try:
            # Читаем файл
            with open(file_path, "rb") as f:
                file_data = f.read()

            # Отправляем POST
            data = aiohttp.FormData()
            data.add_field('file', file_data, filename=file_path.name, content_type='image/jpeg')

            async with session.post(SERVER_URL, data=data) as response:
                if response.status == 200:
                    content = await response.read()
                    status = response.headers.get("Clean-Status", "unknown")
                    
                    # Сохраняем результат
                    save_path = TEST_OUTPUT_DIR / file_path.name
                    with open(save_path, "wb") as f_out:
                        f_out.write(content)
                    
                    return status
                else:
                    return f"Error {response.status}"
                    
        except Exception as e:
            return f"Exception: {str(e)}"

async def main():
    # Подготовка
    if not INPUT_DIR.exists():
        print(f"❌ Нет папки {INPUT_DIR}")
        return
    
    if TEST_OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True)

    # Собираем список файлов
    files = list(INPUT_DIR.glob("*.*"))
    files = [f for f in files if f.suffix.lower() in {'.jpg', '.png', '.jpeg'}]
    
    if not files:
        print("❌ Нет файлов для теста.")
        return

    print(f"🚀 Запуск теста API Клиента")
    print(f"📂 Файлов: {len(files)}")
    print(f"⚡ Потоков: {CONCURRENCY_LIMIT}")
    print("-" * 40)

    # Семафор (тот самый лимитер на клиенте)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.time()
        
        # Создаем задачи
        for f in files:
            task = process_file(session, f, semaphore)
            tasks.append(task)

        # Запускаем с прогресс-баром
        results = await tqdm.gather(*tasks)
        
        total_time = time.time() - start_time

    # Статистика
    cleaned = results.count("cleaned")
    skipped = results.count("skipped")
    errors = len(results) - cleaned - skipped
    
    fps = len(files) / total_time

    print("\n" + "="*40)
    print(f"⏱  Время:    {total_time:.2f} сек")
    print(f"🏎  FPS:      {fps:.2f} фото/сек")
    print("-" * 40)
    print(f"✅ Cleaned:  {cleaned}")
    print(f"⏭  Skipped:  {skipped}")
    print(f"❌ Errors:   {errors}")
    print("="*40)

if __name__ == "__main__":
    # Windows hack для asyncio
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())