import requests
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def ensure_model(file_path: Path, url: str):
    """
    Проверяет наличие файла. Если нет — качает с прогресс-баром.
    """
    if file_path.exists():
        # Можно добавить проверку хэша, но пока просто по наличию
        return

    logger.info(f"📥 Модель не найдена: {file_path.name}")
    logger.info(f"   Скачивание с {url}...")
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(file_path, "wb") as file, tqdm(
            desc=file_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
                
        logger.info("✅ Скачивание завершено.")
        
    except Exception as e:
        logger.critical(f"❌ Ошибка скачивания {file_path.name}: {e}")
        if file_path.exists():
            file_path.unlink() # Удаляем битый файл
        raise e