"""
REST API Сервер для удаления ватермарок.
Запуск: uvicorn 4_server_api:app --host 0.0.0.0 --port 8000
"""

import io
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image

import config
from core.detector import YourClassDetector
from core.cleaner import ImageInpainter

# Настройка логгера
logging.basicConfig(level=logging.WARNING) # WARNING чтобы не спамил INFO сообщениями
logger = logging.getLogger("API")
logger.setLevel(logging.INFO)

# Глобальные переменные
models = {}
gpu_lock = asyncio.Lock() # "Светофор" для видеокарты

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте сервера"""
    logger.info("🚀 API Server Starting...")
    logger.info(f"   Device: {config.DEVICE}")
    
    try:
        # 1. Загрузка
        models["detector"] = YourClassDetector()
        models["cleaner"] = ImageInpainter()
        
        # 2. Прогрев (Warmup)
        # Прогоняем фейк, чтобы прогрузить CUDA контекст
        logger.info("🌡️ Warming up GPU...")
        dummy = Image.new("RGB", (640, 640), (128, 128, 128))
        async with gpu_lock:
            m = models["detector"].get_mask(dummy)
            if m.getbbox(): models["cleaner"].clean(dummy, m)
            
        logger.info("✅ SERVER READY! Listening on port 8000.")
        
    except Exception as e:
        logger.critical(f"🔥 Startup Failed: {e}")
        raise e
        
    yield
    
    models.clear()
    logger.info("🛑 Server Stopped.")

app = FastAPI(lifespan=lifespan)

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    # 1. Валидация типа файла
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only jpg/png/webp.")

    try:
        # 2. Чтение (RAM)
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="File is not a valid image.")

        # 3. Обработка (с блокировкой GPU)
        processing_status = "skipped"
        result_image = image

        # Ждем очереди на GPU
        async with gpu_lock:
            mask = models["detector"].get_mask(image)
            
            if mask.getbbox():
                result_image = models["cleaner"].clean(image, mask)
                processing_status = "cleaned"
            
        # 4. Ответ
        img_byte_arr = io.BytesIO()
        # Сохраняем формат (если JPEG - то JPEG, иначе PNG)
        fmt = "JPEG" if file.content_type == "image/jpeg" else "PNG"
        result_image.save(img_byte_arr, format=fmt, quality=95)
        
        return Response(
            content=img_byte_arr.getvalue(), 
            media_type=file.content_type,
            headers={"Clean-Status": processing_status}
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Internal Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    # Запуск: workers=1, так как у нас 1 GPU и мы держим модели в памяти
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")