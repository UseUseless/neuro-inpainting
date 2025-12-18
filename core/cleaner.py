"""
Удаляет по маске (LaMa) после сегментации

Загружает LaMa - модель для изменения изображения.

Принимает картинку + маску (после сегментации) -> Отдает чистое изображение без следов удаления.
"""

import logging
from PIL import Image
from simple_lama_inpainting import SimpleLama


class ImageInpainter:
    """
    Отвечает за удаление твоего класса (Inpainting) с помощью LaMa.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading LaMa model...")

        # SimpleLama сама скачает веса при первом запуске
        # Она автоматически использует GPU, если установлен onnxruntime-gpu
        # (или CPU, если нет, но это тоже работает достаточно быстро)
        try:
            self.model = SimpleLama()
            self.logger.info("LaMa model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load LaMa: {e}")
            raise e

    def clean(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Принимает оригинал и ч/б маску.
        Возвращает очищенное изображение.
        """
        # 1. Быстрая проверка: если маска полностью черная (твоего класса нет),
        # то не тратим время на нейросеть, возвращаем оригинал.
        if not mask.getbbox():
            self.logger.debug("Empty mask provided. Skipping inpainting.")
            return image

        try:
            # 2. Запуск удаления
            # LaMa ожидает PIL Images
            result = self.model(image, mask)

            return result

        except Exception as e:
            self.logger.error(f"Error during inpainting: {e}")
            # В случае ошибки возвращаем оригинал, чтобы не крашить программу
            return image


if __name__ == "__main__":
    try:
        cleaner = ImageInpainter()
        print("✅ Cleaner initialized.")
    except Exception as e:
        print(e)