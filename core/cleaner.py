"""
Модуль Очистки (Inpainting).
Заполняет вырезанные области (маску) сгенерированным фоном.

Использует LaMa (Large Mask Inpainting). Это SOTA (State of the Art) решение
для восстановления фона. Оно работает намного лучше, чем старые методы (Telea, Navier-Stokes).
"""

import logging
from PIL import Image
from simple_lama_inpainting import SimpleLama
import config

class ImageInpainter:
    """
    Класс-обертка для LaMa.
    Принимает изображение и маску, возвращает чистое изображение.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("⏳ Инициализация LaMa Inpainting...")

        try:
            # SimpleLama сама скачает веса (big-lama.pt) при первом запуске.
            # Она автоматически найдет GPU, если установлен onnxruntime-gpu.
            self.model = SimpleLama()
            self.logger.info("✅ LaMa успешно загружена.")
        except Exception as e:
            self.logger.critical(f"❌ Фатальная ошибка при загрузке LaMa: {e}")
            self.logger.critical("   -> Проверь интернет (для скачивания весов) и версию onnxruntime.")
            raise e

    def clean(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Удаляет объекты с изображения по маске.

        Args:
            image: Исходное цветное изображение (PIL RGB).
            mask: Черно-белая маска (PIL L), где белое = область удаления.

        Returns:
            Очищенное изображение (PIL RGB).
        """
        # 1. ОПТИМИЗАЦИЯ: Если маска черная (пустая), ничего не делаем.
        # Метод getbbox() возвращает None, если картинка полностью черная.
        if not mask.getbbox():
            # self.logger.debug("Маска пуста. Возвращаем оригинал.")
            return image

        try:
            # 2. Запуск нейросети
            # LaMa принимает PIL Images и возвращает PIL Image.
            # Внутри simple_lama уже есть препроцессинг (ресайз и паддинг),
            # поэтому можно скармливать картинки любого размера.
            result = self.model(image, mask)

            return result

        except Exception as e:
            self.logger.error(f"❌ Ошибка во время очистки (LaMa): {e}")
            # В случае сбоя лучше вернуть оригинал (с ватермаркой),
            # чем крашить весь пайплайн или возвращать черный квадрат.
            return image

# Тест модуля
if __name__ == "__main__":
    try:
        cleaner = ImageInpainter()
        print("Тест инициализации пройден.")
    except Exception as e:
        print(f"Ошибка: {e}")