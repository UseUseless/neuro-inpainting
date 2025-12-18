"""
Делает точную маску (SAM) после детектора

Загружает SAM 2 - модель для сегментации.

Принимает картинку + координаты квадрата после детектора
->
Отдает черно-белую маску (PIL Image), где белым закрашен только твой класс.
"""

import logging
import numpy as np
import torch
from PIL import Image
from ultralytics import SAM
import config


class MaskRefiner:
    """
    Отвечает за создание точной маски (Segmentation) на основе коробок от YOLO.
    Использует SAM 2 (Segment Anything Model).
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading SAM model: {config.SAM_MODEL_NAME}...")

        # Загружаем модель (скачается сама при первом запуске)
        self.model = SAM(config.SAM_MODEL_NAME)

    def create_mask(self, image: Image.Image, boxes: list) -> Image.Image:
        """
        Принимает оригинал и список координат коробок [x1, y1, x2, y2].
        Возвращает черно-белую маску (PIL Image), где белое = твой класс.
        """
        # Создаем пустую черную маску размером с картинку
        final_mask = Image.new("L", image.size, 0)  # L = Grayscale (0-255)

        if not boxes:
            return final_mask

        try:
            # SAM 2 в Ultralytics принимает аргумент bboxes для подсказок
            # Это заставляет его искать объекты ТОЛЬКО внутри этих квадратов
            results = self.model(
                source=image,
                bboxes=boxes,
                device=config.DEVICE,
                verbose=False,
                retina_masks=True  # Высокое качество масок
            )

            # Собираем результаты
            for i, result in enumerate(results):
                if result.masks is not None:
                    # masks.data - это тензоры на GPU
                    # Нам нужно превратить их в обычную картинку
                    masks_data = result.masks.data.cpu().numpy()

                    if masks_data.size == 0:
                        self.logger.warning(f"⚠️ SAM не нашел объект внутри коробки №{i}. Пропускаем.")
                        continue

                    for mask_array in masks_data:
                        # mask_array - это массив 0 и 1 (или float)
                        # Превращаем в картинку 0-255
                        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8)).resize(image.size,
                                                                                               Image.Resampling.NEAREST)

                        # Добавляем эту маску к общей маске (сложение слоев)
                        # Если было черное (0), станет белым (255)
                        final_mask.paste(255, (0, 0), mask_img)

                else:
                    self.logger.warning(f"⚠️ SAM вернул None для коробки №{i}")

            self.logger.debug("Mask generated successfully.")
            return final_mask

        except Exception as e:
            self.logger.error(f"Error in SAM segmentation: {e}")
            # В случае ошибки вернем пустую маску, чтобы не упал весь пайплайн
            return final_mask


if __name__ == "__main__":
    try:
        refiner = MaskRefiner()
        print("✅ Segmenter initialized.")
    except Exception as e:
        print(e)