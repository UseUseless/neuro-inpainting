"""
Модуль Детекции (YOLO-Seg).
"""

import logging
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import config

class YourClassDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not config.YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"❌ Модель не найдена: {config.YOLO_MODEL_PATH}")

        self.logger.info(f"⏳ Загрузка YOLO: {config.YOLO_MODEL_PATH}...")
        try:
            self.model = YOLO(config.YOLO_MODEL_PATH)
        except Exception as e:
            self.logger.critical(f"❌ Ошибка YOLO: {e}")
            raise e

    def get_mask(self, image: Image.Image) -> Image.Image:
        w, h = image.size

        try:
            results = self.model.predict(
                source=image,
                conf=config.YOLO_CONFIDENCE,
                device=config.DEVICE,
                verbose=False,
                retina_masks=True
            )
        except Exception as e:
            self.logger.error(f"Prediction Error: {e}")
            return Image.new("L", (w, h), 0)

        final_mask = np.zeros((h, w), dtype=np.uint8)

        if results and results[0].masks:
            result = results[0]
            for polygon in result.masks.xy:
                if len(polygon) == 0: continue
                points = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(final_mask, [points], 255)

        # Dilation из конфига
        dil_px = config.CLEANER_MASK_DILATION
        if dil_px > 0:
            kernel = np.ones((dil_px, dil_px), np.uint8)
            final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        return Image.fromarray(final_mask)

if __name__ == "__main__":
    YourClassDetector()