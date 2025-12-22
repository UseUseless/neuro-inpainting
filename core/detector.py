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
# Может убрать?
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """CLAHE с настройками из конфига"""
        try:
            img_np = np.array(image)
            if len(img_np.shape) == 3:
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)

                # Берем настройки
                clip = config.DETECTOR_CLAHE_CLIP
                grid = config.DETECTOR_CLAHE_GRID

                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid))
                cl = clahe.apply(l)

                limg = cv2.merge((cl, a, b))
                final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=config.DETECTOR_CLAHE_CLIP, tileGridSize=(8,8))
                final = clahe.apply(img_np)
            return Image.fromarray(final)
        except Exception:
            return image

    def get_mask(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        input_image = image
        if config.IS_CLAHE_DETECT:
            input_image = self._enhance_image(image)

        try:
            results = self.model.predict(
                source=input_image,
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