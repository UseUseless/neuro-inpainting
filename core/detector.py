"""
Находит твой класс (YOLO) после обучения

Загружает best.pt - дообученная модель из 2_train_model
Принимает картинку -> Отдает координаты квадрата [x1, y1, x2, y2].
"""

import logging
from typing import List, Tuple
from PIL import Image
from ultralytics import YOLO
import config


class YourClassDetector:
    """
    Отвечает за поиск размеченных классов на изображении с помощью YOLO11.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Проверяем наличие файла весов
        if not config.YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"❌ Не найден файл модели YOLO: {config.YOLO_MODEL_PATH}\n"
                f"Сначала обучи модель и скопируй best.pt в папку models!"
            )

        self.logger.info(f"Loading YOLO model from {config.YOLO_MODEL_PATH}...")
        self.model = YOLO(config.YOLO_MODEL_PATH)

    def detect(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Принимает изображение PIL.
        Возвращает список координат [x1, y1, x2, y2] для найденных классов.
        """
        # Запускаем предсказание
        # conf=... отсекает слабые предсказания
        # verbose=False отключает спам в консоль
        results = self.model.predict(
            source=image,
            conf=config.YOLO_CONFIDENCE,
            device=config.DEVICE,
            verbose=False
        )

        boxes_list = []

        # Разбираем результат (YOLO может вернуть несколько классов на одном фото)
        for result in results:
            for box in result.boxes:
                # Получаем координаты x1, y1, x2, y2
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords

                boxes_list.append((x1, y1, x2, y2))

        if not boxes_list:
            self.logger.debug("No your_classes detected.")
        else:
            self.logger.debug(f"Detected {len(boxes_list)} your_classes.")

        return boxes_list


# Небольшой тест, чтобы можно было проверить этот файл отдельно
if __name__ == "__main__":
    # Для теста нужен файл best.pt в папке models!
    try:
        detector = YourClassDetector()
        print("✅ Detector initialized successfully.")
    except Exception as e:
        print(e)