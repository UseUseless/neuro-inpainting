"""
Находит объекты (текст, логотипы) на изображении с помощью YOLO.

Загружает обученную модель (best.pt) и возвращает информацию о найденных
объектах: координаты рамки, уверенность и ID класса.
"""

import logging
from typing import List, Tuple
from PIL import Image
from ultralytics import YOLO
import config


def reduce_boxes(detections: List[Tuple]) -> List[Tuple]:
    """
    Фильтрует рамки отдельно для каждого класса.
    Если для класса 0 лимит=1, а для класса 1 лимит=1,
    то функция оставит 1 лучший текст И 1 лучшее лого.
    """
    if not config.LIMIT_BOXES:
        return detections

    if not detections:
        return []

    # 1. Группируем по классам
    # grouped = { 0: [det1, det2], 1: [det3] }
    grouped: Dict[int, List] = {}

    for det in detections:
        cls_id = det[5]  # 6-й элемент это ID класса
        if cls_id not in grouped:
            grouped[cls_id] = []
        grouped[cls_id].append(det)

    final_list = []

    # 2. Проходим по каждому классу отдельно
    for cls_id, items in grouped.items():
        # Сортируем этот класс по уверенности (от большей к меньшей)
        # item[4] это confidence
        items.sort(key=lambda x: x[4], reverse=True)

        # Узнаем лимит для этого класса из конфига
        params = config.CLASS_PARAMS.get(cls_id, config.DEFAULT_PARAMS)
        limit = params.get('max_instances', 1)  # По дефолту 1, если забыл указать

        # Берем ТОП-N
        kept_items = items[:limit]
        final_list.extend(kept_items)

    return final_list


class YourClassDetector:
    """
    Класс-обертка для детектора YOLO.
    Отвечает за загрузку модели и поиск объектов на изображении.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Проверяем, существует ли файл с обученной моделью
        if not config.YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"❌ НЕ НАЙДЕН ФАЙЛ МОДЕЛИ YOLO: {config.YOLO_MODEL_PATH}\n"
                f"   -> Сначала запусти '1_prepare_dataset.py', затем '2_train_model.py'\n"
                f"   -> Скопируй файл 'best.pt' из папки runs/detect/train_run/weights/ в models/"
            )

        self.logger.info(f"⏳ Загрузка модели YOLO из: {config.YOLO_MODEL_PATH}")
        try:
            # Загружаем модель YOLO
            self.model = YOLO(config.YOLO_MODEL_PATH)
            self.logger.info("✅ Модель YOLO успешно загружена.")
        except Exception as e:
            self.logger.error(f"❌ Ошибка при загрузке модели YOLO: {e}")
            raise e

    def detect(self, image: Image.Image) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Выполняет детекцию объектов на изображении.

        Args:
            image: Объект PIL.Image.

        Returns:
            Список найденных объектов. Каждый объект - это кортеж:
            (x1, y1, x2, y2, confidence, class_id)
            где:
                x1, y1, x2, y2 - координаты левого верхнего и правого нижнего углов рамки.
                confidence - уверенность модели (от 0.0 до 1.0).
                class_id - ID класса (0, 1, ...), соответствует порядку в classes.txt.
        """
        # Запускаем предсказание YOLO
        # conf - минимальный порог уверенности для обнаружения
        # verbose=False - отключаем подробный вывод YOLO в консоль
        try:
            results = self.model.predict(
                source=image,
                conf=config.YOLO_CONFIDENCE,
                device=config.DEVICE,
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"❌ Ошибка предсказания: {e}")
            return []

        raw_detections = []
        if results:
            result = results[0]
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                raw_detections.append((
                    coords[0], coords[1], coords[2], coords[3],
                    conf, cls_id
                ))

        # === ФИЛЬТРАЦИЯ ===
        filtered_detections = reduce_boxes(raw_detections)

        if len(filtered_detections) < len(raw_detections):
            self.logger.debug(f"Фильтр классов: {len(raw_detections)} -> {len(filtered_detections)} объектов.")

        return filtered_detections

# Блок для тестирования файла отдельно (если нужно)
if __name__ == "__main__":
    # Убедись, что в папке 'models' есть файл best.pt
    try:
        print("Тестирование core/detector.py...")
        detector = YourClassDetector()
        print("✅ Детектор успешно инициализирован.")

        # Пример использования (нужна картинка в images_input)
        test_image_path = config.INPUT_DIR / "test_image.jpg" # Замени на реальное имя файла
        if test_image_path.exists():
             try:
                 img = Image.open(test_image_path)
                 detected_objects = detector.detect(img)
                 print(f"Найдено объектов на {test_image_path.name}: {len(detected_objects)}")
                 for obj in detected_objects:
                     print(f"  - coords: ({obj[0]}, {obj[1]}), ({obj[2]}, {obj[3]}), conf: {obj[4]:.2f}, class_id: {obj[5]}")
             except Exception as e:
                 print(f"Ошибка при обработке тестового изображения: {e}")
        else:
            print(f"⚠️ Тестовое изображение '{test_image_path}' не найдено. Пропустил детекцию.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")