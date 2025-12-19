"""
Модуль Сегментации.
Превращает прямоугольные рамки (от YOLO) в точные черно-белые маски.

Главная фишка: Использует разные стратегии (BOX, SAM, OCR) в зависимости от
ID класса, указанного в config.py.
"""

import logging
import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import SAM
import config

# Пробуем импортировать EasyOCR (он нужен только для стратегии 'OCR')
try:
    import easyocr
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


class MaskRefiner:
    """
    Класс, отвечающий за создание маски удаления.
    Управляет моделями SAM и EasyOCR.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sam_model = None
        self.ocr_reader = None

        # 1. Анализируем конфиг: какие стратегии нам вообще нужны?
        # Собираем все стратегии, упомянутые в config.CLASS_PARAMS
        needed_strategies = set()
        for params in config.CLASS_PARAMS.values():
            needed_strategies.add(params['strategy'])

        # Также добавим стратегию по умолчанию
        needed_strategies.add(config.DEFAULT_PARAMS['strategy'])

        self.logger.info(f"📋 Требуемые стратегии сегментации: {needed_strategies}")

        # 2. Ленивая загрузка SAM
        if 'SAM' in needed_strategies:
            self.logger.info(f"⏳ Загрузка модели SAM: {config.SAM_MODEL_NAME}...")
            try:
                self.sam_model = SAM(config.SAM_MODEL_NAME)
                self.logger.info("✅ SAM загружен.")
            except Exception as e:
                self.logger.error(f"❌ Не удалось загрузить SAM: {e}")
                # Если SAM упал, программа не сломается, но вернет пустые маски для логотипов

        # 3. Ленивая загрузка OCR
        if 'OCR' in needed_strategies:
            if HAS_OCR:
                self.logger.info("⏳ Загрузка модели EasyOCR...")
                # Папка для кэширования моделей OCR
                ocr_storage = config.MODELS_DIR / "easyocr"
                ocr_storage.mkdir(exist_ok=True)

                try:
                    self.ocr_reader = easyocr.Reader(
                        config.OCR_LANGS,
                        gpu=(config.DEVICE == 'cuda'),
                        model_storage_directory=str(ocr_storage),
                        download_enabled=True,
                        verbose=False
                    )
                    self.logger.info("✅ EasyOCR загружен.")
                except Exception as e:
                    self.logger.error(f"❌ Ошибка EasyOCR: {e}")
            else:
                self.logger.warning("⚠️ Стратегия 'OCR' выбрана, но библиотека easyocr не установлена!")

    def create_mask(self, image: Image.Image, detections: list) -> Image.Image:
        """
        Главный метод. Создает маску для всех найденных объектов.

        Args:
            image: Исходное изображение (PIL).
            detections: Список [(x1, y1, x2, y2, conf, cls_id), ...].

        Returns:
            Черно-белая маска (PIL Image), где белое = удалять.
        """
        img_w, img_h = image.size

        # Создаем черный холст (пустая маска)
        final_mask_array = np.zeros((img_h, img_w), dtype=np.uint8)

        if not detections:
            return Image.fromarray(final_mask_array)

        # Проходим по каждому найденному объекту
        for det in detections:
            # Распаковываем данные (теперь 6 элементов!)
            x1, y1, x2, y2, conf, cls_id = det

            # Получаем настройки для конкретного класса из конфига
            # Если класса нет в конфиге, берем DEFAULT_PARAMS
            params = config.CLASS_PARAMS.get(cls_id, config.DEFAULT_PARAMS)

            strategy = params['strategy']
            pad = params['padding']
            dilation = params['dilation']

            # === 1. PADDING (Расширение рамки) ===
            nx1 = max(0, x1 - pad)
            ny1 = max(0, y1 - pad)
            nx2 = min(img_w, x2 + pad)
            ny2 = min(img_h, y2 + pad)

            # Бокс для обработки
            box_expanded = [nx1, ny1, nx2, ny2]

            # Переменная для маски конкретного объекта
            mask_part = None

            # === 2. ВЫПОЛНЕНИЕ СТРАТЕГИИ ===

            # --- СТРАТЕГИЯ: BOX ---
            if strategy == 'BOX':
                # Самый простой и надежный вариант для текста.
                # Просто рисуем белый прямоугольник.
                mask_part = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.rectangle(mask_part, (int(nx1), int(ny1)), (int(nx2), int(ny2)), 255, -1)

            # --- СТРАТЕГИЯ: SAM ---
            elif strategy == 'SAM':
                if self.sam_model:
                    mask_part = self._run_sam(image, box_expanded, img_w, img_h)
                else:
                    self.logger.error("Требуется SAM, но модель не загружена. Пропускаю.")

            # --- СТРАТЕГИЯ: OCR ---
            elif strategy == 'OCR':
                if self.ocr_reader:
                    mask_part = self._run_ocr(image, box_expanded, img_w, img_h)
                else:
                    self.logger.error("Требуется OCR, но модель не загружена.")

            # Если стратегия вернула None (ошибка) или неизвестная стратегия -> Фоллбэк на BOX
            if mask_part is None:
                mask_part = np.zeros((img_h, img_w), dtype=np.uint8)
                # Рисуем бокс, чтобы хоть что-то удалить
                cv2.rectangle(mask_part, (int(nx1), int(ny1)), (int(nx2), int(ny2)), 255, -1)

            # === 3. DILATION (Расширение маски) ===
            # Применяем утолщение, если задано в конфиге
            if dilation > 0:
                kernel = np.ones((dilation, dilation), np.uint8)
                # iterations=1 - один проход расширения
                mask_part = cv2.dilate(mask_part, kernel, iterations=1)

            # === 4. ОБЪЕДИНЕНИЕ ===
            # Добавляем маску этого объекта к общей маске (логическое ИЛИ)
            # np.maximum выбирает максимальное значение пикселя (если где-то уже было 255, останется 255)
            final_mask_array = np.maximum(final_mask_array, mask_part)

        return Image.fromarray(final_mask_array)

    def _run_sam(self, image: Image.Image, box: list, w: int, h: int) -> np.ndarray:
        """Запуск SAM для одного объекта"""
        try:
            # Препроцессинг (улучшение контраста), если включено в конфиге
            processed_image = self._preprocess_image(image)

            # SAM требует список боксов
            results = self.sam_model.predict(
                source=processed_image,
                bboxes=[box],
                device=config.DEVICE,
                verbose=False,
                retina_masks=True
            )

            full_mask = np.zeros((h, w), dtype=np.uint8)

            # Собираем результат
            for r in results:
                if r.masks is not None:
                    # data - это массив масок
                    for m in r.masks.data.cpu().numpy():
                        # Масштабируем маску под размер изображения (на всякий случай)
                        m_resized = cv2.resize((m * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        full_mask = np.maximum(full_mask, m_resized)

            return full_mask

        except Exception as e:
            self.logger.error(f"SAM Runtime Error: {e}")
            return None

    def _run_ocr(self, image: Image.Image, box: list, w: int, h: int) -> np.ndarray:
        """Запуск OCR для поиска текста внутри бокса"""
        try:
            x1, y1, x2, y2 = map(int, box)

            # Защита координат
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            # Вырезаем кусочек картинки для OCR
            # Конвертируем в numpy array (OpenCV формат)
            img_np = np.array(image)
            crop = img_np[y1:y2, x1:x2]

            # Улучшаем кроп перед OCR
            if config.OCR_ENHANCE_CONTRAST:
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                crop = clahe.apply(gray)

            # Читаем текст
            results = self.ocr_reader.readtext(
                crop,
                text_threshold=config.OCR_TEXT_THRESHOLD
            )

            mask_part = np.zeros((h, w), dtype=np.uint8)

            for (bbox, text, prob) in results:
                # bbox - координаты внутри кропа. Нужно перевести в глобальные.
                # bbox[0] = top_left, bbox[2] = bottom_right
                local_tl = bbox[0]
                local_br = bbox[2]

                gx1 = int(local_tl[0] + x1)
                gy1 = int(local_tl[1] + y1)
                gx2 = int(local_br[0] + x1)
                gy2 = int(local_br[1] + y1)

                # Рисуем "кирпич" на слове
                expand = config.OCR_EXPAND_PIXELS
                cv2.rectangle(mask_part,
                              (gx1 - expand, gy1 - expand),
                              (gx2 + expand, gy2 + expand),
                              255, -1)

            return mask_part

        except Exception as e:
            self.logger.error(f"OCR Runtime Error: {e}")
            return None

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Улучшение контраста перед подачей в SAM.
        Помогает, если логотип сливается с фоном.
        """
        if not getattr(config, 'SAM_ENHANCE_CONTRAST', False):
            return image

        try:
            img_np = np.array(image)
            # RGB -> LAB
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Применяем CLAHE к каналу L (Lightness)
            clahe = cv2.createCLAHE(
                clipLimit=getattr(config, 'SAM_CLAHE_CLIP', 2.0),
                tileGridSize=(getattr(config, 'SAM_CLAHE_GRID', 8),)*2
            )
            cl = clahe.apply(l)

            # Обратно в RGB
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            return Image.fromarray(final)

        except Exception:
            return image

# Тест модуля
if __name__ == "__main__":
    print("Инициализация MaskRefiner...")
    try:
        refiner = MaskRefiner()
        print("✅ Успешно.")
    except Exception as e:
        print(f"❌ Ошибка: {e}")