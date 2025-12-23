"""
Модуль Очистки (Inpainting).
Заполняет вырезанные области (маску) сгенерированным фоном.
Использует локальную модель LaMa (TorchScript) с автоскачиванием.
"""

import logging
import torch
import numpy as np
from PIL import Image
import config
from core.utils import ensure_model

class ImageInpainter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = config.DEVICE
        self.model_path = config.LAMA_MODEL_PATH

        # 1. Проверяем и качаем модель, если её нет
        try:
            ensure_model(self.model_path, config.LAMA_MODEL_URL)
        except Exception as e:
            self.logger.critical(f"❌ Не удалось получить модель LaMa: {e}")
            raise e

        self.logger.info(f"⏳ Загрузка LaMa: {self.model_path.name}...")
        try:
            # 2. Грузим TorchScript модель
            self.model = torch.jit.load(str(self.model_path), map_location=self.device)
            self.model.eval()
            self.model.to(self.device)
            self.logger.info("✅ LaMa готова к работе.")
        except Exception as e:
            self.logger.critical(f"❌ Битый файл модели или ошибка CUDA: {e}")
            raise e

    def _preprocess(self, image: Image.Image, mask: Image.Image):
        """
        Подготовка тензоров для LaMa.
        Модель требует, чтобы размеры были кратны 8.
        """
        w, h = image.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        
        # Ресайз (если не кратно 8)
        img_resized = image.resize((new_w, new_h), Image.BILINEAR)
        mask_resized = mask.resize((new_w, new_h), Image.NEAREST)

        # Image: (H, W, 3) -> (3, H, W) -> Float 0..1
        img_np = np.array(img_resized).transpose(2, 0, 1)
        img_t = torch.from_numpy(img_np).float() / 255.0
        
        # Mask: (H, W) -> (1, H, W) -> Float 0..1
        mask_np = np.array(mask_resized)
        if len(mask_np.shape) == 2:
            mask_np = mask_np[np.newaxis, :, :] # Добавляем канал
        else:
             mask_np = mask_np.transpose(2, 0, 1) # Если вдруг RGB маска

        mask_t = torch.from_numpy(mask_np).float() / 255.0
        
        # Батч (1, 3, H, W)
        img_t = img_t.unsqueeze(0).to(self.device)
        mask_t = mask_t.unsqueeze(0).to(self.device)
        
        # Бинаризация маски (на всякий случай)
        mask_t = (mask_t > 0.5).float()
            
        return img_t, mask_t, w, h

    def clean(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Главный метод очистки.
        """
        # Если маска пустая — возвращаем оригинал сразу
        if not mask.getbbox():
            return image

        try:
            # 1. Препроцессинг
            img_t, mask_t, orig_w, orig_h = self._preprocess(image, mask)

            # 2. Инференс (без градиентов для скорости)
            with torch.no_grad():
                output = self.model(img_t, mask_t)

            # 3. Постпроцессинг (Tensor -> PIL)
            output = output[0].permute(1, 2, 0).cpu().numpy() # (3, H, W) -> (H, W, 3)
            output = np.clip(output * 255, 0, 255).astype(np.uint8)
            
            result_img = Image.fromarray(output)
            
            # Возвращаем исходный размер (если он был не кратен 8)
            if result_img.size != (orig_w, orig_h):
                 result_img = result_img.resize((orig_w, orig_h), Image.LANCZOS)
                 
            return result_img

        except Exception as e:
            self.logger.error(f"❌ Ошибка во время inpainting: {e}")
            return image

if __name__ == "__main__":
    # Тест запуска
    try:
        cleaner = ImageInpainter()
        print("LaMa initialized successfully.")
    except Exception as e:
        print(f"Failed: {e}")