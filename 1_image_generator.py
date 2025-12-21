"""
Генератор Синтетических Данных для YOLOv11-seg.
"""

import cv2
import numpy as np
import random
import yaml
from pathlib import Path
from tqdm import tqdm
import config

def repair_source_alpha(img, dilation_size):
    """Лечит края ватермарки"""
    if dilation_size <= 0:
        return img

    b, g, r, a = cv2.split(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))

    # Расширяем альфу и цвета
    a = cv2.dilate(a, kernel, iterations=1)
    b = cv2.dilate(b, kernel, iterations=1)
    g = cv2.dilate(g, kernel, iterations=1)
    r = cv2.dilate(r, kernel, iterations=1)

    return cv2.merge((b, g, r, a))

# ... (функции get_yolo_polygon, smart_resize, apply_blur, apply_random_color БЕЗ ИЗМЕНЕНИЙ) ...
def get_yolo_polygon(mask, img_w, img_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50: continue
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 4: continue
        points = []
        for point in approx:
            x, y = point[0]
            nx = max(0, min(1, x / img_w))
            ny = max(0, min(1, y / img_h))
            points.append(f"{nx:.6f} {ny:.6f}")
        polygons.append(" ".join(points))
    return polygons

def smart_resize(img, w, h):
    h_orig, w_orig = img.shape[:2]
    if w > w_orig or h > h_orig:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def apply_blur(img_rgba):
    if random.random() < config.GEN_BLUR_PROB:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img_rgba, (k, k), 0)
    return img_rgba

def apply_random_color(img_rgba):
    if random.random() < config.GEN_COLOR_PROB:
        color = np.random.randint(0, 256, (3,), dtype=np.uint8)
        gray = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_BGR2GRAY) / 255.0
        result = img_rgba.copy()
        for c in range(3):
            colored_layer = np.ones_like(gray) * color[c]
            result[:, :, c] = (colored_layer * gray).astype(np.uint8)
        return result
    return img_rgba

def generate_dataset():
    if not config.WATERMARK_SOURCE.exists():
        print(f"❌ Нет файла {config.WATERMARK_SOURCE}")
        return
    bg_files = list(config.BACKGROUNDS_DIR.glob("*.jpg"))
    if not bg_files:
        print(f"❌ Нет фонов. Запусти 0_download_backgrounds.py")
        return

    base_dir = config.TRAIN_DATASET_DIR
    for split in ['train', 'val']:
        (base_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Грузим исходник
    wm_original = cv2.imread(str(config.WATERMARK_SOURCE), cv2.IMREAD_UNCHANGED)
    if wm_original is None:
        print("❌ Ошибка чтения watermark.png")
        return

    if wm_original.shape[2] == 3:
        b, g, r = cv2.split(wm_original)
        alpha = np.ones_like(b) * 255
        wm_original = cv2.merge((b, g, r, alpha))

    # === ЛЕЧЕНИЕ (ОПЦИОНАЛЬНО) ===
    dil_size = getattr(config, 'GEN_SOURCE_REPAIR_DILATION', 0)
    if dil_size > 0:
        print(f"🚑 Расширяем исходник на {dil_size} пикселей...")
        wm_prepared = repair_source_alpha(wm_original, dil_size)
    else:
        print("👌 Исходник не меняем (Repair = 0).")
        wm_prepared = wm_original
    # =============================

    print(f"🚀 Генерация {config.GEN_TOTAL_COUNT} фото...")

    for i in tqdm(range(config.GEN_TOTAL_COUNT)):
        split = "train" if random.random() < config.GEN_TRAIN_RATIO else "val"

        bg_path = random.choice(bg_files)
        bg = cv2.imread(str(bg_path))
        if bg is None: continue
        h_bg, w_bg = bg.shape[:2]

        wm_curr = wm_prepared.copy() # Берем подготовленную версию

        # 1. Аугментации
        if random.random() < config.GEN_INVERT_PROB:
            wm_curr[:, :, :3] = cv2.bitwise_not(wm_curr[:, :, :3])

        if random.random() < config.GEN_ROTATION_PROB:
            wm_curr = cv2.rotate(wm_curr, cv2.ROTATE_90_CLOCKWISE)

        scale_ratio = random.uniform(*config.GEN_SCALE_RANGE)
        h_wm, w_wm = wm_curr.shape[:2]
        new_w = int(w_bg * scale_ratio)
        new_h = int(h_wm * (new_w / w_wm))

        if new_h > h_bg:
            new_h = int(h_bg * 0.9)
            new_w = int(w_wm * (new_h / h_wm))

        if new_w < 10 or new_h < 10: continue

        wm_resized = smart_resize(wm_curr, new_w, new_h)
        wm_resized = apply_blur(wm_resized)
        wm_resized = apply_random_color(wm_resized)

        opacity = random.uniform(*config.GEN_OPACITY_RANGE)

        wm_rgb = wm_resized[:, :, :3]
        wm_alpha = wm_resized[:, :, 3] / 255.0
        wm_alpha = wm_alpha * opacity

        # 2. Наложение
        y_off = random.randint(0, h_bg - new_h)
        x_off = random.randint(0, w_bg - new_w)

        roi = bg[y_off:y_off+new_h, x_off:x_off+new_w]
        for c in range(3):
            roi[:, :, c] = (1.0 - wm_alpha) * roi[:, :, c] + \
                           wm_alpha * wm_rgb[:, :, c]
        bg[y_off:y_off+new_h, x_off:x_off+new_w] = roi

        # 3. Маска
        mask_binary = np.zeros((h_bg, w_bg), dtype=np.uint8)
        wm_shape = (wm_resized[:, :, 3] > 10).astype(np.uint8) * 255
        mask_binary[y_off:y_off+new_h, x_off:x_off+new_w] = wm_shape

        # 4. Полигоны
        polygons = get_yolo_polygon(mask_binary, w_bg, h_bg)

        if polygons:
            filename = f"syn_{i:05d}"
            cv2.imwrite(str(base_dir / 'images' / split / f"{filename}.jpg"), bg)
            with open(base_dir / 'labels' / split / f"{filename}.txt", "w") as f:
                for poly in polygons:
                    f.write(f"0 {poly}\n")

    yaml_data = {
        'path': base_dir.absolute().as_posix(),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'watermark'}
    }
    with open(base_dir / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"\n✅ Генерация завершена! Создано {config.GEN_TOTAL_COUNT} примеров.")

if __name__ == "__main__":
    generate_dataset()