"""
Генератор Синтетических Данных для YOLOv11-seg.
Использует 100% файлов из backgrounds + добавляет градиенты.
"""

import cv2
import numpy as np
import random
import yaml
from pathlib import Path
from tqdm import tqdm
import config


def generate_gradient(w, h):
    """
    Генерирует случайный градиент (Линейный или Диагональный).
    ОПТИМИЗИРОВАННАЯ ВЕРСИЯ (NumPy Vectorization).
    Работает в 100 раз быстрее циклов.
    """
    # 1. Выбор цветов (оставляем твою логику)
    if random.random() < 0.5:
        c1 = np.random.randint(200, 256, (3,))  # Светлый 1
        c2 = np.random.randint(150, 230, (3,))  # Светлый 2
    else:
        c1 = np.random.randint(50, 120, (3,))  # Темный 1
        c2 = np.random.randint(10, 80, (3,))  # Темный 2

    # 2. Создаем сетку коэффициентов alpha (от 0.0 до 1.0)
    # Вместо циклов создаем массивы координат

    # Тип градиента
    rnd_type = random.random()

    if rnd_type < 0.7:
        # Линейный (Вертикальный или Горизонтальный)
        if random.random() < 0.5:
            # Вертикальный: меняется только по высоте (axis 0)
            # Shape: (h, 1, 1) — чтобы бродкастить на ширину и каналы
            alpha = np.linspace(0, 1, h).reshape(h, 1, 1)
        else:
            # Горизонтальный: меняется только по ширине (axis 1)
            # Shape: (1, w, 1)
            alpha = np.linspace(0, 1, w).reshape(1, w, 1)
    else:
        # Диагональный (имитация того, что у тебя было (i/h + j/w)/2)
        # Создаем вертикальный вектор Y и горизонтальный X
        Y = np.linspace(0, 1, h).reshape(h, 1, 1)
        X = np.linspace(0, 1, w).reshape(1, w, 1)
        # Складываем (NumPy сам расширит их до матрицы h x w)
        alpha = (Y + X) / 2.0

    # 3. Магия NumPy (Broadcasting)
    # alpha имеет форму (h, w, 1) (или бродкастится до неё)
    # c1 и c2 имеют форму (3,)
    # NumPy сам умножит каждый пиксель на цвет.
    img = (1.0 - alpha) * c1 + alpha * c2

    return img.astype(np.uint8)

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
    if w > w_orig or h > h_orig: return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else: return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

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

def apply_edge_corruption(img_rgba):
    if random.random() > getattr(config, 'GEN_PROB_EDGE_CORRUPTION', 0.5): return img_rgba
    b, g, r, a = cv2.split(img_rgba)
    ksize = random.choice([3, 5])
    kernel = np.ones((ksize, ksize), np.uint8)
    if random.random() < 0.5: a = cv2.erode(a, kernel, iterations=1)
    else: a = cv2.dilate(a, kernel, iterations=1)
    if random.random() < 0.5: a = cv2.GaussianBlur(a, (3, 3), 0)
    return cv2.merge((b, g, r, a))

def process_single_image(bg, wm_original, index, base_dir):
    """Логика обработки одного фона и сохранения"""
    split = "train" if random.random() < config.GEN_TRAIN_RATIO else "val"
    h_bg, w_bg = bg.shape[:2]
    filename = f"syn_{index:06d}" # 6 цифр

    # === NEGATIVE SAMPLE ===
    if random.random() < config.GEN_PROB_NEGATIVE:
        cv2.imwrite(str(base_dir / 'images' / split / f"{filename}.jpg"), bg)
        open(base_dir / 'labels' / split / f"{filename}.txt", "w").close()
        return

    wm_curr = wm_original.copy()
    is_hard_mode = random.random() > config.GEN_PROB_EASY_MODE

    # Трансформации
    if random.random() < config.GEN_ROTATION_PROB:
        wm_curr = cv2.rotate(wm_curr, cv2.ROTATE_90_CLOCKWISE)

    if is_hard_mode and random.random() < config.GEN_INVERT_PROB:
        wm_curr[:, :, :3] = cv2.bitwise_not(wm_curr[:, :, :3])

    scale_ratio = random.uniform(*config.GEN_SCALE_RANGE)
    h_wm, w_wm = wm_curr.shape[:2]
    new_w = int(w_bg * scale_ratio)
    new_h = int(h_wm * (new_w / w_wm))

    if new_h > h_bg: new_h = int(h_bg * 0.9); new_w = int(w_wm * (new_h / h_wm))
    if new_w < 10: return

    wm_resized = smart_resize(wm_curr, new_w, new_h)

    if is_hard_mode:
        wm_resized = apply_random_color(wm_resized)
        if random.random() < config.GEN_BLUR_PROB:
            k = random.choice([3, 5])
            wm_resized = cv2.GaussianBlur(wm_resized, (k, k), 0)
        wm_resized = apply_edge_corruption(wm_resized)
        opacity = random.uniform(*config.GEN_OPACITY_RANGE)
    else:
        opacity = random.uniform(0.7, 1.0)

    # Наложение
    wm_rgb = wm_resized[:, :, :3]
    wm_alpha = wm_resized[:, :, 3] / 255.0
    wm_alpha = wm_alpha * opacity

    y_off = random.randint(0, h_bg - new_h)
    x_off = random.randint(0, w_bg - new_w)

    roi = bg[y_off:y_off+new_h, x_off:x_off+new_w]
    for c in range(3):
        roi[:, :, c] = (1.0 - wm_alpha) * roi[:, :, c] + \
                       wm_alpha * wm_rgb[:, :, c]
    bg[y_off:y_off+new_h, x_off:x_off+new_w] = roi

    # Маска
    mask_binary = np.zeros((h_bg, w_bg), dtype=np.uint8)
    wm_shape = (wm_resized[:, :, 3] > 10).astype(np.uint8) * 255
    mask_binary[y_off:y_off+new_h, x_off:x_off+new_w] = wm_shape

    polygons = get_yolo_polygon(mask_binary, w_bg, h_bg)

    if polygons:
        cv2.imwrite(str(base_dir / 'images' / split / f"{filename}.jpg"), bg)
        with open(base_dir / 'labels' / split / f"{filename}.txt", "w") as f:
            for poly in polygons:
                f.write(f"0 {poly}\n")

def generate_dataset():
    # Проверки
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
    if wm_original.shape[2] == 3:
        b, g, r = cv2.split(wm_original)
        alpha = np.ones_like(b) * 255
        wm_original = cv2.merge((b, g, r, alpha))

    # Лечение исходника
    dil_size = getattr(config, 'GEN_SOURCE_REPAIR_DILATION', 0)
    if dil_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_size, dil_size))
        b,g,r,a = cv2.split(wm_original)
        a = cv2.dilate(a, kernel, iterations=1)
        b = cv2.dilate(b, kernel, iterations=1)
        g = cv2.dilate(g, kernel, iterations=1)
        r = cv2.dilate(r, kernel, iterations=1)
        wm_original = cv2.merge((b,g,r,a))

    # 1. Считаем, сколько нужно градиентов
    total_real = len(bg_files)
    total_solid = int(total_real * config.GEN_EXTRA_SOLID_PERCENT)
    total_count = total_real + total_solid

    print(f"🚀 Генерация полного датасета:")
    print(f"   📸 Реальные фоны: {total_real}")
    print(f"   🎨 Градиенты:     {total_solid}")
    print(f"   ∑  Итого:         {total_count}")

    counter = 0

    # 2. Проходим по ВСЕМ реальным фото
    print("   [1/2] Обработка реальных фонов...")
    for bg_path in tqdm(bg_files):
        bg = cv2.imread(str(bg_path))
        if bg is None: continue
        process_single_image(bg, wm_original, counter, base_dir)
        counter += 1

    # 3. Генерируем градиенты
    print("   [2/2] Генерация градиентов...")
    for _ in tqdm(range(total_solid)):
        h = random.randint(600, 1024)
        w = random.randint(600, 1024)
        bg = generate_gradient(w, h)
        process_single_image(bg, wm_original, counter, base_dir)
        counter += 1

    # data.yaml
    yaml_data = {
        'path': base_dir.absolute().as_posix(),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'watermark'}
    }
    with open(base_dir / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"\n✅ Генерация завершена! Всего создано {counter} примеров.")

if __name__ == "__main__":
    generate_dataset()