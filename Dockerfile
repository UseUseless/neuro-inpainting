# 1. Берем базу: Официальный образ PyTorch с поддержкой CUDA (драйвера NVIDIA)
# Это сэкономит тебе часы настройки драйверов на Linux внутри контейнера.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 2. Настройки Python (чтобы не создавал мусор .pyc и выводил логи сразу)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Устанавливаем системные библиотеки, которые нужны для OpenCV
# Без этого cv2 выдаст ошибку "ImportError: libGL.so.1 cannot open shared object file"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Создаем рабочую папку внутри контейнера
WORKDIR /app

# 5. Сначала копируем только requirements.txt
# Это хитрость Докера: если ты поменяешь код, но не библиотеки,
# он не будет заново качать гигабайты библиотек, а возьмет их из кэша.
COPY requirements.txt .

# 6. Устанавливаем библиотеки
# --no-cache-dir уменьшает размер образа
RUN pip install --no-cache-dir -r requirements.txt

# 7. Теперь копируем весь твой проект внутрь
COPY . .

# 8. По умолчанию запускаем командную строку (bash)
CMD ["/bin/bash"]