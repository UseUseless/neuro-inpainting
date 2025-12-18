"""
Кастомное логирование для обработки фото после обучения


pipeline.log (Всё подряд)
Содержит: Полная история работы (INFO, WARNING, ERROR).
Хронология запуска, статус загрузки моделей, статистика скорости, успешные завершения.

errors.log (Только проблемы)
Содержит: Ошибки и исключения (ERROR, CRITICAL).

failed_files.txt (Для ретрая)
Файлы, на которых скрипт упал. Можно скормить этот список программе заново.

Консоль
Дубликат pipeline.log + Progress Bar.
Мониторинг в реальном времени.
"""

import logging
import sys
import config


def setup_logger():
    """
    Настраивает ГЛОБАЛЬНЫЙ (Root) логгер.
    Теперь сообщения из detector, segmenter и cleaner будут автоматически
    попадать в эти же файлы.
    """

    # Создаем папку для логов
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Форматтер
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Получаем КОРНЕВОЙ логгер (пустые скобки)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Очищаем старые хендлеры (чтобы не дублировалось при перезапусках)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # === 1. Файл pipeline.log (INFO) ===
    general_path = config.LOG_DIR / "pipeline.log"
    general_handler = logging.FileHandler(general_path, mode='a', encoding='utf-8')
    general_handler.setLevel(logging.INFO)
    general_handler.setFormatter(formatter)
    root_logger.addHandler(general_handler)

    # === 2. Файл errors.log (ERROR) ===
    error_path = config.LOG_DIR / "errors.log"
    error_handler = logging.FileHandler(error_path, mode='a', encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # === 3. Консоль (INFO) ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Чтобы библиотеки типа PIL или urllib не спамили в лог, заглушим их
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    return root_logger