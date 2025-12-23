"""
Проверка доступности GPU (NVIDIA CUDA) для PyTorch.
"""

import sys
import torch
import platform

def check_gpu():
    print("="*40)
    print(f"🐍 Python: {sys.version.split()[0]} ({platform.system()})")
    print(f"🔥 PyTorch: {torch.__version__}")
    print("="*40)

    if torch.cuda.is_available():
        print("✅ CUDA AVAILABLE! Видеокарта обнаружена.")
        device_count = torch.cuda.device_count()
        print(f"🔢 Количество GPU: {device_count}")
        
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   👉 GPU {i}: {name} ({mem:.2f} GB VRAM)")
            
        print("\n🧪 Тестовый прогон тензоров...")
        try:
            x = torch.rand(5, 3).cuda()
            print("   ✅ Тензор успешно создан в VRAM!")
            print(f"   Устройство тензора: {x.device}")
        except Exception as e:
            print(f"   ❌ Ошибка при работе с VRAM: {e}")

    else:
        print("⚠️  CUDA NOT AVAILABLE.")
        print("   Работаем на процессоре (CPU). Это будет медленно.")
        print("   Если у тебя NVIDIA карта:")
        print("   1. Проверь драйверы.")
        print("   2. Убедись, что установил PyTorch с поддержкой CUDA:")
        print("      pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124")

    print("="*40)

if __name__ == "__main__":
    check_gpu()