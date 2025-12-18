"""
Проверки:
-работает ли CUDA;
-точно ли у нас библиотека для работы LaMa на видеокарте, а не на CPU.
"""

import torch
import onnxruntime as ort

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Working on CPU!")

print('='*30)

print(f"ONNX Runtime version: {ort.__version__}")
print(f"ONNX Device: {ort.get_device()}")
print(f"Available Providers: {ort.get_available_providers()}")

if 'CUDAExecutionProvider' in ort.get_available_providers():
    print("=) LaMa будет работать на видеокарте (GPU)!")
else:
    print("=( LaMa видит только процессор (CPU).")