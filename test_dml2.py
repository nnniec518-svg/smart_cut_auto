import sys
sys.path.insert(0, '.')

print("测试 torch-directml...")

try:
    import torch
    print(f"torch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
except Exception as e:
    print(f"torch 错误: {e}")

try:
    import torch_directml
    print(f"torch_directml 可用")
    dml = torch_directml.device()
    print(f"DirectML 设备: {dml}")
except Exception as e:
    print(f"torch_directml 错误: {e}")
