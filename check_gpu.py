import torch
import torch_directml

print(f"torch: {torch.__version__}")

try:
    dml = torch_directml.device()
    # 尝试在 DML 上做一个简单的矩阵乘法
    a = torch.tensor([[1., 2.], [3., 4.]]).to(dml)
    b = (a @ a).cpu()
    print(f"DirectML OK: {b.flatten().tolist()}")
except Exception as e:
    print(f"DirectML Error: {e}")