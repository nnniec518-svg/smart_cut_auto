"""
将 pytorch_model.bin 转换为 pytorch_model.safetensors
"""
import torch
from safetensors.torch import save_file
from pathlib import Path

# 转换路径
model_dir = Path("models/bge-large-zh-v1.5")
bin_path = model_dir / "pytorch_model.bin"
safetensors_path = model_dir / "pytorch_model.safetensors"

print(f"正在加载模型: {bin_path}")
state_dict = torch.load(bin_path, map_location="cpu")

print(f"正在保存为 safetensors 格式: {safetensors_path}")
save_file(state_dict, safetensors_path)

print(f"转换完成！")
print(f"原始文件大小: {bin_path.stat().st_size / 1024**3:.2f} GB")
print(f"转换后文件大小: {safetensors_path.stat().st_size / 1024**3:.2f} GB")
