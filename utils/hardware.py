"""
硬件抽象层 - AMD Strix Halo DirectML 加速
强制使用 AMD Radeon Graphics (DirectML)，禁止自动回退 CPU
"""

import torch
import torch_directml

# 禁用一些与 DirectML 不兼容的 torch 功能
torch._inductor.config.triton.cudagraphs = False

# 全局设备 - 强制 DirectML
DEVICE = None

def get_device():
    """
    获取 DirectML 设备
    经过 check_gpu.py 验证，此设备在当前环境(Torch 2.4.1)下完全可用
    """
    global DEVICE
    if DEVICE is None:
        DEVICE = torch_directml.device()
        print(f"[DirectML] Hardware acceleration locked: {DEVICE} (AMD Radeon Graphics)")
    return DEVICE

# 初始化设备
get_device()

def get_device_for_models():
    """
    获取可用于模型加载的设备
    由于 torch 2.4.1+cpu 不支持 cuda:0 trick，返回设备对象
    """
    return get_device()

def is_directml() -> bool:
    """检查是否为 DirectML 设备"""
    global DEVICE
    if DEVICE is None:
        return False
    return "privateuseone" in str(DEVICE)
