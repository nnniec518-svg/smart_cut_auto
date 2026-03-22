"""
硬件抽象层 - AMD Strix Halo 平台优化
支持 DirectML (RDNA 3.5 iGPU) 和 CPU 回退
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 全局设备状态
_device = None
_device_name = "cpu"
_device_type = "cpu"  # "directml" or "cpu"


def init_device(force_cpu: bool = False) -> any:
    """
    初始化推理设备
    
    Args:
        force_cpu: 是否强制使用 CPU
        
    Returns:
        torch.device
    """
    global _device, _device_name, _device_type
    
    if _device is not None and not force_cpu:
        return _device
    
    if force_cpu:
        _device_name = "cpu"
        _device_type = "cpu"
        import torch
        _device = torch.device("cpu")
        logger.info("使用 CPU 设备 (force_cpu=True)")
        return _device
    
    # 尝试使用 DirectML
    try:
        import torch
        import torch_directml
        
        dml_device = torch_directml.device()
        _device = dml_device
        _device_name = f"DirectML: {dml_device}"
        _device_type = "directml"
        
        # 输出设备信息
        logger.info(f"=" * 50)
        logger.info(f"DirectML 设备初始化成功")
        logger.info(f"设备: {_device_name}")
        logger.info(f"设备类型: {_device_type}")
        logger.info(f"=" * 50)
        
        # 验证设备可用性
        test_tensor = torch.tensor([1.0], device=_device)
        logger.info(f"设备测试: {test_tensor}")
        
        return _device
        
    except ImportError:
        logger.warning("torch-directml 未安装，将使用 CPU")
        import torch
        _device = torch.device("cpu")
        _device_name = "cpu"
        _device_type = "cpu"
        return _device
    except Exception as e:
        logger.warning(f"DirectML 初始化失败: {e}，回退到 CPU")
        import torch
        _device = torch.device("cpu")
        _device_name = "cpu"
        _device_type = "cpu"
        return _device


def init_device_safe() -> any:
    """
    安全初始化设备，优先 DirectML，失败则 CPU
    用于向量模型（因为 torch 版本兼容性问题）
    """
    global _device, _device_name, _device_type
    
    # 总是使用 CPU 作为向量模型的设备（避免 torch 版本冲突）
    import torch
    _device = torch.device("cpu")
    _device_name = "cpu"
    _device_type = "cpu"
    logger.info("向量模型使用 CPU 设备（避免 torch 版本兼容性问题）")
    return _device


def get_device() -> any:
    """获取当前设备，如果没有初始化则初始化"""
    global _device
    if _device is None:
        return init_device()
    return _device


def get_device_info() -> dict:
    """获取设备信息"""
    return {
        "device": _device,
        "device_name": _device_name,
        "device_type": _device_type
    }


def is_directml_available() -> bool:
    """检查 DirectML 是否可用"""
    try:
        import torch_directml
        return True
    except ImportError:
        return False


def set_batch_size_for_igpu(batch_size: int = 32) -> int:
    """
    根据设备类型调整批量大小
    
    Strix Halo 有高带宽统一内存，适合大 batch
    
    Args:
        batch_size: 基础批量大小
        
    Returns:
        调整后的批量大小
    """
    if _device_type == "directml":
        # DirectML 设备使用较大 batch
        return min(batch_size, 64)
    else:
        return batch_size
