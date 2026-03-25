#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接转换 .bin 文件为 .safetensors 格式
不通过 SentenceTransformers,避免 PyTorch 版本限制
"""

import os
import sys
import shutil
from pathlib import Path
import json
from datetime import datetime
from collections import OrderedDict

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
SENTENCE_TRANSFORMERS_HOME = MODELS_DIR / "sentence_transformers"

print("=" * 70)
print("直接转换 .bin 为 .safetensors")
print("=" * 70)

# 模型路径
MODEL_NAME = "BAAI/bge-large-zh-v1.5"
SNAPSHOT_DIR = SENTENCE_TRANSFORMERS_HOME / f"models--{MODEL_NAME.replace('/', '--')}" / "snapshots" / "79e7739b6ab944e86d6171e44d24c997fc1e0116"

print(f"快照目录: {SNAPSHOT_DIR}")
print()

# 检查文件
bin_file = SNAPSHOT_DIR / "pytorch_model.bin"
config_file = SNAPSHOT_DIR / "config.json"
safetensors_file = SNAPSHOT_DIR / "model.safetensors"

if not bin_file.exists():
    print(f"❌ 未找到 .bin 文件: {bin_file}")
    sys.exit(1)

if not config_file.exists():
    print(f"❌ 未找到 config.json: {config_file}")
    sys.exit(1)

# 检查是否已存在 safetensors
if safetensors_file.exists():
    print(f"⚠️  safetensors 文件已存在: {safetensors_file}")
    response = input("是否覆盖? (y/n): ")
    if response.lower() != 'y':
        print("已取消转换")
        sys.exit(0)

# 备份原始文件
backup_dir = SNAPSHOT_DIR.parent / "backup"
backup_dir.mkdir(exist_ok=True)

backup_file = backup_dir / f"pytorch_model.bin.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"备份原始文件到: {backup_file.name}")
shutil.copy2(bin_file, backup_file)
print(f"✅ 备份完成")
print()

try:
    # 方法1: 使用 torch.load (如果版本允许)
    print("尝试方法1: 使用 torch.load...")
    try:
        import torch

        # 直接加载 .bin 文件
        state_dict = torch.load(str(bin_file), map_location='cpu', weights_only=True)
        print(f"✅ 加载了 {len(state_dict)} 个参数")
        print()

        # 转换为 OrderedDict 并确保为 CPU
        state_dict = OrderedDict((k, v.cpu() if hasattr(v, 'cpu') else v)
                            for k, v in state_dict.items())

        # 使用 safetensors 保存
        from safetensors.torch import save_file
        save_file(state_dict, str(safetensors_file))
        print(f"✅ 保存到 safetensors 格式: {safetensors_file.name}")

        # 获取文件大小
        bin_size = bin_file.stat().st_size / (1024**2)
        safe_size = safetensors_file.stat().st_size / (1024**2)
        compression = (1 - safe_size / bin_size) * 100 if bin_size > 0 else 0

        print()
        print(f"原始 .bin 大小: {bin_size:.2f} MB")
        print(f"safetensors 大小: {safe_size:.2f} MB")
        print(f"压缩率: {compression:.1f}%")
        print()

        # 验证
        print("验证 safetensors 文件...")
        from safetensors import safe_open

        tensors = {}
        with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        print(f"✅ 验证通过! 包含 {len(tensors)} 个张量")
        print()

    except ValueError as e:
        if "torch" in str(e).lower() and "v2.6" in str(e):
            print(f"⚠️  PyTorch 版本过低: {e}")
            print("尝试方法2: 使用旧版 torch.load...")
            print()

            # 方法2: 使用 weights_only=False (不安全但可用)
            state_dict = torch.load(str(bin_file), map_location='cpu', weights_only=False)

            print(f"✅ 加载了 {len(state_dict)} 个参数")
            print()

            # 转换为 OrderedDict
            state_dict = OrderedDict((k, v.cpu() if hasattr(v, 'cpu') else v)
                                for k, v in state_dict.items())

            # 保存
            from safetensors.torch import save_file
            save_file(state_dict, str(safetensors_file))
            print(f"✅ 保存到 safetensors 格式")

            bin_size = bin_file.stat().st_size / (1024**2)
            safe_size = safetensors_file.stat().st_size / (1024**2)
            compression = (1 - safe_size / bin_size) * 100 if bin_size > 0 else 0

            print(f"原始 .bin 大小: {bin_size:.2f} MB")
            print(f"safetensors 大小: {safe_size:.2f} MB")
            print(f"压缩率: {compression:.1f}%")
            print()
        else:
            raise

    except ImportError:
        print("❌ 未安装 torch 库")
        print()
        print("请安装:")
        print("  pip install torch")
        sys.exit(1)

    # 询问是否删除原始文件
    response = input("是否删除原始 .bin 文件? (y/n): ")
    if response.lower() == 'y':
        bin_file.unlink()
        print(f"✅ 已删除原始 .bin 文件")
    else:
        print(f"原始 .bin 文件已保留: {bin_file}")

    print()
    print("=" * 70)
    print("✅ 转换完成!")
    print("=" * 70)
    print()
    print("下一步:")
    print("1. 测试模型: python main.py")
    print("2. 如需恢复,使用备份文件")

except Exception as e:
    print(f"❌ 转换失败: {e}")
    import traceback
    traceback.print_exc()

    print()
    print("建议:")
    print("1. 升级 PyTorch: pip install --upgrade torch")
    print("2. 或使用备用模型")
