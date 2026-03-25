#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接使用路径加载 safetensors 模型
"""

import os
import sys
from pathlib import Path

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'

print("=" * 60)
print("直接路径加载测试")
print("=" * 60)

# 模型路径
model_path = PROJECT_ROOT / "models" / "sentence_transformers" / "models--BAAI--bge-large-zh-v1.5" / "snapshots" / "79e7739b6ab944e86d6171e44d24c997fc1e0116"

print(f"模型路径: {model_path}")
print(f"pytorch_model.safetensors 存在: {(model_path / 'pytorch_model.safetensors').exists()}")
print()

try:
    from sentence_transformers import SentenceTransformer

    print("直接加载模型...")
    model = SentenceTransformer(str(model_path))

    print("✅ 模型加载成功!")
    print(f"   模型类型: {type(model).__name__}")
    print(f"   嵌入维度: {model.get_sentence_embedding_dimension()}")

    # 测试编码
    print("\n测试编码...")
    test_texts = ["测试", "京东"]
    embeddings = model.encode(test_texts)
    print(f"✅ 编码成功! 形状: {embeddings.shape}")

    print("\n✅ 直接路径加载成功!")

except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()
