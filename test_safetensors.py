#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试转换后的 safetensors 模型
"""

import os
import sys
from pathlib import Path

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(PROJECT_ROOT / "models" / "sentence_transformers")

print("=" * 60)
print("测试 safetensors 格式的 BGE 模型")
print("=" * 60)
print(f"SENTENCE_TRANSFORMERS_HOME: {os.environ.get('SENTENCE_TRANSFORMERS_HOME')}")
print()

try:
    from sentence_transformers import SentenceTransformer

    # 设置缓存目录
    cache_folder = str(PROJECT_ROOT / "models" / "sentence_transformers")

    print("正在加载 BGE 模型...")
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5', cache_folder=cache_folder)

    print("✅ 模型加载成功!")
    print(f"   模型类型: {type(model).__name__}")
    print(f"   嵌入维度: {model.get_sentence_embedding_dimension()}")

    # 测试编码
    print("\n测试文本编码...")
    test_texts = ["这是一个测试句子", "京东电器", "洗烘套装"]
    embeddings = model.encode(test_texts)

    print(f"✅ 编码成功!")
    print(f"   输入形状: {len(test_texts)} 个文本")
    print(f"   输出形状: {embeddings.shape}")

    # 计算相似度
    print("\n计算相似度矩阵:")
    from sentence_transformers.util import cos_sim
    import numpy as np

    for i, text1 in enumerate(test_texts):
        for j, text2 in enumerate(test_texts):
            if i <= j:
                sim = float(cos_sim(embeddings[i], embeddings[j]))
                print(f"   '{text1}' ↔ '{text2}': {sim:.4f}")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)
    print("\nBGE 模型 (safetensors 格式) 可以正常使用")
    print("现在可以运行主程序了:")
    print("  python main.py")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n请检查:")
    print("1. model.safetensors 文件是否存在")
    print("2. 所有配置文件是否完整")
    sys.exit(1)
