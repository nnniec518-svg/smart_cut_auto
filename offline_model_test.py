#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线模型快速测试
"""

import os
import sys
from pathlib import Path

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"

# 设置环境变量（确保使用本地模型）
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR / "sentence_transformers")
# 禁用网络（确保完全离线）
os.environ["HF_HUB_OFFLINE"] = "1"

print("=" * 60)
print("离线模型测试")
print("=" * 60)
print(f"模型目录: {MODELS_DIR}")
print(f"SENTENCE_TRANSFORMERS_HOME: {os.environ['SENTENCE_TRANSFORMERS_HOME']}")
print(f"HF_HUB_OFFLINE: {os.environ['HF_HUB_OFFLINE']}")
print()

try:
    from sentence_transformers import SentenceTransformer
    
    print("正在加载 BGE 模型...")
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    
    print("✅ 模型加载成功!")
    print(f"   维度: {model.get_sentence_embedding_dimension()}")
    
    # 测试编码
    print("\n测试文本编码...")
    texts = ["这是一个测试", "京东电器", "洗烘套装"]
    embeddings = model.encode(texts)
    print(f"✅ 编码成功! 形状: {embeddings.shape}")
    
    # 计算相似度
    print("\n相似度测试:")
    import numpy as np
    from sentence_transformers.util import cos_sim
    for i in range(len(texts)):
        for j in range(i, len(texts)):
            sim = float(cos_sim(embeddings[i], embeddings[j]))
            print(f"   '{texts[i]}' ↔ '{texts[j]}': {sim:.4f}")
    
    print("\n✅ 所有测试通过! 模型可以完全离线运行。")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
