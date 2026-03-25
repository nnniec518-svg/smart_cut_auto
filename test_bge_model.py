#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 BGE 模型加载
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置模型路径
models_dir = project_root / "models"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(models_dir / "sentence_transformers")

print("=" * 60)
print("测试 BGE 模型加载")
print("=" * 60)
print(f"模型目录: {models_dir}")
print(f"SENTENCE_TRANSFORMERS_HOME: {os.environ.get('SENTENCE_TRANSFORMERS_HOME')}")

# 检查本地模型是否存在
local_model_path = models_dir / "sentence_transformers" / "models--BAAI--bge-large-zh-v1.5"
print(f"\n本地模型路径: {local_model_path}")
print(f"模型存在: {'✅ 是' if local_model_path.exists() else '❌ 否'}")

if local_model_path.exists():
    print("\n📦 模型文件:")
    snapshots_dir = local_model_path / "snapshots"
    if snapshots_dir.exists():
        for snapshot in snapshots_dir.iterdir():
            print(f"   - {snapshot.name}/")
            # 使用最新的快照
            if not 'model_snapshot' in locals():
                model_snapshot = snapshot

try:
    from sentence_transformers import SentenceTransformer
    
    # 使用 model_name 而不是直接路径，让 SentenceTransformers 自己处理缓存
    print("\n🔄 正在加载模型...")
    print(f"   模型名: BAAI/bge-large-zh-v1.5")
    print(f"   缓存目录: {os.environ.get('SENTENCE_TRANSFORMERS_HOME')}")
    
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    
    print(f"✅ 模型加载成功!")
    print(f"   模型名称: BAAI/bge-large-zh-v1.5")
    print(f"   嵌入维度: {model.get_sentence_embedding_dimension()}")
    
    # 测试编码
    print("\n🧪 测试文本编码...")
    test_texts = [
        "这是一个测试句子",
        "京东电器",
        "洗烘套装"
    ]
    
    embeddings = model.encode(test_texts)
    print(f"   编码形状: {embeddings.shape}")
    
    # 计算相似度
    import numpy as np
    from sentence_transformers.util import cos_sim
    
    print("\n📊 文本相似度矩阵:")
    for i, text1 in enumerate(test_texts):
        for j, text2 in enumerate(test_texts):
            similarity = float(cos_sim(embeddings[i], embeddings[j]))
            if i <= j:
                print(f"   '{text1}' ↔ '{text2}': {similarity:.4f}")
    
    print("\n✅ 所有测试通过!")
    
except Exception as e:
    print(f"\n❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
