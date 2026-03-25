#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的模型加载逻辑
"""

import os
import sys
from pathlib import Path

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置缓存目录
MODELS_DIR = PROJECT_ROOT / "models"
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(MODELS_DIR / "sentence_transformers")

print("=" * 60)
print("测试修复后的模型加载逻辑")
print("=" * 60)
print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
print(f"SENTENCE_TRANSFORMERS_HOME: {os.environ.get('SENTENCE_TRANSFORMERS_HOME')}")
print()

# 测试方法1: 使用模型名 + cache_folder
print("方法1: 使用模型名 + cache_folder")
try:
    from sentence_transformers import SentenceTransformer

    cache_folder = str(MODELS_DIR / "sentence_transformers")
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5', cache_folder=cache_folder)

    print(f"✅ 模型加载成功!")
    print(f"   模型: {type(model).__name__}")
    print(f"   维度: {model.get_sentence_embedding_dimension()}")

    # 测试编码
    test_text = "这是一个测试"
    embedding = model.encode(test_text)
    print(f"   编码形状: {embedding.shape}")

except Exception as e:
    print(f"❌ 方法1失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 测试方法2: 使用 Matcher 类
print("方法2: 使用 Matcher 类")
try:
    from core.matcher import Matcher

    matcher = Matcher('BAAI/bge-large-zh-v1.5')
    print(f"✅ Matcher 初始化成功!")
    print(f"   模型: {type(matcher.model).__name__ if matcher.model else 'None'}")

    if matcher.model:
        test_texts = ["测试1", "测试2"]
        embeddings = matcher.model.encode(test_texts)
        print(f"   编码形状: {embeddings.shape}")

except Exception as e:
    print(f"❌ 方法2失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 测试方法3: 使用 Planner 类
print("方法3: 使用 SequencePlanner 类")
try:
    from core.sequence_planner import SequencePlanner

    planner = SequencePlanner()
    print(f"✅ SequencePlanner 初始化成功!")
    print(f"   模型: {type(planner.model).__name__ if planner.model else 'None'}")

    if planner.model:
        test_texts = ["测试1", "测试2"]
        embeddings = planner.encode(test_texts)
        print(f"   编码形状: {embeddings.shape}")

except Exception as e:
    print(f"❌ 方法3失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("所有测试完成")
print("=" * 60)
