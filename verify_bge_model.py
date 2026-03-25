#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 BAAI/bge-large-zh-v1.5 模型是否正确下载和配置
"""
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = MODELS_DIR / "sentence_transformers"

print("=" * 60)
print("验证 BAAI/bge-large-zh-v1.5 模型")
print("=" * 60)
print(f"缓存目录: {CACHE_DIR}")
print()

# 检查本地缓存
model_cache = CACHE_DIR / "models--BAAI--bge-large-zh-v1.5"
print(f"检查本地缓存: {model_cache}")
if model_cache.exists():
    print("✓ 本地缓存目录存在")
    print("  文件列表:")
    for item in model_cache.rglob("*"):
        if item.is_file() and not any(x in item.parts for x in ['.git', 'snapshots']):
            size = item.stat().st_size / 1024
            print(f"    - {item.relative_to(model_cache)} ({size:.2f} KB)")
else:
    print("✗ 本地缓存目录不存在")
    sys.exit(1)

print()

try:
    from sentence_transformers import SentenceTransformer
    import yaml

    # 读取配置
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config.get("models", {}).get("embedding_model", "")
    print(f"配置文件中的模型: {model_name}")
    print()

    if "bge-large-zh-v1.5" in model_name.lower():
        print("✓ 配置文件中已设置为 BGE 模型")
    else:
        print(f"✗ 配置文件中的模型不是 BGE: {model_name}")

    print()
    print("尝试加载模型...")

    # 尝试加载模型
    model = SentenceTransformer(
        model_name,
        cache_folder=str(CACHE_DIR),
        device="cpu"
    )

    print()
    print("✓ 模型加载成功!")
    print(f"  模型名称: {model_name}")
    print(f"  向量维度: {model.get_sentence_embedding_dimension()}")
    print()

    # 测试模型
    print("测试编码功能...")
    test_texts = ["京东电器火三月", "洗烘套装补贴价"]
    embeddings = model.encode(test_texts)
    print(f"✓ 编码测试成功: {embeddings.shape}")

    print()
    print("测试相似度计算...")
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"✓ 相似度计算成功: {sim:.3f}")

    print()
    print("=" * 60)
    print("✓ 模型验证通过！")
    print("=" * 60)

except Exception as e:
    print()
    print("=" * 60)
    print("✗ 验证失败")
    print("=" * 60)
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
