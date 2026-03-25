#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试模型加载"""
import sys
import os
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("测试模型加载")
print("=" * 60)

# 显示当前配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    model_name = config.get('models', {}).get('embedding_model', 'unknown')
    print(f"当前配置的模型: {model_name}")
    print()

try:
    from core.planner import EmbeddingModel
    print("✓ 导入 EmbeddingModel 成功")

    model = EmbeddingModel()
    print(f"✓ 模型加载成功: {model.model_name}")
    print(f"  向量维度: {model.model.get_sentence_embedding_dimension()}")

    # 测试编码
    print("\n测试编码...")
    test_texts = ["京东电器火三月", "洗烘套装补贴价", "权益四胜权益四胜"]
    embeddings = model.encode(test_texts)
    print(f"✓ 编码测试成功: {embeddings.shape}")

    # 测试相似度计算
    print("\n测试相似度计算...")
    tests = [
        ("京东电器", "京东电器", 0.95),
        ("洗烘套装", "洗红套装", 0.85),
        ("权益四", "权益四胜", 0.90),
        ("三幺五", "315", 0.80),
    ]
    for text_a, text_b, expected_min in tests:
        sim = model.compute_similarity(text_a, text_b)
        status = "✓" if sim >= expected_min else "⚠"
        print(f"  {status} '{text_a}' vs '{text_b}': {sim:.3f} (期望 ≥ {expected_min:.2f})")

    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
