#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用国内镜像下载 BAAI/bge-large-zh-v1.5 模型
"""
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = MODELS_DIR / "sentence_transformers"

print("=" * 60)
print("下载 BAAI/bge-large-zh-v1.5 模型")
print("=" * 60)
print(f"镜像源: {os.environ.get('HF_ENDPOINT', 'default')}")
print(f"缓存目录: {CACHE_DIR}")
print()

try:
    from sentence_transformers import SentenceTransformer

    model_name = "BAAI/bge-large-zh-v1.5"

    print("开始下载模型...")
    print("这可能需要几分钟时间，请耐心等待...")
    print()

    # 下载模型到本地缓存
    model = SentenceTransformer(
        model_name,
        cache_folder=str(CACHE_DIR),
        device="cpu"
    )

    print()
    print("=" * 60)
    print("✓ 模型下载成功！")
    print("=" * 60)
    print(f"模型名称: {model_name}")
    print(f"向量维度: {model.get_sentence_embedding_dimension()}")
    print(f"缓存位置: {CACHE_DIR}")
    print()

    # 测试模型
    print("测试模型...")
    test_texts = ["京东电器火三月", "洗烘套装补贴价"]
    embeddings = model.encode(test_texts)
    print(f"✓ 编码测试成功: {embeddings.shape}")

    sim = model.similarity(embeddings[0:1], embeddings[1:2])
    print(f"✓ 相似度计算成功: {sim[0][0]:.3f}")

    print()
    print("=" * 60)
    print("模型已准备就绪，可以使用了！")
    print("=" * 60)
    print()
    print("下一步：")
    print("1. 更新 config.yaml 中的 embedding_model")
    print("2. 重新运行主程序")

except Exception as e:
    print()
    print("=" * 60)
    print("✗ 下载失败")
    print("=" * 60)
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("可能的原因：")
    print("1. 网络连接问题")
    print("2. 镜像源不可用")
    print("3. 磁盘空间不足")
    print()
    print("建议：")
    print("1. 检查网络连接")
    print("2. 尝试使用 VPN")
    print("3. 或继续使用本地已有的 paraphrase-multilingual-MiniLM-L12-v2 模型")
