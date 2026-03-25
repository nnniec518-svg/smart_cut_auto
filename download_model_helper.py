#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型下载与配置工具
支持多种下载方式：镜像源、手动下载、使用本地缓存
"""

import os
import json
import sys
from pathlib import Path
import urllib.request
import zipfile
import hashlib

# ==================== 配置部分 ====================

# HuggingFace 镜像源
HF_MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co-mirror.com",
]

# 模型配置
MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "url": "https://hf-mirror.com/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/pytorch_model.bin",
        "filename": "pytorch_model.bin",
        "size_mb": 435,
        "description": "多语言语义相似度模型（支持中文）"
    },
    "bge-large-zh-v1.5": {
        "url": "https://hf-mirror.com/BAAI/bge-large-zh-v1.5/resolve/main/pytorch_model.bin",
        "filename": "pytorch_model.bin",
        "size_mb": 1300,
        "description": "BAAI 大型中文嵌入模型（推荐）"
    }
}

# 缓存目录
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

# ==================== 工具函数 ====================

def create_hf_config():
    """创建 HuggingFace 配置文件"""
    hf_home = Path.home() / ".cache" / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    
    config_file = hf_home / "config.json"
    config = {
        "endpoint": "https://hf-mirror.com"
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ HuggingFace 配置文件已创建: {config_file}")
    print(f"   镜像地址: {config['endpoint']}")
    return config_file

def download_file(url, dest_path, show_progress=True):
    """下载文件并显示进度"""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    def progress_hook(count, block_size, total_size):
        if show_progress and total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r下载进度: {percent}% ({mb:.1f}MB / {total_mb:.1f}MB)", end='')
            sys.stdout.flush()
    
    try:
        print(f"\n正在下载: {url}")
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print(f"\n✅ 下载完成: {dest_path}")
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False

def verify_file(file_path, expected_size_mb=None):
    """验证文件完整性"""
    file_path = Path(file_path)
    if not file_path.exists():
        return False
    
    size_mb = file_path.stat().st_size / (1024 * 1024)
    
    if expected_size_mb:
        if abs(size_mb - expected_size_mb) / expected_size_mb > 0.1:
            print(f"⚠️ 文件大小异常: {size_mb:.1f}MB (预期约 {expected_size_mb}MB)")
            return False
    
    print(f"✅ 文件验证通过: {file_path} ({size_mb:.1f}MB)")
    return True

def find_local_models():
    """查找本地已有的模型缓存"""
    cache_dir = CACHE_DIR
    
    if not cache_dir.exists():
        print(f"❌ 缓存目录不存在: {cache_dir}")
        return []
    
    # 查找所有模型目录
    model_dirs = []
    for item in cache_dir.iterdir():
        if item.is_dir() and "models--" in item.name:
            model_name = item.name.replace("models--", "").replace("--", "/")
            model_dirs.append({
                "name": model_name,
                "path": item
            })
    
    return model_dirs

def check_model_cache():
    """检查模型缓存状态"""
    print("\n" + "=" * 60)
    print("检查本地模型缓存")
    print("=" * 60)
    
    models = find_local_models()
    
    if not models:
        print("❌ 未找到本地模型缓存")
        return None
    
    print(f"✅ 找到 {len(models)} 个模型缓存:")
    for i, model in enumerate(models, 1):
        size = sum(f.stat().st_size for f in model['path'].rglob('*') if f.is_file()) / (1024 * 1024)
        print(f"   {i}. {model['name']} ({size:.1f}MB)")
    
    return models

def download_model(model_name):
    """下载指定的模型"""
    if model_name not in MODELS:
        print(f"❌ 不支持的模型: {model_name}")
        return False
    
    model_info = MODELS[model_name]
    print(f"\n准备下载模型: {model_name}")
    print(f"描述: {model_info['description']}")
    print(f"大小: 约 {model_info['size_mb']}MB")
    
    confirm = input("\n确认下载? (y/n): ")
    if confirm.lower() != 'y':
        print("已取消下载")
        return False
    
    # 尝试从镜像源下载
    url = model_info['url']
    dest_path = CACHE_DIR / "models--" + model_name.replace("/", "--") / "snapshots" / "main" / model_info['filename']
    
    success = download_file(url, dest_path)
    
    if success:
        verify_file(dest_path, model_info['size_mb'])
        print(f"\n✅ 模型下载完成: {model_name}")
        return True
    else:
        print(f"\n❌ 模型下载失败")
        return False

def manual_download_guide():
    """手动下载指南"""
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    
    print("\n" + "=" * 60)
    print("手动下载指南")
    print("=" * 60)
    print(f"\n如果您无法自动下载，可以手动下载模型:")
    print(f"\n1. 访问 HuggingFace 镜像站:")
    print(f"   https://hf-mirror.com/sentence-transformers/{model_name}")
    print(f"\n2. 下载以下文件:")
    print(f"   - pytorch_model.bin (约 435MB)")
    print(f"   - config.json")
    print(f"   - tokenizer_config.json")
    print(f"   - vocab.txt")
    print(f"\n3. 将文件放入:")
    print(f"   {CACHE_DIR}/models--sentence-transformers--{model_name.replace('paraphrase-multilingual', '')}/snapshots/main/")
    print(f"\n或者，您可以使用国内 AI 社区提供的预打包模型:")
    print(f"   - ModelScope (魔搭社区): https://modelscope.cn/")
    print(f"   - Gitee AI: https://ai.gitee.com/")

def main():
    print("=" * 60)
    print("HuggingFace 模型下载与配置工具")
    print("=" * 60)
    
    # 检查当前状态
    models = check_model_cache()
    
    if models:
        print("\n✅ 本地已有模型缓存，无需下载")
        choice = input("\n是否仍然下载新模型? (y/n): ")
        if choice.lower() != 'y':
            return
    
    # 创建配置
    create_hf_config()
    
    print("\n可用模型:")
    for i, (name, info) in enumerate(MODELS.items(), 1):
        print(f"   {i}. {name}")
        print(f"      {info['description']} ({info['size_mb']}MB)")
    
    print("\n   0. 手动下载（查看指南）")
    
    choice = input("\n请选择 (0-2): ")
    
    if choice == "0":
        manual_download_guide()
    elif choice == "1":
        download_model("paraphrase-multilingual-MiniLM-L12-v2")
    elif choice == "2":
        download_model("bge-large-zh-v1.5")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
