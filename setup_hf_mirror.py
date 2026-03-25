#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HuggingFace 镜像配置和模型管理脚本
"""

import os
import sys
from pathlib import Path

def setup_huggingface_mirror():
    """设置 HuggingFace 镜像源"""
    # 方法1: 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 方法2: 修改 HuggingFace Hub 配置文件
    huggingface_config_dir = Path.home() / '.huggingface'
    huggingface_config_file = huggingface_config_dir / 'config.json'
    
    huggingface_config_dir.mkdir(exist_ok=True)
    
    import json
    config = {
        "endpoint": "https://hf-mirror.com"
    }
    
    with open(huggingface_config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ HuggingFace 镜像配置完成")
    print(f"   镜像地址: {config['endpoint']}")
    print(f"   配置文件: {huggingface_config_file}")

def check_model_cache():
    """检查模型缓存"""
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    
    if cache_dir.exists():
        models = list(cache_dir.glob('models--*'))
        print(f"\n📦 本地已缓存的模型数量: {len(models)}")
        for model in models[:5]:  # 只显示前5个
            model_name = model.name.replace('models--', '').replace('--', '/')
            print(f"   - {model_name}")
        if len(models) > 5:
            print(f"   ... 还有 {len(models) - 5} 个模型")
    else:
        print("\n📦 本地暂无模型缓存")

def download_model_with_mirror():
    """使用镜像下载模型"""
    from sentence_transformers import SentenceTransformer
    
    print("\n🔄 正在从镜像源下载模型...")
    print("   这可能需要几分钟时间，请耐心等待...")
    
    try:
        # 尝试下载当前配置的模型
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ 模型下载成功！")
        return True
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        print("\n💡 建议:")
        print("   1. 检查网络连接")
        print("   2. 手动下载模型文件")
        print("   3. 使用本地已有的模型")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("HuggingFace 镜像配置和模型管理")
    print("=" * 60)
    
    # 设置镜像
    setup_huggingface_mirror()
    
    # 检查缓存
    check_model_cache()
    
    # 询问是否下载模型
    print("\n是否现在尝试下载模型? (y/n): ", end='')
    choice = input().strip().lower()
    
    if choice == 'y':
        download_model_with_mirror()
    else:
        print("跳过模型下载。您可以稍后运行程序时会自动下载。")
