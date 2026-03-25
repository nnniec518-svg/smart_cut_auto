#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更新配置文件以使用 BAAI/bge-large-zh-v1.5 模型
"""
import yaml
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.yaml"

print("=" * 60)
print("更新配置文件")
print("=" * 60)

# 读取配置
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 更新模型配置
print("\n当前配置:")
print(f"  embedding_model: {config['models']['embedding_model']}")
print(f"  fallback_embedding: {config['models']['fallback_embedding']}")

# 更新为新的中文模型
config['models']['embedding_model'] = "BAAI/bge-large-zh-v1.5"
config['models']['fallback_embedding'] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("\n新配置:")
print(f"  embedding_model: {config['models']['embedding_model']}")
print(f"  fallback_embedding: {config['models']['fallback_embedding']}")

# 备份原配置
backup_file = CONFIG_FILE.with_suffix('.yaml.bak')
with open(backup_file, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
print(f"\n✓ 已备份原配置到: {backup_file}")

# 写入新配置
with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

print(f"✓ 已更新配置文件: {CONFIG_FILE}")

print("\n" + "=" * 60)
print("配置更新完成！")
print("=" * 60)
print("\n下一步:")
print("1. 运行测试: python test_model.py")
print("2. 运行主程序: python main.py")
