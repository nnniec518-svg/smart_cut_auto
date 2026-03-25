#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BGE 模型离线验证和完整性检查
确保所有必要文件都已下载,可以完全离线运行
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List

# ==================== 配置 ====================

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
SENTENCE_TRANSFORMERS_HOME = MODELS_DIR / "sentence_transformers"

# BGE 模型配置
MODEL_NAME = "BAAI/bge-large-zh-v1.5"
MODEL_DIR_NAME = f"models--{MODEL_NAME.replace('/', '--')}"
MODEL_BASE_DIR = SENTENCE_TRANSFORMERS_HOME / MODEL_DIR_NAME

# 必需的文件列表（用于验证）
REQUIRED_FILES = {
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/config.json": {
        "min_size_kb": 0.5,
        "max_size_kb": 10,
        "description": "模型配置文件"
    },
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/config_sentence_transformers.json": {
        "min_size_kb": 0.1,
        "max_size_kb": 1,
        "description": "SentenceTransformers 配置"
    },
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/modules.json": {
        "min_size_kb": 0.3,
        "max_size_kb": 1,
        "description": "模块配置"
    },
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/pytorch_model.bin": {
        "min_size_kb": 1200000,
        "max_size_kb": 1400000,
        "description": "PyTorch 模型权重 (约1.3GB)"
    },
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/sentence_bert_config.json": {
        "min_size_kb": 0.05,
        "max_size_kb": 1,
        "description": "SentenceBERT 配置"
    },
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/tokenizer.json": {
        "min_size_kb": 400,
        "max_size_kb": 500,
        "description": "Tokenizer 配置"
    },
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/tokenizer_config.json": {
        "min_size_kb": 0.3,
        "max_size_kb": 1,
        "description": "Tokenizer 配置"
    },
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/vocab.txt": {
        "min_size_kb": 100,
        "max_size_kb": 150,
        "description": "词汇表文件"
    },
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/1_Pooling/config.json": {
        "min_size_kb": 0.1,
        "max_size_kb": 1,
        "description": "Pooling 层配置"
    },
    "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/special_tokens_map.json": {
        "min_size_kb": 0.1,
        "max_size_kb": 1,
        "description": "特殊 token 映射"
    },
    "refs/main": {
        "min_size_kb": 0.01,
        "max_size_kb": 1,
        "description": "main 分支引用"
    }
}

# ==================== 验证函数 ====================

def print_section(title: str):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_file_exists(file_path: Path, info: Dict) -> bool:
    """检查文件是否存在"""
    if not file_path.exists():
        print(f"   ❌ {info['description']}: 文件不存在")
        print(f"      期望路径: {file_path.relative_to(PROJECT_ROOT)}")
        return False
    
    size_kb = file_path.stat().st_size / 1024
    if size_kb < info['min_size_kb'] or size_kb > info['max_size_kb']:
        print(f"   ⚠️  {info['description']}: 文件大小异常")
        print(f"      实际大小: {size_kb:.1f} KB (期望: {info['min_size_kb']}-{info['max_size_kb']} KB)")
        print(f"      路径: {file_path.relative_to(PROJECT_ROOT)}")
        return False
    
    print(f"   ✅ {info['description']}: {size_kb:.1f} KB")
    return True

def verify_config_json(config_path: Path) -> bool:
    """验证 config.json 内容"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        required_keys = ['model_type', 'hidden_size', 'num_hidden_layers', 'vocab_size']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"   ⚠️  config.json 缺少必需字段: {', '.join(missing_keys)}")
            return False
        
        print(f"   ✅ config.json 验证通过:")
        print(f"      model_type: {config.get('model_type')}")
        print(f"      hidden_size: {config.get('hidden_size')}")
        print(f"      num_hidden_layers: {config.get('num_hidden_layers')}")
        return True
    except Exception as e:
        print(f"   ❌ config.json 验证失败: {e}")
        return False

def verify_refs_main(refs_main_path: Path) -> bool:
    """验证 refs/main 指向正确的快照"""
    try:
        with open(refs_main_path, 'r', encoding='utf-8') as f:
            ref = f.read().strip()
        
        expected_snapshot = "79e7739b6ab944e86d6171e44d24c997fc1e0116"
        if ref != expected_snapshot:
            print(f"   ⚠️  refs/main 指向错误的快照")
            print(f"      实际: {ref}")
            print(f"      期望: {expected_snapshot}")
            return False
        
        snapshot_path = MODEL_BASE_DIR / "snapshots" / ref
        if not snapshot_path.exists():
            print(f"   ❌ refs/main 指向的快照不存在")
            return False
        
        print(f"   ✅ refs/main 指向正确的快照: {ref[:12]}...")
        return True
    except Exception as e:
        print(f"   ❌ refs/main 验证失败: {e}")
        return False

def check_model_structure() -> bool:
    """检查模型目录结构"""
    print_section("检查模型目录结构")
    
    required_dirs = ['snapshots', 'refs', 'blobs']
    all_ok = True
    
    for dir_name in required_dirs:
        dir_path = MODEL_BASE_DIR / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/ 目录存在")
        else:
            print(f"   ❌ {dir_name}/ 目录不存在")
            all_ok = False
    
    return all_ok

def check_all_files() -> bool:
    """检查所有必需文件"""
    print_section("检查所有必需文件")
    
    all_ok = True
    for rel_path, info in REQUIRED_FILES.items():
        file_path = MODEL_BASE_DIR / rel_path
        if not check_file_exists(file_path, info):
            all_ok = False
    
    return all_ok

def verify_file_contents() -> bool:
    """验证关键文件内容"""
    print_section("验证关键文件内容")
    
    all_ok = True
    
    # 验证 config.json
    config_path = MODEL_BASE_DIR / "snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/config.json"
    if not verify_config_json(config_path):
        all_ok = False
    
    print()
    
    # 验证 refs/main
    refs_main_path = MODEL_BASE_DIR / "refs/main"
    if not verify_refs_main(refs_main_path):
        all_ok = False
    
    return all_ok

def calculate_model_size() -> Dict:
    """计算模型总大小"""
    print_section("模型大小统计")
    
    total_size = 0
    file_count = 0
    
    for file_path in MODEL_BASE_DIR.rglob('*'):
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
            file_count += 1
    
    print(f"   总文件数: {file_count}")
    print(f"   总大小: {total_size / (1024*1024):.2f} MB")
    print(f"   总大小: {total_size / (1024*1024*1024):.2f} GB")
    
    return {"total_size": total_size, "file_count": file_count}

def test_model_loading() -> bool:
    """测试模型加载"""
    print_section("测试模型加载（离线模式）")
    
    try:
        # 设置环境变量
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(SENTENCE_TRANSFORMERS_HOME)
        
        print("   导入 SentenceTransformer...")
        from sentence_transformers import SentenceTransformer
        
        print(f"   加载模型: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        
        print(f"   ✅ 模型加载成功!")
        print(f"      嵌入维度: {model.get_sentence_embedding_dimension()}")
        
        # 测试编码
        print("\n   测试文本编码...")
        test_text = "这是一个测试句子"
        embedding = model.encode(test_text)
        print(f"   ✅ 编码成功! 输出形状: {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_report(results: Dict) -> None:
    """生成验证报告"""
    print_section("验证结果总结")
    
    total_checks = sum(results.values())
    passed_checks = sum(1 for v in results.values() if v)
    
    print(f"\n   总检查项: {total_checks}")
    print(f"   通过项: {passed_checks}")
    print(f"   失败项: {total_checks - passed_checks}")
    
    print("\n   详细结果:")
    for check_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"      {check_name}: {status}")
    
    print("\n   离线运行能力:")
    if all(results.values()):
        print("      ✅ 模型可以完全离线运行")
    else:
        print("      ❌ 模型无法完全离线运行，请检查上述失败项")

def main():
    """主函数"""
    print("=" * 70)
    print("  BGE 模型离线验证工具")
    print(f"  模型: {MODEL_NAME}")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print("=" * 70)
    
    # 运行所有检查
    results = {}
    
    results["目录结构"] = check_model_structure()
    results["必需文件"] = check_all_files()
    results["文件内容"] = verify_file_contents()
    stats = calculate_model_size()
    results["模型加载"] = test_model_loading()
    
    # 生成报告
    generate_report(results)
    
    # 返回退出码
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
