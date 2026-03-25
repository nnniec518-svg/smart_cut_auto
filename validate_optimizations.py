#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化验证脚本
验证所有中期优化步骤是否已正确实现
"""
import sys
import os

# 设置控制台输出编码（Windows）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from core.sequence_planner import SequencePlanner


def check_model_upgrade():
    """检查模型升级"""
    print("=" * 60)
    print("[步骤 1] 检查中文向量模型升级")
    print("=" * 60)

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model = config.get('models', {}).get('embedding_model', '')
    expected_model = 'BAAI/bge-large-zh-v1.5'

    if model == expected_model:
        print(f"✓ 模型已升级: {model}")
        return True
    else:
        print(f"✗ 模型未升级: 当前={model}, 期望={expected_model}")
        return False


def check_number_extraction():
    """检查中文数字提取功能"""
    print("\n" + "=" * 60)
    print("[步骤 2] 检查中文数字提取功能")
    print("=" * 60)

    test_cases = [
        ("一三九九", {"1399"}),
        ("三幺五", {"315"}),
        ("权益四", {"4"}),
        ("权益四胜", {"4"}),
        ("315", {"315"}),
        ("价格1399", {"1399"}),
    ]

    all_passed = True
    for text, expected in test_cases:
        result = SequencePlanner._extract_numbers(text)
        if result == expected:
            print(f"✓ {text} → {result}")
        else:
            print(f"✗ {text} → {result} (期望: {expected})")
            all_passed = False

    return all_passed


def check_keywords():
    """检查本地生活关键词配置"""
    print("\n" + "=" * 60)
    print("[步骤 3] 检查本地生活关键词配置")
    print("=" * 60)

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    keywords = config.get('local_life_keywords', {})
    all_passed = True

    # 检查门店名
    store_names = keywords.get('store_names', [])
    expected_stores = ["京东电器", "小天鹅", "美的", "格力", "海尔"]
    for store in expected_stores:
        if store in store_names:
            print(f"✓ 门店名已包含: {store}")
        else:
            print(f"✗ 门店名缺失: {store}")
            all_passed = False

    # 检查活动名
    events = keywords.get('events', [])
    expected_events = ["火三月", "315活动", "双11", "618"]
    for event in expected_events:
        if event in events:
            print(f"✓ 活动名已包含: {event}")
        else:
            print(f"✗ 活动名缺失: {event}")
            all_passed = False

    # 检查产品
    products = keywords.get('products', [])
    expected_products = ["洗烘套装", "空调", "冰箱", "洗衣机"]
    for product in expected_products:
        if product in products:
            print(f"✓ 产品已包含: {product}")
        else:
            print(f"✗ 产品缺失: {product}")
            all_passed = False

    return all_passed


def check_soft_number_constraint():
    """检查软数字约束实现"""
    print("\n" + "=" * 60)
    print("[步骤 4] 检查软数字约束实现")
    print("=" * 60)

    with open('core/sequence_planner.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否实现了软约束（不是硬排除）
    has_soft_constraint = '数字不匹配，应用软约束惩罚' in content
    has_hard_exclude = '数字不匹配，直接返回' in content

    if has_soft_constraint and not has_hard_exclude:
        print("✓ 已实现软数字约束（数字不匹配时降低分数，不直接排除）")
        return True
    else:
        print("✗ 软数字约束实现不正确")
        if has_hard_exclude:
            print("  - 仍在使用硬排除逻辑")
        if not has_soft_constraint:
            print("  - 缺少软约束逻辑")
        return False


def check_weights():
    """检查相似度权重配置"""
    print("\n" + "=" * 60)
    print("[步骤 5] 检查相似度权重配置")
    print("=" * 60)

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    weights = config.get('match_weights', {})
    expected = {
        'keyword': 0.20,
        'semantic': 0.70,
        'edit': 0.10
    }

    all_passed = True
    for key, expected_value in expected.items():
        actual_value = weights.get(key)
        if actual_value == expected_value:
            print(f"✓ {key} = {actual_value}")
        else:
            print(f"✗ {key} = {actual_value} (期望: {expected_value})")
            all_passed = False

    return all_passed


def check_correction_dict():
    """检查纠错词典"""
    print("\n" + "=" * 60)
    print("[步骤 6] 检查纠错词典配置")
    print("=" * 60)

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    corrections = config.get('correction_dict', {})
    test_cases = [
        ("洗红", "洗烘"),
        ("美丽奥", "美的"),
        ("权益四胜", "权益四"),
        ("三幺五", "315"),
    ]

    all_passed = True
    for wrong, correct in test_cases:
        if corrections.get(wrong) == correct:
            print(f"✓ 纠错: {wrong} → {correct}")
        else:
            print(f"✗ 纠错缺失: {wrong} → {correct}")
            all_passed = False

    return all_passed


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("中期优化验证")
    print("=" * 60)

    results = []

    # 执行所有检查
    results.append(("模型升级", check_model_upgrade()))
    results.append(("中文数字提取", check_number_extraction()))
    results.append(("关键词配置", check_keywords()))
    results.append(("软数字约束", check_soft_number_constraint()))
    results.append(("相似度权重", check_weights()))
    results.append(("纠错词典", check_correction_dict()))

    # 输出总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {name}")

    print("\n" + "=" * 60)
    print(f"总计: {passed}/{total} 项检查通过")
    print("=" * 60)

    if passed == total:
        print("\n✓ 所有优化步骤已正确实现！")
        return 0
    else:
        print("\n✗ 部分优化步骤未完成，请检查配置")
        return 1


if __name__ == "__main__":
    sys.exit(main())
