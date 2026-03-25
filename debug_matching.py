#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试匹配过程"""
from core.sequence_planner import SequencePlanner
import logging

logging.basicConfig(level=logging.DEBUG)

# 初始化规划器
planner = SequencePlanner(db_path="storage/materials.db")

# 测试数字约束
text_a = "送至高1499元豪礼"
text_b = "志高省一三九九元"

nums_a = planner._extract_numbers(text_a)
nums_b = planner._extract_numbers(text_b)
print(f"文案: {text_a}")
print(f"  提取的数字: {nums_a}")
print(f"素材: {text_b}")
print(f"  提取的数字: {nums_b}")

match = planner._check_number_match(text_a, text_b)
print(f"\n数字匹配结果: {match}")

# 测试混合相似度
sim = planner._compute_hybrid_similarity(text_a, text_b)
print(f"混合相似度: {sim:.4f}")
