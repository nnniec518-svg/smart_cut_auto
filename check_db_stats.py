#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查数据库中的素材分类统计
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.models import Database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_cut")

print("=" * 60)
print("数据库素材分类统计")
print("=" * 60)

db = Database("storage/materials.db")
session = db.get_session()

from db.models import Asset

# 查询所有素材
assets = session.query(Asset).all()

# 统计
stats = {
    "total": 0,
    "A_ROLL": 0,
    "B_ROLL": 0,
    "quality_status": {}
}

for asset in assets:
    stats["total"] += 1
    stats[asset.track_type] += 1
    
    status = asset.quality_status
    if status not in stats["quality_status"]:
        stats["quality_status"][status] = 0
    stats["quality_status"][status] += 1

print(f"总素材数: {stats['total']}")
print(f"A_ROLL: {stats['A_ROLL']}")
print(f"B_ROLL: {stats['B_ROLL']}")
print()
print("质量状态分布:")
for status, count in sorted(stats["quality_status"].items()):
    print(f"  {status}: {count}")

print()
print("=" * 60)
print("废片过滤示例 (前10个 invalid_junk):")
print("=" * 60)

junk_assets = session.query(Asset).filter_by(quality_status="invalid_junk").limit(10).all()
if junk_assets:
    for asset in junk_assets:
        print(f"  {asset.file_name}")
        print(f"    轨道: {asset.track_type}")
        print(f"    评分: {asset.a_roll_score:.1f}")
        print(f"    文本: {asset.asr_text[:50] if asset.asr_text else 'N/A'}...")
        print()
else:
    print("  未找到 invalid_junk 素材")
    print()
    print("这可能是因为:")
    print("  1. 素材在废片判断前已被其他规则标记")
    print("  2. 还没有足够的素材被处理")

print("=" * 60)
