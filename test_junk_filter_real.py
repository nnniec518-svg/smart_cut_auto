#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实际素材测试 - 验证废片过滤
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.processor import VideoPurifier
from db.models import Database
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smart_cut")

print("=" * 60)
print("实际素材测试 - 验证废片过滤")
print("=" * 60)
print()

# 初始化
db = Database("storage/materials.db")
purifier = VideoPurifier(db)

# 测试素材目录
test_dir = "storage/materials-all"
import glob

# 获取前10个素材进行测试
video_files = sorted(glob.glob(os.path.join(test_dir, "*.MOV")))[:10]

print(f"找到 {len(video_files)} 个测试素材")
print()

# 统计
results = {
    "A_ROLL": 0,
    "B_ROLL": 0,
    "invalid_junk": 0,
    "invalid_noise": 0,
    "invalid_short": 0,
    "valid": 0
}

# 处理每个素材
for i, video_path in enumerate(video_files, 1):
    print(f"[{i}/{len(video_files)}] 处理: {os.path.basename(video_path)}")

    # 强制重新处理
    asset = purifier.purify(video_path, force_reprocess=True)

    # 统计
    results[asset.track_type] += 1
    if hasattr(asset, 'quality_status'):
        status = asset.quality_status
        if status in results:
            results[status] += 1
        else:
            results[status] = 1

    print(f"  结果: {asset.track_type}, 质量: {asset.quality_status}")
    print()

# 打印统计
print("=" * 60)
print("统计结果")
print("=" * 60)
print(f"A_ROLL (有效主轨):  {results['A_ROLL']}")
print(f"B_ROLL (辅助素材):  {results['B_ROLL']}")
print()
print(f"质量分类:")
print(f"  - valid (有效):      {results.get('valid', 0)}")
print(f"  - invalid_junk (废片): {results.get('invalid_junk', 0)}")
print(f"  - invalid_noise (噪声): {results.get('invalid_noise', 0)}")
print(f"  - invalid_short (过短): {results.get('invalid_short', 0)}")
print("=" * 60)

# 检查数据库中的实际分类
print()
print("数据库验证 - 查询素材分类:")
print("=" * 60)

session = db.get_session()
from db.models import Asset

# 查询所有素材
assets = session.query(Asset).limit(10).all()

for asset in assets:
    print(f"  {asset.file_name}: {asset.track_type} (质量: {asset.quality_status}, 评分: {asset.a_roll_score:.1f})")

print("=" * 60)
print("测试完成！")
