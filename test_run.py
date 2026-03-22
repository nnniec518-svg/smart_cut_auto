#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试运行脚本"""
import sys
sys.path.insert(0, '.')

print("测试导入模块...")

try:
    from db.models import Database
    print("[OK] db.models 导入成功")
except Exception as e:
    print(f"[FAIL] db.models 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from core.processor import VideoPurifier
    print("[OK] core.processor 导入成功")
except Exception as e:
    print(f"[FAIL] core.processor 导入失败: {e}")
    sys.exit(1)

try:
    from core.planner import SequencePlanner
    print("[OK] core.planner 导入成功")
except Exception as e:
    print(f"[FAIL] core.planner 导入失败: {e}")
    sys.exit(1)

try:
    from core.auto_cutter import VideoAutoCutter
    print("[OK] core.auto_cutter 导入成功")
except Exception as e:
    print(f"[FAIL] core.auto_cutter 导入失败: {e}")
    sys.exit(1)

print("\n所有模块导入成功！")
