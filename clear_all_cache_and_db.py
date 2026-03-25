#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清除所有缓存文件和数据库文件
"""
import os
import sys
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

print("=" * 60)
print("清除缓存和数据库")
print("=" * 60)

# 1. 清除 temp 目录
temp_dir = PROJECT_ROOT / "temp"
if temp_dir.exists():
    print(f"\n清除临时目录: {temp_dir}")
    for item in temp_dir.iterdir():
        try:
            if item.is_file():
                item.unlink()
                print(f"  删除文件: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"  删除目录: {item.name}")
        except Exception as e:
            print(f"  删除失败 {item.name}: {e}")
else:
    print(f"\n临时目录不存在: {temp_dir}")

# 2. 清除 logs 目录
logs_dir = PROJECT_ROOT / "logs"
if logs_dir.exists():
    print(f"\n清除日志目录: {logs_dir}")
    for item in logs_dir.iterdir():
        try:
            if item.is_file():
                item.unlink()
                print(f"  删除日志: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"  删除目录: {item.name}")
        except Exception as e:
            print(f"  删除失败 {item.name}: {e}")
else:
    print(f"\n日志目录不存在: {logs_dir}")

# 3. 清除 Python 缓存 (__pycache__)
print(f"\n清除 Python 缓存目录")
pycache_count = 0
for pycache_path in PROJECT_ROOT.rglob("__pycache__"):
    try:
        shutil.rmtree(pycache_path)
        print(f"  删除: {pycache_path.relative_to(PROJECT_ROOT)}")
        pycache_count += 1
    except Exception as e:
        print(f"  删除失败 {pycache_path}: {e}")

if pycache_count == 0:
    print("  没有找到 __pycache__ 目录")

# 4. 清除 .pyc 文件
print(f"\n清除 .pyc 文件")
pyc_count = 0
for pyc_file in PROJECT_ROOT.rglob("*.pyc"):
    try:
        pyc_file.unlink()
        print(f"  删除: {pyc_file.relative_to(PROJECT_ROOT)}")
        pyc_count += 1
    except Exception as e:
        print(f"  删除失败 {pyc_file}: {e}")

if pyc_count == 0:
    print("  没有找到 .pyc 文件")

# 5. 清除数据库文件
print(f"\n清除数据库文件")
db_files = list(PROJECT_ROOT.rglob("*.db"))
for db_file in db_files:
    try:
        db_file.unlink()
        print(f"  删除数据库: {db_file.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"  删除失败 {db_file}: {e}")

if not db_files:
    print("  没有找到数据库文件")

print("\n" + "=" * 60)
print("清理完成！")
print("=" * 60)
