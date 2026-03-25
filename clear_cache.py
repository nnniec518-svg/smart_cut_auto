#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""清理缓存和数据库"""
import os
import shutil
import sys
from pathlib import Path

# 设置控制台输出编码（Windows）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent
TEMP_DIR = PROJECT_ROOT / "temp"
DB_FILE = PROJECT_ROOT / "storage" / "materials.db"

print("=" * 60)
print("清理缓存和数据库")
print("=" * 60)

# 清理临时文件
if TEMP_DIR.exists():
    print(f"\n清理临时文件: {TEMP_DIR}")
    for item in TEMP_DIR.iterdir():
        if item.is_file():
            item.unlink()
            print(f"  删除文件: {item.name}")
        elif item.is_dir():
            shutil.rmtree(item)
            print(f"  删除目录: {item.name}")
    print("OK 临时文件清理完成")
else:
    print("\n临时文件目录不存在")

# 删除数据库
if DB_FILE.exists():
    print(f"\n删除数据库: {DB_FILE}")
    DB_FILE.unlink()
    print("OK 数据库删除完成")
else:
    print("\n数据库文件不存在")


print("\n" + "=" * 60)
print("清理完成！")
print("=" * 60)
