#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修正后的数据库分析"""
import sqlite3
import json

conn = sqlite3.connect('storage/materials.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 统计包含关键词的素材 (考虑空格分隔)
keywords = ['权益', '洗烘', '代金券', '补贴', '京东', '315', '美的', '小天鹅']
print("=== 关键词统计 (考虑空格) ===")
for keyword in [f' {w} ' for w in keywords]:
    cursor.execute("SELECT COUNT(*) FROM assets WHERE asr_text LIKE ?", (f'%{keyword}%',))
    count = cursor.fetchone()[0]
    print(f"'{keyword.strip()}': {count} 个素材包含")

# 检查数字
print("\n=== 数字相关素材 ===")
for num in ['一三九九', '一四九九', '三二零零', '三幺五']:
    cursor.execute("SELECT file_name, asr_text FROM assets WHERE asr_text LIKE ?", (f'%{num}%',))
    results = cursor.fetchall()
    print(f"\n'{num}': {len(results)} 个素材")
    for row in results[:2]:  # 只显示前2个
        print(f"  {row['file_name']}: ...{row['asr_text'][row['asr_text'].find(num)-10:row['asr_text'].find(num)+20]}...")

# 检查带空格的关键词
cursor.execute("SELECT file_name, asr_text FROM assets WHERE asr_text LIKE '% 权 %' OR asr_text LIKE '% 权 益 %'")
print("\n=== 包含'权益'的素材 ===")
for row in cursor.fetchall():
    print(f"{row['file_name']}: {row['asr_text'][:60]}")

conn.close()
