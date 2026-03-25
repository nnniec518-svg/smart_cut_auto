#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析数据库素材分布"""
import sqlite3
import json

conn = sqlite3.connect('storage/materials.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 统计 A_ROLL 和 B_ROLL 数量
cursor.execute("SELECT track_type, COUNT(*) as count FROM assets GROUP BY track_type")
print("=== 素材类型分布 ===")
for row in cursor.fetchall():
    print(f"{row['track_type']}: {row['count']}")

# 统计 A_ROLL 素材的 ASR 文本长度
cursor.execute("""
    SELECT file_name, track_type, asr_text, transcript_json
    FROM assets
    WHERE track_type = 'A_ROLL'
    ORDER BY file_name
""")
print("\n=== A_ROLL 素材详情 ===")
for row in cursor.fetchall():
    print(f"\n{row['file_name']}:")
    print(f"  ASR Text: {row['asr_text'][:100] if row['asr_text'] else '(empty)'}")
    if row['transcript_json']:
        try:
            data = json.loads(row['transcript_json'])
            segments = data.get('segments', [])
            print(f"  Segments: {len(segments)}")
            for seg in segments[:3]:  # 只显示前3个片段
                print(f"    [{seg.get('start', 0):.2f}-{seg.get('end', 0):.2f}] {seg.get('text', '')[:30]}")
        except:
            pass

# 检查是否包含关键词
keywords = ['权益', '洗烘', '代金券', '补贴', '京东', '315', '美的', '小天鹅']
print("\n=== 关键词统计 ===")
for keyword in keywords:
    cursor.execute("SELECT COUNT(*) FROM assets WHERE asr_text LIKE ?", (f'%{keyword}%',))
    count = cursor.fetchone()[0]
    print(f"'{keyword}': {count} 个素材包含")

# 检查数字相关
cursor.execute("SELECT file_name, asr_text FROM assets WHERE asr_text LIKE '%1399%' OR asr_text LIKE '%1499%' OR asr_text LIKE '%3200%'")
print("\n=== 包含关键金额的素材 ===")
for row in cursor.fetchall():
    print(f"{row['file_name']}: {row['asr_text'][:80]}")

conn.close()
