#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查ASR文本的字符编码"""
import sqlite3

conn = sqlite3.connect('storage/materials.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 获取一个包含"权益"的素材
cursor.execute("SELECT file_name, asr_text FROM assets WHERE file_name = 'IMG_8723'")
row = cursor.fetchone()

print(f"文件名: {row['file_name']}")
print(f"ASR文本: {row['asr_text']}")
print(f"\n字符编码分析:")

# 找到"权益"的位置
text = row['asr_text']
idx = text.find('权')
print(f"'权' 在位置 {idx}, Unicode: U+{ord('权'):04X}")

# 检查"权"和"益"之间的字符
if idx >= 0 and idx + 2 < len(text):
    print(f"'权' 后面的字符: '{text[idx+1]}' (U+{ord(text[idx+1]):04X})")
    print(f"'益' 在位置 {idx+2}, Unicode: U+{ord('益'):04X}")

# 显示所有空白字符
print(f"\n所有空白字符:")
for i, char in enumerate(text):
    if char in [' ', '\t', '\n', '\r']:
        print(f"  位置 {i}: U+{ord(char):04X} '{char}'")

conn.close()
