"""调试jieba分词"""
import jieba
import re
from pathlib import Path
import json

# 过滤词
FILTER_WORDS = {'走', '嗯', '啊', '呀', '哦', '哈', '嘿', '喂', '呃', '那个', '这个', '然后', '就是', '这个那个', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '一二三', '一二三四五', '321', '三二一', '胜', '可以', '好的', 'OK', 'ok'}

# 读取素材缓存
cache_dir = Path("storage/material_cache")
materials_sentences = []

for f in sorted(cache_dir.glob("*.JSON")):
    with open(f, 'r', encoding='utf-8') as fp:
        cache = json.load(fp)
    asr_result = cache.get('asr_result', cache)
    if asr_result and asr_result.get('segments'):
        for seg in asr_result['segments']:
            text = seg.get('text', '')
            if text:
                # 去除空格
                text = text.replace(' ', '')
                materials_sentences.append(text)
                # 打印jieba分词结果
                words = list(jieba.cut(text))
                words_filtered = [w for w in words if w and len(w) > 1 and w not in FILTER_WORDS]
                print(f"原始: {text[:50]}")
                print(f"分词: {words}")
                print(f"过滤后: {words_filtered}")
                print("-" * 40)
