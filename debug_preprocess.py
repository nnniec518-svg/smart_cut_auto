"""调试文本预处理"""
import re
from core.matcher import Matcher

matcher = Matcher()

test_pairs = [
    ("全屋智能家电套购", "全 益 一 全 屋 智 能 家 电 套 购"),
    ("送至高1499元豪礼", "送 志 高 送 志 高 一 九 九 九 元 好 礼"),
    ("美的四大权益都可以享受", "我 再 问 一 下 不 用 问 美 的 四 大 权 益 都 可 以 享 受"),
]

for target, asr_text in test_pairs:
    # 预处理
    t1 = matcher._preprocess_text(target)
    t2 = matcher._preprocess_text(asr_text)
    print(f"目标原始: {target}")
    print(f"目标预处理: {t1}")
    print(f"ASR原始: {asr_text}")
    print(f"ASR预处理: {t2}")
    
    # 分词
    w1 = matcher._tokenize(target)
    w2 = matcher._tokenize(asr_text)
    print(f"目标词: {w1}")
    print(f"ASR词: {w2}")
    
    # 相似度
    sim = matcher.text_similarity(target, asr_text)
    print(f"相似度: {sim}")
    print("-" * 40)
