"""调试相似度"""
import json
from pathlib import Path
from core.matcher import Matcher

matcher = Matcher()

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
                materials_sentences.append(text)

# 文案句子
SCRIPT = """不用确定我们京东315活动3C、家电政府补贴至高15%，还能跟美的代金券叠加使用
我再问一下
不用问,美的四大权益都可以享受
权益一
全屋智能家电套购
送至高1499元豪礼
权益二
空气机及无风感系列空调一价全包
至高省1399元
权益三
购指定型号享最高
价值3200元局改服务
权益四
上抖音团购领
美的专属优惠券"""

target_sentences = matcher._split_sentences(SCRIPT)

print(f"素材句子数: {len(materials_sentences)}")
print(f"文案句子数: {len(target_sentences)}")

# 测试几个关键匹配
print("\n测试关键匹配:")
test_targets = [
    "权益三",
    "购指定型号享最高",
    "价值3200元局改服务",
    "权益四",
    "上抖音团购领",
]

for target in test_targets:
    print(f"\n目标: {target}")
    best_sim = 0
    best_text = ""
    for mat_text in materials_sentences:
        sim = matcher.text_similarity(target, mat_text)
        if sim > best_sim:
            best_sim = sim
            best_text = mat_text
    
    print(f"  最佳匹配: {best_text[:40]}...")
    print(f"  相似度: {best_sim:.3f}")
