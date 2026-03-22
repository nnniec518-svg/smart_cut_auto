"""测试修复后的匹配效果"""
import sys
import os
import json
import logging
from pathlib import Path
from core.asr import ASR
from core.matcher import Matcher

# 禁用ASR日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("test")

# 文案
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
美的专属优惠券
活动时间：2026年3月1日-3月31日
美的火三月福利，今天带你们薅小天鹅洗烘套装的羊毛，错过等一年！
美的50代150 还可以叠加京东活动真的是太划算了
首选这套本色洗烘套装！蓝氧护色黑科技太懂女生了——白T恤越洗越亮，彩色衣服不串色，洗完烘完直接能穿，不用再担心晒不干有异味。现在领50代150的券，叠加京东315补贴，算下来比平时便宜大几百！
重点来了！京东315还有新房全屋全套5折抢，老房焕新补贴直接省到家。买小天鹅 先用券减100，再享政府补贴，双重优惠叠加，这便宜不占白不占！
我在京东电器旗舰店等你们！想买洗烘套装的赶紧来，记得领50代150的券，叠加补贴真的超划算～"""

def main():
    # 加载素材
    materials_dir = Path("storage/materials")
    materials = []
    for f in sorted(materials_dir.glob("*.MOV"))[:20]:  # 只取前20个
        materials.append({"name": f.stem, "path": str(f)})
    
    print(f"素材数量: {len(materials)}")
    
    # 收集所有素材的句子
    all_material_sentences = []
    
    for mat in materials:
        print(f"分析: {mat['name']}")
        cache_file = Path(f"storage/material_cache/{mat['name']}.json")
        
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"  使用缓存: {cache.get('text', '')[:50]}...")
        else:
            print("  无缓存，跳过")
            continue
        
        # 检查是否有segments
        if cache and isinstance(cache, dict):
            asr_result = cache.get('asr_result', cache)
            if asr_result and asr_result.get('segments'):
                for seg in asr_result['segments']:
                    text = seg.get('text', '')
                    if text:
                        all_material_sentences.append({
                            'material_name': mat['name'],
                            'text': text,
                            'start': seg.get('start', 0),
                            'end': seg.get('end', 0)
                        })
                        print(f"    句子: {text}")
    
    print(f"\n总素材句子数: {len(all_material_sentences)}")
    
    # 分句文案
    print("\n初始化Matcher...")
    matcher = Matcher()
    target_sentences = matcher._split_sentences(SCRIPT)
    print(f"文案句子数: {len(target_sentences)}")
    
    # 测试相似度计算
    print("\n测试修复后的相似度计算:")
    
    # 测试几个关键句子
    test_pairs = [
        ("全屋智能家电套购", "全 益 一 全 屋 智 能 家 电 套 购"),
        ("送至高1499元豪礼", "送 志 高 送 志 高 一 九 九 九 元 好 礼"),
        ("美的四大权益都可以享受", "我 再 问 一 下 不 用 问 美 的 四 大 权 益 都 可 以 享 受"),
    ]
    
    for target, asr_text in test_pairs:
        sim = matcher.text_similarity(target, asr_text)
        print(f"目标: {target}")
        print(f"ASR: {asr_text}")
        print(f"相似度: {sim:.3f}")
        print("")
    
    # 执行匹配
    print("执行文案匹配:")
    
    # 构建material_sentences格式
    material_sentences_dict = {}
    for i, mat in enumerate(materials):
        mat_sentences = []
        for seg in all_material_sentences:
            if seg['material_name'] == mat['name']:
                mat_sentences.append({
                    'text': seg['text'],
                    'start': seg['start'],
                    'end': seg['end']
                })
        if mat_sentences:
            material_sentences_dict[i] = mat_sentences
    
    print(f"有效素材数: {len(material_sentences_dict)}")
    
    # 使用阈值0.3进行匹配
    matched = matcher._greedy_sentence_matching(
        target_sentences,
        material_sentences_dict,
        threshold=0.3,
        single_threshold=0.5
    )
    
    # 计算匹配率
    matched_count = sum(1 for s in matched if not s.get('missing', False))
    match_rate = matched_count / len(target_sentences) if target_sentences else 0
    
    print(f"\n匹配结果:")
    print(f"  总文案句子: {len(target_sentences)}")
    print(f"  成功匹配: {matched_count}")
    print(f"  匹配率: {match_rate:.1%}")
    
    # 显示匹配详情
    print("\n匹配详情:")
    for i, m in enumerate(matched):
        status = "✓" if not m.get('missing') else "✗"
        print(f"  {status} [{i}] {m.get('text', '')[:40]}")
        if not m.get('missing'):
            print(f"       -> 素材{m.get('material_index')}, {m.get('start'):.1f}-{m.get('end'):.1f}, 相似度{m.get('similarity', 0):.2f}")

if __name__ == "__main__":
    main()
