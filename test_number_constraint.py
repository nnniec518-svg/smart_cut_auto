"""
测试数字约束修复效果
验证"权益一"不再被"权益四"抢占
"""
import sys
import json
import logging
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.matcher import Matcher

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test")

# 测试文案
TEST_SCRIPT = """不用确定我们京东315活动3C、家电政府补贴至高15%，还能跟美的代金券叠加使用
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

def test_number_extraction():
    """测试数字提取功能"""
    print("=" * 60)
    print("测试1: 数字提取功能")
    print("=" * 60)

    test_cases = [
        ("权益一", {"1"}),
        ("权益二", {"2"}),
        ("权益三", {"3"}),
        ("权益四", {"4"}),
        ("权益五", {"5"}),
        ("第1个", {"1"}),
        ("第2号", {"2"}),
        ("至高1499元", {"1499"}),
        ("权益四胜权益四胜抖音团购领美的专属优惠券", {"4"}),
    ]

    for text, expected in test_cases:
        numbers = Matcher._extract_numbers(text)
        status = "[OK]" if numbers == expected else "[FAIL]"
        print(f"{status} '{text}' -> {numbers} (期望: {expected})")

def test_number_match():
    """测试数字匹配检查"""
    print("\n" + "=" * 60)
    print("测试2: 数字匹配检查")
    print("=" * 60)

    matcher = Matcher()

    test_cases = [
        # (文案, 素材, 期望结果)
        ("权益一", "权益一", True),
        ("权益一", "权益二", False),
        ("权益四", "权益四胜权益四胜抖音团购领美的专属优惠券", True),
        ("权益一", "权益四胜权益四胜抖音团购领美的专属优惠券", False),
        ("全屋智能家电套购", "全屋智能家电套购", True),
        ("至高1499元", "至高1499元", True),
        ("至高1499元", "至高1399元", False),
    ]

    for script, material, expected in test_cases:
        result = matcher._check_number_match(script, material)
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"{status} 文案: '{script}' vs 素材: '{material[:20]}...' -> {result} (期望: {expected})")

def test_matching_with_constraint():
    """测试带数字约束的匹配"""
    print("\n" + "=" * 60)
    print("测试3: 带数字约束的匹配")
    print("=" * 60)

    # 从数据库加载素材
    from db.models import Database
    db = Database("storage/materials.db")

    # 获取所有素材
    materials = db.get_assets_by_type("A_ROLL")
    print(f"从数据库加载了 {len(materials)} 个A_ROLL素材")

    # 模拟构造available_sentences
    available_sentences = []
    for mat in materials:
        try:
            # 解析transcript_json
            if mat.transcript_json:
                transcript = json.loads(mat.transcript_json)
                segments = transcript.get("segments", [])
                for seg in segments:
                    text = seg.get("text", "")
                    if text and len(text.strip()) > 0:
                        available_sentences.append({
                            "material_index": mat.id,
                            "text": text,
                            "start": seg.get("start", 0),
                            "end": seg.get("end", 0),
                            "video_name": mat.file_name
                        })
        except Exception as e:
            logger.warning(f"解析素材失败: {mat.file_name}, {e}")

    print(f"总共有 {len(available_sentences)} 个ASR句子")

    # 分句文案
    matcher = Matcher()
    target_sentences = matcher._split_sentences(TEST_SCRIPT)
    print(f"\n文案句子数: {len(target_sentences)}")
    print("文案句子:")
    for i, sent in enumerate(target_sentences):
        print(f"  [{i+1}] {sent}")

    # 执行匹配（启用数字约束）
    print("\n" + "-" * 60)
    print("执行匹配（启用数字约束）")
    print("-" * 60)
    matched = matcher._greedy_sentence_matching(
        target_sentences,
        available_sentences,
        threshold=0.5,
        enable_number_constraint=True,
        penalty_number_mismatch=0.4
    )

    # 分析匹配结果
    print("\n" + "=" * 60)
    print("匹配结果分析")
    print("=" * 60)

    # 统计权益句子的匹配情况
    equity_sentences = {}
    for i, sent in enumerate(target_sentences):
        if "权益" in sent and len(sent) <= 3:
            equity_num = sent[-1]  # "权益一" -> "一"
            equity_sentences[sent] = {
                "idx": i,
                "num": equity_num,
                "matched": matched[i] if i < len(matched) else None
            }

    print("\n权益句子匹配检查:")
    all_correct = True
    for sent, info in equity_sentences.items():
        match = info["matched"]
        if match and not match.get("missing", False):
            matched_text = match.get("text", "")
            matched_nums = Matcher._extract_numbers(matched_text)

            # 转换期望数字为阿拉伯数字
            num_map = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5'}
            expected_num = num_map.get(info["num"], info["num"])

            has_correct_num = expected_num in matched_nums
            status = "[OK]" if has_correct_num else "[FAIL]"
            print(f"{status} 文案: '{sent}' (期望数字: {expected_num}, 提取的数字: {matched_nums})")
            print(f"     匹配到: '{matched_text[:50]}...'")
            if not has_correct_num:
                all_correct = False
        else:
            print(f"[FAIL] 文案: '{sent}' -> 未找到匹配")
            all_correct = False

    # 总体统计
    total = len(target_sentences)
    matched_count = sum(1 for m in matched if m and not m.get("missing", False))
    match_rate = matched_count / total if total > 0 else 0

    print("\n" + "=" * 60)
    print(f"总体统计:")
    print(f"  总句子数: {total}")
    print(f"  匹配成功: {matched_count}")
    print(f"  匹配率: {match_rate:.2%}")
    print(f"  权益句子全部正确: {'[OK]' if all_correct else '[FAIL]'}")
    print("=" * 60)

    return all_correct

if __name__ == "__main__":
    print("\n")
    print("============================================================")
    print("              数字约束修复验证测试")
    print("============================================================")
    print()

    try:
        # 测试1: 数字提取
        test_number_extraction()

        # 测试2: 数字匹配
        test_number_match()

        # 测试3: 完整匹配
        success = test_matching_with_constraint()

        print("\n")
        if success:
            print("[OK][OK][OK] 所有测试通过！'权益一'不再被'权益四'抢占 [OK][OK][OK]")
        else:
            print("[FAIL][FAIL][FAIL] 测试失败，请检查日志 [FAIL][FAIL][FAIL]")
        print()

    except Exception as e:
        logger.error(f"测试执行失败: {e}", exc_info=True)
        print(f"\n[FAIL] 测试执行失败: {e}")
