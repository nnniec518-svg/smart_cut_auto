"""测试字符级重叠匹配"""
import re
from pathlib import Path
import json

# 字符级重叠匹配
def char_similarity(text1: str, text2: str) -> float:
    """基于字符重叠的相似度"""
    # 预处理：去除空格和非中文字符
    t1 = re.sub(r'[^\u4e00-\u9fff]', '', text1)
    t2 = re.sub(r'[^\u4e00-\u9fff]', '', text2)
    
    if not t1 or not t2:
        return 0.0
    
    print(f"  t1: {t1}")
    print(f"  t2: {t2}")
    
    # 方法1: 字符级Jaccard
    set1 = set(t1)
    set2 = set(t2)
    intersection = set1 & set2
    union = set1 | set2
    jaccard = len(intersection) / len(union) if union else 0
    
    # 方法2: 最长公共子串
    def lcs(s1, s2):
        m, n = len(s1), len(s2)
        if m == 0 or n == 0:
            return 0
        # 空间优化
        prev = [0] * (n + 1)
        max_len = 0
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    curr[j] = prev[j-1] + 1
                    max_len = max(max_len, curr[j])
            prev = curr
        return max_len
    
    lcs_len = lcs(t1, t2)
    lcs_ratio = lcs_len / max(len(t1), len(t2))
    
    print(f"  jaccard: {jaccard:.3f}, lcs_ratio: {lcs_ratio:.3f}")
    
    # 综合
    return jaccard * 0.3 + lcs_ratio * 0.7

# 测试
test_targets = [
    ("购指定型号享最高", "你走权益三购指定型号想至高三千两百元局改服局改服务"),
    ("价值3200元局改服务", "你走权益三购指定型号想至高三千两百元局改服局改服务"),
    ("上抖音团购领", "权益四胜权益四胜抖音团购领美的专属优惠券"),
]

for target, mat in test_targets:
    print(f"\n目标: {target}")
    print(f"素材: {mat}")
    sim = char_similarity(target, mat)
    print(f"  最终相似度: {sim:.3f}")
