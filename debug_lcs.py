"""测试LCS算法"""
import re

def lcs_debug(s1, s2):
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0, ""
    
    # 创建DP表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
    
    # 找到最长公共子串
    max_len = 0
    end_idx = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                end_idx = i
    
    lcs_str = s1[end_idx - max_len:end_idx] if max_len > 0 else ""
    return max_len, lcs_str

# 测试
s1 = "权益三"
s2 = "你走权益三购指定型号想至高"

lcs_len, lcs_str = lcs_debug(s1, s2)
print(f"s1: {s1}")
print(f"s2: {s2}")
print(f"LCS长度: {lcs_len}")
print(f"LCS字符串: {lcs_str}")
print(f"LCS比例: {lcs_len / max(len(s1), len(s2)):.3f}")
