# 手动测试更精确的重复检测
text = "你走今天带你们和今天带你们和小天鹅极红套装的羊毛"
text_no_space = text.replace(" ", "")

print(f"文本: {text_no_space}")
print()

# 方法：找到所有连续重复的模式
# 例如 "今天带你们和" 重复了

# 找第一个短语第二次出现的位置
# "今天带你们和" 第一次: 位置 2-8
# "今天带你们和" 第二次: 位置 8-14

# 用简单的子串查找
for length in range(4, 10):
    for i in range(len(text_no_space) - length * 2):
        substr = text_no_space[i:i+length]
        # 检查这个子串是否在后面重复
        next_pos = text_no_space.find(substr, i + length)
        if next_pos != -1:
            gap = next_pos - (i + length)
            if gap <= 5:  # 间隔小于5
                print(f"找到重复: '{substr}', 位置 {i} 和 {next_pos}, 间隔 {gap}")
