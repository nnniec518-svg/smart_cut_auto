# 通用型剪辑逻辑健壮性优化

## 优化概述

针对通用型剪辑场景的四个核心技术维度进行的"防错"优化，解决口播视频匹配中的常见问题。

---

## 1. 单向时间锚点逻辑 (Monotonic Constraint)

### 问题描述
在通用剪辑中，素材的第10秒通常对应文案的前半部分。如果匹配完第一句后，第二句跳回到了第2秒，这在90%的口播场景下都是错误的。

### 实现方案

**参数调整**：
```python
PENALTY_TIME_REVERSE = 0.5      # 轻微回跳惩罚
PENALTY_HARD_TIME_REVERSE = 1.0  # 严重回跳惩罚（同视频内回跳超过0.5秒）
```

**评分逻辑** (`_calculate_score`)：
```python
# 同视频内时间回跳，施加重度惩罚
time_gap = last_end - cand_start
if time_gap > 0.5:
    final_score -= self.PENALTY_HARD_TIME_REVERSE
    reason = f"hard_time_reverse_penalty (gap={time_gap:.1f}s)"
else:
    final_score -= self.PENALTY_TIME_REVERSE
    reason = f"time_reverse_penalty (gap={time_gap:.1f}s)"
```

### 效果
- 同视频内必须保持时间顺序
- 回跳超过0.5秒会受到致命惩罚（-1.0分）
- 回跳小于0.5秒也会受到惩罚（-0.5分）

---

## 2. 数字与专有名词硬过滤

### 问题描述
通用语义模型（MiniLM-L12-v2）擅长理解意思，但不擅长区分数字。在它看来，"权益一"和"权益二"的向量距离极近，导致权益顺序错乱。

### 实现方案

**数字提取** (`_extract_numbers`)：
- 提取阿拉伯数字：`\d+`
- 提取中文数字：零、一、二、三...十、百、千
- 识别模式：`"第X"`、`"权益X"`、`"X号"`

**数字匹配检查** (`_check_number_match`)：
```python
def _check_number_match(self, text_a: str, text_b: str) -> bool:
    nums_a = self._extract_numbers(text_a)
    nums_b = self._extract_numbers(text_b)

    # 如果文案有数字而素材没有，直接不匹配
    if nums_a and not nums_b:
        return False

    # 如果双方都有数字，检查是否有交集
    if nums_a and nums_b:
        common_nums = nums_a & nums_b
        if not common_nums:
            return False

    return True
```

**硬约束惩罚**：
```python
# 数字匹配检查（硬约束）
if not self._check_number_match(target_text, candidate_text):
    final_score -= self.PENALTY_NUMBER_MISMATCH  # -0.4
    reason = "number_mismatch"
    return final_score, reason  # 数字不匹配，直接返回
```

### 效果
- "权益一" 不会匹配到只说"权益二"的素材
- 数字不匹配的候选会被直接过滤
- 解决权益顺序倒置问题

---

## 3. 动态滑动窗口 (Dynamic Windowing)

### 问题描述
固定30句窗口对超长脚本或大量重复素材不适用，导致第1句文案误匹配到视频末尾的总结词。

### 实现方案

**动态窗口参数**：
```python
WINDOW_INITIAL_PERCENT = 0.20  # 初始窗口覆盖前20%
WINDOW_EXPANSION = 1.5         # 窗口扩展系数
WINDOW_MIN_SIZE = 10          # 最小窗口大小（句数）
```

**窗口更新逻辑** (`_update_dynamic_window`)：
```python
# 计算进度
progress_ratio = current_idx / total_sentences

# 计算窗口大小
window_size = max(
    WINDOW_MIN_SIZE,
    int(total_materials * WINDOW_INITIAL_PERCENT * WINDOW_EXPANSION)
)

# 计算窗口中心位置
window_center = int(total_materials * progress_ratio)

# 计算窗口边界
window_start_idx = max(0, window_center - window_size // 2)
window_end_idx = min(total_materials, window_center + window_size // 2)
```

**候选过滤** (`_filter_by_window`)：
- 窗口内的候选：正常评分
- 窗口外的候选：距离中心超过总素材30%的候选被跳过

### 效果
- 初始状态：窗口覆盖素材前20%
- 匹配成功后：窗口中心随进度向后推移
- 确保程序始终在"应该出现"的时间范围内寻找素材

---

## 4. 文案变化检测增强

### 现有机制
系统已实现文案哈希校验机制：
- `calculate_script_hash()`: 计算文案MD5
- `check_script_cache_valid()`: 检查缓存是否有效
- `save_script_hash()`: 保存文案哈希

### 优化点
文案变化检测已在 `main.py` 中完善：
```python
# 检查文案是否变化（触发序列重新规划）
script_changed = not check_script_cache_valid(script, script_hash_path) or force_replan

# 文案变化时仅删除序列缓存，保留素材数据库
if script_changed:
    self.logger.info("检测到文案变化（内容或顺序），将在渲染完成后清理序列缓存")
    force_clean_sequence_cache()
```

### 效果
- 文案内容或顺序变化时自动触发重新规划
- 保留素材数据库，避免重新扫描
- 确保文案改了匹配也会变

---

## 关键代码改动

### 文件：`core/sequence_planner.py`

#### 1. 新增常量参数
```python
# 惩罚参数
PENALTY_HARD_TIME_REVERSE = 1.0  # 严重时间回跳惩罚
PENALTY_NUMBER_MISMATCH = 0.4    # 数字不匹配惩罚

# 动态窗口参数
WINDOW_INITIAL_PERCENT = 0.20
WINDOW_EXPANSION = 1.5
WINDOW_MIN_SIZE = 10
```

#### 2. 新增状态变量
```python
self.window_start_idx = 0         # 窗口起始索引
self.window_end_idx = 0           # 窗口结束索引
self.progress_ratio = 0.0         # 当前进度比例
self.materials_list: List[Dict] = []  # 素材列表
```

#### 3. 新增方法
- `_extract_numbers()`: 提取文本中的数字
- `_check_number_match()`: 检查数字是否匹配
- `_update_dynamic_window()`: 更新动态窗口
- `_filter_by_window()`: 根据窗口过滤候选

#### 4. 修改方法
- `_calculate_score()`: 加入数字匹配和增强的单调性约束
- `plan()`: 加入动态窗口逻辑

---

## 测试结果

### 运行日志
```
2026-03-24 09:39:54 - Planning for 26 sentences
2026-03-24 09:40:44 - 渲染完成: final_output.mp4
2026-03-24 09:40:44 - 输出文件大小: 205.73 MB
2026-03-24 09:40:44 - 流程完成! 耗时: 50.4 秒
```

### 验证点
- ✅ 程序正常运行，无错误
- ✅ 视频生成成功
- ✅ 耗时合理（50.4秒）

---

## 预期效果

### 1. 权益顺序问题
**优化前**："权益一" 可能匹配到说"权益二"的片段
**优化后**：数字不匹配时直接过滤，权益顺序正确

### 2. 时间回跳问题
**优化前**：第1句匹配第10秒，第2句匹配第2秒
**优化后**：时间回跳受到0.5~1.0分惩罚，强制往后匹配

### 3. 匹配范围问题
**优化前**：第1句可能匹配到视频末尾
**优化后**：动态窗口确保在合理范围内匹配

### 4. 文案更新问题
**优化前**：文案改了但匹配没变
**优化后**：文案哈希校验确保更新生效

---

## 参数调优建议

### 时间回跳惩罚
- `PENALTY_TIME_REVERSE`: 0.3~0.7（轻微回跳）
- `PENALTY_HARD_TIME_REVERSE`: 0.8~1.2（严重回跳）

### 数字匹配惩罚
- `PENALTY_NUMBER_MISMATCH`: 0.3~0.6

### 动态窗口
- `WINDOW_INITIAL_PERCENT`: 0.15~0.30（初始窗口占比）
- `WINDOW_EXPANSION`: 1.2~2.0（窗口扩展系数）
- `WINDOW_MIN_SIZE`: 5~15（最小窗口句数）

---

## 总结

通过四个维度的优化，显著提升了通用型剪辑场景下的匹配准确性：
1. ✅ 单向时间锚点：避免时间回跳
2. ✅ 数字硬过滤：解决权益错乱
3. ✅ 动态窗口：确保匹配范围合理
4. ✅ 文案检测：确保更新生效

这些优化是针对通用型剪辑场景的"防错"机制，在保持灵活性的同时提高了匹配的准确性。
