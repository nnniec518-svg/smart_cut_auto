# "末次优先"原则优化方案

## 问题描述

### 原有问题

文档提到算法阶段4为"**末次优先:同内容多版本时,选择最后录制的**"。

但在长文案匹配中,这存在冲突:
- **末次优先原则**: 希望选择最后录制的版本(通常在素材库末尾)
- **顺序约束机制**: 通过索引距离惩罚,强制选择靠近预期位置的片段

### 冲突示例

假设素材库中有3个版本的同一文案片段:

| 片段 | 索引位置 | 基础相似度 | 索引距离惩罚 | 最终分数 |
|------|----------|------------|--------------|----------|
| 片段A(旧) | 50 | 0.85 | 0.1 × (50-1) = 4.9 | **0.806** |
| 片段B(中) | 200 | 0.85 | 0.1 × (200-1) = 19.9 | **-1.39** |
| 片段C(新) | 500 | 0.85 | 0.1 × (500-1) = 49.9 | **-4.09** |

**问题**: 最后录制的片段反而因距离惩罚被淘汰!

## 优化方案

### 核心思路

将"末次优先"从**全局规则**降级为**平局决胜规则**:

- **Before**: 任何情况下都强制选择最后录制 → 与顺序约束冲突
- **After**: 只有当候选分数差距 < 0.05 时,才选择最后录制 → 平衡两者

### 实现细节

#### 1. 新增参数 (sequence_planner.py:142-144)

```python
# 平局决胜参数
TIEBREAK_THRESHOLD = 0.05      # 平局判定阈值
LATEST_RECORDING_BONUS = 0.01  # "末次优先"奖励分数
```

#### 2. 平局决胜函数 (_apply_latest_recording_tiebreaker)

```python
def _apply_latest_recording_tiebreaker(self,
                                       candidates: List[Dict],
                                       base_scores: List[float],
                                       final_scores: List[Tuple[float, str]],
                                       best_final_score: float,
                                       best_idx: int) -> Tuple[int, float]:
    """
    当多个候选的最终分数差距小于 TIEBREAK_THRESHOLD 时,
    选择最后录制的版本(索引最大的)
    """
    # 收集分数接近最佳分数的候选
    tie_candidates = []
    for i, (score, _) in enumerate(final_scores):
        if abs(score - best_final_score) < self.TIEBREAK_THRESHOLD:
            tie_candidates.append(i)

    if len(tie_candidates) <= 1:
        return best_idx, best_final_score  # 无平局

    # 有平局,选择索引最大(最后录制)的候选
    best_tie_idx = sorted(tie_candidates, reverse=True)[0]
    best_final_score_with_bonus = best_final_score + self.LATEST_RECORDING_BONUS

    return best_tie_idx, best_final_score_with_bonus
```

#### 3. A_ROLL匹配流程 (sequence_planner.py:590-627)

```python
# 计算所有候选的评分
all_candidates_scores = []
for cand in candidates:
    base_score = self.model.compute_similarity(sentence, cand.get("text", ""))
    final_score, reason = self._calculate_score(cand, base_score)
    all_candidates_scores.append((final_score, reason))
    # ... 更新最佳候选 ...

# 应用"末次优先"平局决胜规则
if len(all_candidates_scores) > 1:
    best_idx, best_final_score = self._apply_latest_recording_tiebreaker(
        candidates, all_base_scores, all_candidates_scores, ...
    )
    best_candidate = candidates[best_idx]
```

#### 4. B_ROLL匹配流程 (sequence_planner.py:476-520)

类似地,B_ROLL匹配也应用了平局决胜逻辑:

```python
# 检测是否有平局
tie_candidates = []
for i, score in enumerate(all_candidates_scores):
    if abs(score - best_score) < self.TIEBREAK_THRESHOLD:
        tie_candidates.append(i)

if len(tie_candidates) > 1:
    # 有平局,选择索引最大(最后录制)的
    best_idx = sorted(tie_candidates, reverse=True)[0]
    best_match = candidates[best_idx]
```

## 优化效果

### Before: 末次优先作为全局规则

```
场景: 文案"权益一全屋智能家电套购"
候选:
  - 索引50: 相似度0.90, 惩罚4.9, 最终0.806  ✓ 选中
  - 索引500: 相似度0.90, 惩罚49.9, 最终-4.09  ✗ 被淘汰
```

**问题**: 虽然相似度相同,但最后录制的版本因距离惩罚被淘汰

### After: 末次优先作为平局决胜规则

```
场景: 文案"权益一全屋智能家电套购"
候选:
  - 索引50: 相似度0.90, 惩罚4.9, 最终0.806  ✗ 平局淘汰
  - 索引500: 相似度0.90, 惩罚49.9, 最终-4.09  ✓ 平局决胜

执行逻辑:
  1. 索引50 最终分数 0.806 (原始)
  2. 索引500 最终分数 -4.09 (原始)
  3. 差距 |0.806 - (-4.09)| = 4.896 > 0.05 → 无平局

实际结果:
  由于索引距离惩罚太大,不会触发平局,仍选择索引50
```

**关键**: 平局决胜只在分数接近(差距<0.05)时才触发

### 典型平局场景

```
场景: 文案"不用确定我们京东315活动3C"
候选:
  - 索引100: 相似度0.95, 惩罚0, 最终0.95
  - 索引101: 相似度0.95, 惩罚0.1, 最终0.85  ✗ 差距0.10 > 0.05
  - 索引102: 相似度0.95, 惩罚0.2, 最终0.75  ✗ 差距0.20 > 0.05
  - 索引103: 相似度0.96, 惩罚0.3, 最终0.66  ✗ 相似度更高但惩罚太大

真正平局场景:
  - 索引200: 相似度0.92, 惩罚0, 最终0.92
  - 索引201: 相似度0.92, 惩罚0.1, 最终0.82  ✗ 差距0.10

真正触发的场景(罕见):
  - 索引50: 相似度0.85, 最终0.85
  - 索引51: 相似度0.85, 最终0.84  ✓ 差距0.01 < 0.05 → 平局决胜!
```

## 参数调优建议

### TIEBREAK_THRESHOLD (平局阈值)

| 值 | 行为 | 适用场景 |
|----|------|----------|
| 0.01 | 极严格平局 | 相似度完全相同时才触发 |
| 0.05 | 平衡(默认) | 大多数情况下平衡两者 |
| 0.10 | 宽松 | 更频繁触发末次优先 |

### LATEST_RECORDING_BONUS (末次奖励)

| 值 | 行为 |
|----|------|
| 0.01 | 象征性奖励(默认) |
| 0.05 | 轻微倾斜 |
| 0.10 | 明确优先 |

## 总结

### 优化前后对比

| 维度 | Before | After |
|------|--------|-------|
| "末次优先"地位 | 全局规则 | 平局决胜规则 |
| 与顺序约束关系 | 冲突 | 协同 |
| 索引距离惩罚影响 | 覆盖末次优先 | 先计算,再平局决胜 |
| 触发频率 | 总是触发 | 仅平局时触发 |
| 代码复杂度 | 简单 | 中等 |

### 关键收益

1. **解决冲突**: 末次优先不再与顺序约束矛盾
2. **保持顺序**: 画面仍按文案顺序推进
3. **智能降级**: 仅在分数接近时应用末次优先
4. **可调参数**: 通过TIEBREAK_THRESHOLD控制触发频率

## 相关文件

- `core/sequence_planner.py`: 主要实现文件
  - 第142-144行: 新增参数
  - 第383-436行: 平局决胜函数
  - 第590-627行: A_ROLL匹配流程(已优化)
  - 第476-520行: B_ROLL匹配流程(已优化)
