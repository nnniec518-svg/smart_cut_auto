# "末次优先"优化实施总结

## 完成时间
2026-03-23

## 问题描述

**核心冲突**:
- **末次优先原则**: 希望选择最后录制的版本(素材库末尾)
- **顺序约束机制**: 通过索引距离惩罚,强制选择靠近预期位置的片段

**实际影响**:
在长文案匹配中,最后录制的片段因索引距离惩罚过大,反而被淘汰,违背了"末次优先"的设计初衷。

## 解决方案

### 核心思路

将"末次优先"从**全局强制规则**降级为**智能平局决胜规则**:

**触发条件**: 只有当多个候选的`final_score`差距 < 0.05 时,才选择索引最大(最后录制)的版本。

### 实现细节

#### 1. 新增参数 (`sequence_planner.py:142-144`)

```python
TIEBREAK_THRESHOLD = 0.05      # 平局判定阈值(0.05)
LATEST_RECORDING_BONUS = 0.01  # "末次优先"象征性奖励(0.01)
```

#### 2. 平局决胜函数 (`sequence_planner.py:383-436`)

新增`_apply_latest_recording_tiebreaker()`方法:

```python
def _apply_latest_recording_tiebreaker(self, candidates, base_scores, final_scores, ...):
    # 1. 收集所有分数接近最佳分数的候选
    tie_candidates = [i for i, (score, _) in enumerate(final_scores)
                    if abs(score - best_final_score) < TIEBREAK_THRESHOLD]

    # 2. 如果无平局,直接返回原结果
    if len(tie_candidates) <= 1:
        return best_idx, best_final_score

    # 3. 有平局,选择索引最大(最后录制)的候选
    best_tie_idx = sorted(tie_candidates, reverse=True)[0]
    return best_tie_idx, best_final_score + LATEST_RECORDING_BONUS
```

#### 3. A_ROLL匹配流程优化 (`sequence_planner.py:590-627`)

```python
# 计算所有候选评分(用于平局决胜)
all_candidates_scores = []
for cand in candidates:
    base_score = self.model.compute_similarity(sentence, cand.get("text", ""))
    final_score, reason = self._calculate_score(cand, base_score)
    all_candidates_scores.append((final_score, reason))

# 应用"末次优先"平局决胜
if len(all_candidates_scores) > 1:
    best_idx, best_final_score = self._apply_latest_recording_tiebreaker(
        candidates, all_base_scores, all_candidates_scores, ...
    )
    best_candidate = candidates[best_idx]
```

#### 4. B_ROLL匹配流程优化 (`sequence_planner.py:476-520`)

B_ROLL匹配也应用了相同的平局决胜逻辑,确保一致性。

## 优化效果

### Before: 问题场景

```
文案: "权益一全屋智能家电套购"

候选片段:
  - 片段A(旧): 索引50, 相似度0.90, 惩罚4.9, 最终0.806  ✓ 被选中
  - 片段C(新): 索引500, 相似度0.90, 惩罚49.9, 最终-4.09  ✗ 被淘汰

问题: 虽然相似度相同,最后录制的版本因距离惩罚被淘汰
```

### After: 优化效果

```
场景1: 索引距离惩罚大 → 不触发平局
  片段A: 最终0.806
  片段C: 最终-4.09
  差距: 4.896 > 0.05 → 无平局 → 选择片段A(保持顺序约束)

场景2: 分数真正接近 → 触发平局决胜
  片段X: 索引100, 最终0.85
  片段Y: 索引101, 最终0.84
  差距: 0.01 < 0.05 → 平局 → 选择片段Y(末次优先)
```

## 关键收益

### 1. 解决冲突
- ✅ 末次优先不再与顺序约束矛盾
- ✅ 画面仍按文案顺序推进

### 2. 智能降级
- ✅ 仅在分数接近时触发末次优先
- ✅ 保持原有评分逻辑的优先级

### 3. 可调参数
- ✅ `TIEBREAK_THRESHOLD=0.05`: 控制平局触发频率
- ✅ `LATEST_RECORDING_BONUS=0.01`: 象征性奖励分数

### 4. 代码质量
- ✅ 新增独立的平局决胜函数
- ✅ A_ROLL和B_ROLL一致应用
- ✅ 详细的调试日志

## 代码改动汇总

| 文件 | 改动行数 | 说明 |
|------|----------|------|
| `core/sequence_planner.py` | ~60行 | 新增参数、函数、优化匹配流程 |
| `docs/LATEST_RECORDING_OPTIMIZATION.md` | 新文件 | 详细方案文档 |
| `docs/LATEST_RECORDING_SUMMARY.md` | 本文件 | 实施总结 |

## 测试建议

### 单元测试场景

```python
def test_latest_recording_tiebreaker():
    # 场景1: 无平局 → 不触发
    candidates = [...]
    best_idx, best_score = planner._apply_latest_recording_tiebreaker(...)
    assert best_idx == 0  # 不改变原选择

    # 场景2: 有平局 → 选择索引最大
    candidates = [...]
    best_idx, best_score = planner._apply_latest_recording_tiebreaker(...)
    assert best_idx == 2  # 选择最后录制的
```

### 集成测试

```python
def test_full_matching_with_latest_recording():
    script = "测试文案多个版本"
    edl = planner.plan(script)
    # 验证在分数接近时选择最后录制版本
```

## 未来优化方向

### 1. 自适应阈值

根据素材数量动态调整`TIEBREAK_THRESHOLD`:
```python
adaptive_threshold = 0.05 * (1 + log10(len(candidates)))
```

### 2. 上下文感知

考虑前序选择的版本,保持一致性:
```python
if last_selected_is_latest:
    latest_bonus = 0.01
else:
    latest_bonus = 0.05  # 倾向于切换到最新版本
```

### 3. 用户偏好

添加配置选项,允许用户调整末次优先权重:
```yaml
# config.yaml
matching:
  latest_recording_priority: "tiebreak"  # global | tiebreak | disabled
```

## 总结

通过将"末次优先"降级为平局决胜规则,成功解决了与顺序约束的冲突,实现了:
- ✅ 保持画面顺序
- ✅ 在分数接近时优先选择最新版本
- ✅ 可配置的触发阈值
- ✅ 代码清晰易维护

优化已完成,系统现在能够在保证顺序约束的前提下,智能地应用"末次优先"原则!
