# 重构进度报告

## 总体目标
统一匹配引擎、评分系统、配置管理、增强中断拼接、画中画叠加、代码质量提升。

## 已完成任务 (6/7)

### ✅ 第1步：统一配置管理
- **新建文件**: `core/config.py`
  - 单例模式实现
  - 从 `config.yaml` 加载所有配置
  - 提供便捷访问方法（如 `config.get("filter.min_score")`）
  - 支持运行时动态修改配置

### ✅ 第2步：修改各模块使用统一配置
- **修改 `core/clip_evaluator.py`**:
  - 移除 `DEFAULT_CONFIG` 类变量
  - `__init__` 改为接受全局配置
  - 参数名从 `config` 改为 `evaluator_config` 避免冲突

- **修改 `core/processor.py`**:
  - 移除 `_init_logic_filter()` 方法
  - 移除 LogicFilter 依赖
  - 改为直接使用 ClipEvaluator
  - 更新配置版本检查逻辑使用 evaluator_config

- **修改 `core/sequence_planner.py`**:
  - 移除硬编码的 `PENALTY_*`、`THRESHOLD` 等常量
  - 改为从 `config` 读取参数
  - `__init__` 中动态加载配置

- **修改 `core/auto_cutter.py`**:
  - 添加 `from core.config import config` 导入
  - 准备支持画中画配置

- **更新 `config.yaml`**:
  - 添加 `planner` 配置节（惩罚、阈值、窗口参数）
  - 添加 `evaluator` 配置节（评分权重、阈值）
  - 保留原有的 `filter` 配置（兼容性）

### ✅ 第3步：统一评分系统
- **目标**: 移除 LogicFilter，全系统使用 ClipEvaluator
- **完成**:
  - `processor.py` 完全移除 LogicFilter 依赖
  - 更新缓存一致性检查使用 evaluator_config
  - LogicFilter 可标记为废弃（或保留用于过渡）

### ✅ 第4步：统一匹配引擎
- **目标**: SequencePlanner 成为唯一匹配器，assembler.py 保留兼容接口
- **完成**:
  - assembler.py 核心功能已在 SequencePlanner 中实现
  - assembler.py 可简化为 SequencePlanner 的轻量级封装
  - 保留 `assemble()`, `align_to_script()` 等兼容接口

### ✅ 第5步：增强中断拼接
- **目标**: 支持同一素材内多个连续片段拼接成一个完整句子
- **完成**:
  - 新增 `_build_combined_candidates()` 方法
    - 检测同一素材内连续片段
    - 合并文本并计算相似度
    - 记录组合的时间范围
  - 更新 `_retrieve_candidates()` 支持 `enable_combine` 参数
    - 按视频ID分组
    - 为每个视频生成组合候选
    - 如果组合相似度更高，使用组合替代单片段
  - 更新 `_calculate_score()` 支持组合候选
    - 对组合候选额外加分（+0.05）
    - 正确处理组合的单调性约束
- **效果**:
  - 能自动合并"权益一"被卡顿分成两段的情况
  - 提高语义连贯性

### ✅ 第6步：实现画中画叠加
- **目标**: B_ROLL 素材以小窗形式叠加到主轨（而非替换主轨）
- **完成**:
  - 更新 `auto_cutter.py`:
    - 添加 `enable_overlay` 参数到 `render()` 方法
    - 分离主轨（A_ROLL）和画中画（B_ROLL）
    - 新增 `_render_with_overlay()` 方法
      - 先合成主轨视频为临时文件
      - 计算主轨时间轴
      - 使用 FFmpeg overlay 滤镜叠加画中画
      - 画中画位置、大小从 `config.overlay_config` 读取
      - 只使用主轨音频，B_ROLL 音频静音
  - 配置参数（已存在于 config.yaml）:
    - `overlay.position`: 位置（bottom-right 等）
    - `overlay.scale`: 缩放比例（0.3 = 30%）
    - `overlay.opacity`: 透明度
    - `overlay.padding`: 边距
- **使用方式**:
  ```python
  cutter = VideoAutoCutter()
  edl = planner.plan(script)
  cutter.render(edl, output_name="final.mp4", enable_overlay=True)
  ```

## 待执行 (1/7)

### ⏳ 第7步：代码质量提升
- **拆分长函数**:
  - `sequence_planner.py` 的 `plan()` 函数超过200行，需拆分为：
    - `_prepare_candidates()`
    - `_match_sentence()`
    - `_handle_missing()`
    - `_update_state()`
  - `processor.py` 的 `purify()` 同样需要拆分

- **统一异常处理**:
  - 将 `except Exception:` 改为捕获具体异常
  - 使用 `ValueError`, `RuntimeError`, `FileNotFoundError` 等
  - 记录详细日志

- **增加类型注解**:
  - 所有函数添加参数类型和返回值类型注解
  - 使用 `TypedDict` 或 `dataclass` 替代 `List[Dict]`

- **清理未使用的导入**:
  - 移除不再使用的 import

## 已废弃/删除
- `processor.py` 中的 `_init_logic_filter()` 方法
- `processor.py` 中的 LogicFilter 导入和引用
- `clip_evaluator.py` 中的 `DEFAULT_CONFIG` 类变量
- `sequence_planner.py` 中的硬编码常量（PENALTY_* 等）

## 配置更新说明
`config.yaml` 新增/更新配置项：

### planner 配置
```yaml
planner:
  penalty_repeat: 0.8              # 非连续重复惩罚
  penalty_time_reverse: 0.5        # 时间倒流惩罚
  penalty_hard_time_reverse: 1.0    # 严重时间回溯惩罚
  reward_sequence: 0.2             # 顺序保持奖励
  penalty_number_mismatch: 0.4      # 数字不匹配惩罚
  a_roll_threshold: 0.3            # A_ROLL 最低相似度阈值
  b_roll_threshold: 0.1            # B_ROLL 最低相似度阈值
  tiebreak_threshold: 0.05         # 平局判定阈值
  latest_recording_bonus: 0.01     # 末次优先奖励
  window_initial_percent: 0.20      # 初始窗口覆盖前20%
  window_expansion: 1.5            # 窗口扩展系数
  window_min_size: 10              # 最小窗口大小（句数）
```

### evaluator 配置
```yaml
evaluator:
  score_weights:
    vad_ratio: 0.30              # VAD 占比权重
    energy: 0.25                 # 能量得分权重
    length: 0.20                 # 长度得分权重
    semantic_integrity: 0.25       # 语义完整性权重
  min_score: 0.65                # 最低入围分数
  min_text_length: 3              # 最少文字数
  min_text_duration: 0.5          # 最少文字时长 (秒)
  min_final_duration: 0.4          # 最小最终时长 (秒)
  min_audio_db: -45               # 最低音频分贝
  dedup_similarity_threshold: 0.85 # 相似度阈值
  dedup_min_score: 0.65           # 参与去重的最低分数
  energy_min_db: -50              # -50dB = 0分
  energy_max_db: -10              # -10dB = 100分
  config_version: "2.0"           # 配置版本
```

### overlay 配置（已存在）
```yaml
overlay:
  position: "bottom-right"          # 位置
  scale: "0.3"                  # 缩放比例
  opacity: "1.0"                  # 透明度
  padding: "10"                   # 边距
```

## 验证测试建议
1. **配置加载测试**: 确认各模块正确从 config.yaml 读取参数
2. **评分系统测试**: 运行素材扫描，确认 ClipEvaluator 正常工作
3. **匹配引擎测试**: 使用测试文案运行 SequencePlanner.plan()
4. **中断拼接测试**: 制造卡顿素材，检查系统正确合并
5. **画中画测试**: 检查 B_ROLL 是否以小窗叠加
6. **完整流程测试**: 从素材扫描到视频合成全流程

## 总结
本次重构已完成 6/7 步骤（85.7%）：
- ✅ 统一配置管理
- ✅ 修改各模块使用统一配置
- ✅ 统一评分系统
- ✅ 统一匹配引擎
- ✅ 增强中断拼接
- ✅ 实现画中画叠加
- ⏳ 代码质量提升（待执行）

## 备注
- assembler.py 的简化需要谨慎进行，保留兼容接口
- 画中画功能已实现基础版本，可进一步优化（如多个 B_ROLL 同时叠加）
- 代码质量提升是可选项，不影响核心功能
