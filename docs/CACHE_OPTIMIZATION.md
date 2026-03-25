# 缓存失效逻辑优化方案

## 问题描述

原有系统存在缓存失效的"断层"问题：

1. **logic_version仅控制ASR重新识别**
   - `logic_version`只触发素材的ASR重新识别
   - 调整`logic_version`会清理数据库，强制重新识别所有素材

2. **文案变化未检测**
   - `SequencePlanner`的匹配结果不会因为文案内容或顺序变化而刷新
   - 用户调整文案顺序后，系统可能使用旧的`sequence.json`
   - 即使文案相同但顺序不同，也不会触发重新计算

3. **关键断链点**
   - `main.py`第340行:`plan_edl(script)`直接调用`planner.plan(script)`
   - `sequence_planner.py`第452-636行:`plan()`方法没有任何缓存机制
   - 缓存机制只存在于素材处理层，不存在于序列规划层

## 优化方案

### 1. 引入文案哈希检测

在`main.py`中添加了三个新函数：

```python
def calculate_script_hash(script: str) -> str:
    """计算文案的MD5哈希值"""

def check_script_cache_valid(script: str, cache_path: Path) -> bool:
    """检查文案缓存是否有效"""

def save_script_hash(script: str, cache_path: Path):
    """保存文案哈希到缓存"""
```

**工作原理**：
- 使用MD5哈希值检测文案内容或顺序变化
- 文案规范化：去除多余空格和换行，确保相同内容不同格式产生相同哈希
- 缓存路径：`temp/script_hash.txt`

### 2. 统一缓存清理机制

新增`force_clean_all_caches()`函数：

```python
def force_clean_all_caches():
    """
    强制清理所有缓存（文案哈希 + logic_version + 数据库）
    """
```

**清理内容**：
1. 文案哈希缓存：`temp/script_hash.txt`
2. 序列缓存：`temp/sequence.json`
3. 逻辑版本缓存：`temp/logic_version.txt`
4. 数据库：`storage/materials.db`
5. 其他JSON缓存文件

### 3. 双重检查机制

在`plan_edl()`方法中实现双重检查：

```python
def plan_edl(self, script: str, force_replan: bool = False) -> list:
    # 检查logic_version是否变化（触发ASR重新识别）
    logic_version_changed = not check_cache_validity()

    # 检查文案是否变化（触发序列重新规划）
    script_changed = not check_script_cache_valid(script, script_hash_path) or force_replan

    # 如果任何一项变化，强制清理所有缓存
    if logic_version_changed or script_changed:
        force_clean_all_caches()

        # 重新扫描素材（logic_version变化时）
        if logic_version_changed:
            self.scan_materials(force_reprocess=True)

    # 执行规划
    edl = self.planner.plan(script)

    # 保存当前文案的哈希（供下次对比）
    save_script_hash(script, script_hash_path)
```

### 4. 新增命令行参数

```bash
python main.py --replan  # 强制重新规划（忽略文案缓存）
```

## 缓存层级结构

### Level 1: 素材处理层 (logic_version)
- 控制内容：ASR识别结果
- 触发条件：`config.yaml`中的`filter.logic_version`变化
- 影响范围：
  - 数据库清理（重新识别所有素材）
  - 强制重走素材扫描流程

### Level 2: 序列规划层 (script_hash)
- 控制内容：Matcher和Assembler的匹配结果
- 触发条件：文案内容或顺序变化
- 影响范围：
  - 序列缓存清理（`sequence.json`）
  - 强制重走`SequencePlanner.plan()`流程

### Level 3: 渲染输出层
- 控制内容：最终生成的视频
- 触发条件：手动指定`--replan`参数
- 影响范围：强制重新渲染

## 工作流程

### 场景1: 文案内容变化
```
用户修改文案 → calculate_script_hash()检测到哈希变化
    ↓
check_script_cache_valid()返回False
    ↓
force_clean_all_caches()清理所有缓存
    ↓
planner.plan()重新生成序列
    ↓
save_script_hash()保存新哈希
```

### 场景2: 文案顺序变化
```
用户调整句子顺序 → calculate_script_hash()检测到哈希变化
    ↓
check_script_cache_valid()返回False
    ↓
force_clean_all_caches()清理所有缓存
    ↓
planner.plan()按新顺序匹配素材
    ↓
save_script_hash()保存新哈希
```

### 场景3: logic_version变化
```
用户修改config.yaml中的logic_version → check_cache_validity()返回False
    ↓
force_clean_all_caches()清理所有缓存
    ↓
scan_materials(force_reprocess=True)重新识别所有素材
    ↓
planner.plan()重新生成序列
    ↓
save_script_hash()保存新哈希
```

## 优势

1. **精确性**：文案哈希能够精确检测任何内容或顺序变化
2. **自动化**：无需手动干预，系统自动检测并清理缓存
3. **性能**：只有真正变化时才重新计算，相同文案可复用缓存
4. **可追溯**：缓存文件保存历史记录，便于调试

## 注意事项

1. **哈希冲突概率极低**：MD5哈希在32位空间上，冲突概率可忽略
2. **规范化处理**：去除空格和换行，确保格式变化不影响哈希
3. **手动覆盖**：可通过`--replan`参数强制重新规划
4. **日志记录**：所有缓存操作都有详细日志，便于追踪

## 测试用例

### 测试1: 内容变化
```python
script1 = "这是第一句话\n这是第二句话"
script2 = "这是第一句话\n这是修改后的第二句话"
# 预期：哈希不同，触发重新规划
```

### 测试2: 顺序变化
```python
script1 = "第一句\n第二句\n第三句"
script2 = "第三句\n第二句\n第一句"
# 预期：哈希不同，触发重新规划
```

### 测试3: 格式变化（不触发）
```python
script1 = "第一句\n第二句\n第三句"
script2 = "  第一句  \n\n  第二句  \n\n  第三句  "
# 预期：哈希相同，使用缓存
```

### 测试4: 完全相同
```python
script1 = "第一句\n第二句\n第三句"
script2 = "第一句\n第二句\n第三句"
# 预期：哈希相同，使用缓存
```

## 相关文件

- `main.py`: 缓存检查逻辑
- `config.yaml`: logic_version配置
- `temp/script_hash.txt`: 文案哈希缓存
- `temp/logic_version.txt`: 逻辑版本缓存
- `storage/materials.db`: 素材数据库
- `temp/sequence.json`: 序列缓存
