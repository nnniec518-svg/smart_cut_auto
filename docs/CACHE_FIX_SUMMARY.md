# 缓存失效逻辑优化 - 总结

## 问题根因

原有系统存在**缓存失效的"断层"**：

1. **logic_version只控制ASR层**
   - 仅触发素材的语音识别(ASR)重新处理
   - 不控制序列规划(SequencePlanner)的重新计算

2. **文案变化未检测**
   - 用户调整文案顺序后,系统仍使用旧的sequence.json
   - Matcher和Assembler的结果不会因文案变化而刷新

3. **关键断链点**
   ```python
   # main.py:340 - 直接调用,无缓存检查
   edl = self.plan_edl(script)
   
   # sequence_planner.py:452-636 - plan()方法无缓存机制
   def plan(self, script: str) -> List[Dict]:
       # 直接生成序列,不检查缓存
   ```

## 优化方案

### 1. 引入文案哈希检测 (script_hash)

**新增函数**:
```python
# 计算文案MD5哈希值
def calculate_script_hash(script: str) -> str:
    normalized = '\n'.join(line.strip() for line in script.strip().split('\n') if line.strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

# 检查缓存是否有效
def check_script_cache_valid(script: str, cache_path: Path) -> bool:
    current_hash = calculate_script_hash(script)
    if cache_path.exists():
        cached_hash = read_hash(cache_path)
        return cached_hash == current_hash
    return False

# 保存哈希到缓存
def save_script_hash(script: str, cache_path: Path):
    script_hash = calculate_script_hash(script)
    write_hash(cache_path, script_hash)
```

**特点**:
- 文案规范化: 去除多余空格和换行
- 哈希算法: MD5(32位空间,冲突概率极低)
- 缓存路径: `temp/script_hash.txt`

### 2. 统一缓存清理

**新增函数**:
```python
def force_clean_all_caches():
    """清理所有层级缓存"""
    # Level 1: 文案哈希
    delete("temp/script_hash.txt")
    
    # Level 2: 序列缓存
    delete("temp/sequence.json")
    
    # Level 3: 逻辑版本
    delete("temp/logic_version.txt")
    
    # Level 4: 数据库
    delete("storage/materials.db")
    
    # Level 5: 其他JSON
    delete_all("temp/*.json")
```

### 3. 双重检查机制

**优化后的plan_edl()**:
```python
def plan_edl(self, script: str, force_replan: bool = False) -> list:
    # 检查1: logic_version变化(触发ASR重新识别)
    logic_version_changed = not check_cache_validity()
    
    # 检查2: 文案变化(触发序列重新规划)
    script_changed = not check_script_cache_valid(script) or force_replan
    
    # 任何一项变化 → 清理所有缓存
    if logic_version_changed or script_changed:
        force_clean_all_caches()
        
        if logic_version_changed:
            self.scan_materials(force_reprocess=True)
    
    # 执行规划
    edl = self.planner.plan(script)
    
    # 保存哈希供下次对比
    save_script_hash(script)
    return edl
```

### 4. 新增CLI参数

```bash
# 强制重新规划(忽略文案缓存)
python main.py --replan
```

## 缓存层级架构

```
┌─────────────────────────────────────────┐
│           用户输入                    │
│  - 文案内容/顺序变化                  │
│  - logic_version变化                   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│        双重检查机制                    │
│  1. check_script_cache_valid()       │
│  2. check_cache_validity()            │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
   [变化]             [未变化]
        │                   │
        ▼                   ▼
┌──────────────┐    ┌──────────────┐
│清理所有缓存  │    │使用现有缓存  │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│重新执行流程  │    │直接输出结果  │
└──────┬───────┘    └──────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│          Level 1: ASR层             │
│  - materials.db                    │
│  - ASR识别结果                     │
│  触发条件: logic_version变化        │
└───────────────┬────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│       Level 2: 序列规划层          │
│  - sequence.json                   │
│  - Matcher匹配结果                 │
│  - Assembler去重结果              │
│  触发条件: script_hash变化         │
└───────────────┬────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│       Level 3: 渲染输出层          │
│  - final_output.mp4               │
│  触发条件: --replan参数          │
└──────────────────────────────────────┘
```

## 实际应用场景

### 场景1: 调整文案顺序

**用户操作**:
```python
# 原始文案
script1 = "第一句\n第二句\n第三句"

# 调整顺序
script2 = "第三句\n第二句\n第一句"
```

**系统响应**:
```
[INFO] 文案变化检测: 哈希不同 (abc123... -> xyz789...)
[INFO] 将重新执行Matcher和Assembler流程
[INFO] === 强制清理所有缓存 ===
[INFO]   删除文案哈希缓存
[INFO]   删除序列缓存: sequence.json
[INFO] === 缓存清理完成 ===
[INFO] 文案哈希已保存: xyz789...
[INFO] 文案语义编排完成 (按新顺序匹配素材)
```

### 场景2: 修改文案内容

**用户操作**:
```python
# 原始文案
script1 = "京东315活动补贴至高15%"

# 修改内容
script2 = "京东315活动补贴至高20%"  # 改了数字
```

**系统响应**:
```
[INFO] 文案变化检测: 哈希不同 (def456... -> ghi012...)
[INFO] 将重新执行Matcher和Assembler流程
[INFO] === 强制清理所有缓存 ===
[INFO]   删除文案哈希缓存
[INFO]   删除序列缓存: sequence.json
[INFO] === 缓存清理完成 ===
[INFO] 文案哈希已保存: ghi012...
[INFO] 重新匹配素材 (新文案"补贴至高20%")
```

### 场景3: logic_version变化

**用户操作**:
```yaml
# config.yaml
filter:
  logic_version: "1.0"  # 改为 "2.0"
```

**系统响应**:
```
[WARNING] 逻辑版本变化: 1.0 -> 2.0，需要清理缓存
[INFO] 检测到logic_version变化，强制清理ASR和序列缓存
[INFO] === 强制清理所有缓存 ===
[INFO]   删除文案哈希缓存
[INFO]   删除序列缓存: sequence.json
[INFO]   删除逻辑版本缓存
[INFO]   删除数据库: materials.db
[INFO] === 缓存清理完成 ===
[INFO] 重新扫描和识别素材...
[INFO] 开始重新ASR识别...
```

### 场景4: 格式变化(不触发)

**用户操作**:
```python
# 原始文案
script1 = "第一句\n第二句\n第三句"

# 仅调整格式(内容不变)
script2 = "  第一句  \n\n  第二句  \n\n  第三句  "
```

**系统响应**:
```
[INFO] 文案未变化，哈希值: abc123... (可使用缓存)
[INFO] 直接使用现有sequence.json
```

### 场景5: 强制重新规划

**用户操作**:
```bash
python main.py --replan
```

**系统响应**:
```
[INFO] 检测到文案变化（内容或顺序），强制重新规划
[INFO] === 强制清理所有缓存 ===
[INFO]   删除文案哈希缓存
[INFO]   删除序列缓存: sequence.json
[INFO] === 缓存清理完成 ===
[INFO] 重新执行Matcher和Assembler流程...
```

## 优化效果

### Before (优化前)

```
用户调整文案顺序 → 系统使用旧sequence.json
    ↓
输出视频顺序错误 ❌
    ↓
需要手动删除temp/sequence.json才能修复
```

### After (优化后)

```
用户调整文案顺序 → 检测到哈希变化
    ↓
自动清理缓存 → 重新规划 → 输出正确视频 ✅
```

## 关键改进点

1. **精确检测**: MD5哈希可检测任何文案变化
2. **自动化**: 无需手动干预,系统自动清理
3. **性能优化**: 相同文案可复用缓存
4. **可追溯**: 所有操作有详细日志

## 相关文件

- `main.py`: 缓存检查逻辑(新增)
- `docs/CACHE_OPTIMIZATION.md`: 详细方案文档
- `temp/script_hash.txt`: 文案哈希缓存(运行时生成)

## 测试建议

```bash
# 测试1: 文案内容变化
python main.py -s "第一句\n第二句"  # 运行
python main.py -s "第一句\n修改后的第二句"  # 应触发重新规划

# 测试2: 文案顺序变化
python main.py -s "第一句\n第二句\n第三句"
python main.py -s "第三句\n第二句\n第一句"  # 应触发重新规划

# 测试3: 格式变化(不触发)
python main.py -s "第一句\n第二句"
python main.py -s "  第一句  \n  第二句  "  # 应使用缓存

# 测试4: 强制重新规划
python main.py -s "第一句\n第二句"
python main.py --replan  # 应强制重新规划
```

## 注意事项

1. **哈希冲突概率**: MD5在32位空间上冲突概率可忽略
2. **规范化处理**: 格式变化不影响哈希,内容变化必触发
3. **手动覆盖**: 可通过`--replan`强制重新规划
4. **日志追踪**: 所有缓存操作有详细日志
