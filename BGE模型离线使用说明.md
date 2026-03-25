# BGE 模型离线使用说明

## 📋 概述

本项目已配置 BAAI/bge-large-zh-v1.5 中文语义向量模型,所有模型文件已下载到本地,可以完全离线运行。

## 📁 模型文件结构

```
smart_cut_auto/
└── models/
    └── sentence_transformers/
        └── models--BAAI--bge-large-zh-v1.5/
            ├── snapshots/
            │   └── 79e7739b6ab944e86d6171e44d24c997fc1e0116/
            │       ├── config.json                    # 模型配置 (1 KB)
            │       ├── config_sentence_transformers.json # ST 配置 (0.1 KB)
            │       ├── modules.json                   # 模块配置 (0.3 KB)
            │       ├── pytorch_model.bin              # 模型权重 (1.27 GB)
            │       ├── sentence_bert_config.json     # SBERT 配置 (0.1 KB)
            │       ├── tokenizer.json                 # 分词器配置 (429 KB)
            │       ├── tokenizer_config.json          # 分词器配置 (0.4 KB)
            │       ├── vocab.txt                      # 词汇表 (107 KB)
            │       ├── 1_Pooling/
            │       │   └── config.json               # Pooling 配置 (0.2 KB)
            │       └── special_tokens_map.json        # 特殊 Token (0.1 KB)
            ├── refs/
            │   └── main                               # 主分支引用 (40 B)
            └── blobs/                                 # Blob 存储 (空,文件在 snapshots)
```

## 📊 模型信息

- **模型名称**: BAAI/bge-large-zh-v1.5
- **模型类型**: BERT-based 中文语义向量模型
- **嵌入维度**: 1024
- **总大小**: 约 2.43 GB
- **文件数量**: 19 个

## ✅ 完整性验证

所有必需文件已验证:

| 文件 | 大小 | 状态 |
|------|------|------|
| config.json | 1.0 KB | ✅ |
| pytorch_model.bin | 1.27 GB | ✅ |
| tokenizer.json | 429 KB | ✅ |
| vocab.txt | 107 KB | ✅ |
| 其他配置文件 | 1.2 KB | ✅ |

## 🚀 离线运行

### 方法1: 直接运行主程序

```powershell
cd C:\Users\nnniec\Program\smart_cut_auto
python main.py
```

程序会自动使用本地模型,无需网络连接。

### 方法2: 运行离线测试

```powershell
cd C:\Users\nnniec\Program\mart_cut_auto
test_offline.bat
```

或使用 Python:

```powershell
python offline_model_test.py
```

### 方法3: 完整验证

```powershell
python verify_bge_offline.py
```

## 🔧 环境配置

项目已自动配置以下环境变量:

```python
SENTENCE_TRANSFORMERS_HOME = models/sentence_transformers
HF_HUB_OFFLINE = 1  # 强制离线模式
```

配置位置:
- `core/matcher.py` - 匹配器使用本地模型
- `config.yaml` - 配置文件指定模型名称

## 📝 技术细节

### 模型加载逻辑

代码优先使用本地模型:

```python
# 1. 检查本地模型路径
local_model_path = MODELS_DIR / "sentence_transformers" / f"models--{model_name.replace('/', '--')}"

# 2. 如果本地存在,使用本地路径
if local_model_path.exists():
    model = SentenceTransformer(model_name)  # 会自动使用缓存

# 3. 否则使用远程下载(本项目不会触发)
else:
    model = SentenceTransformer(model_name, cache_folder=...)
```

### 配置文件

`config.yaml`:

```yaml
models:
  embedding_model: "BAAI/bge-large-zh-v1.5"
  fallback_embedding: "BAAI/bge-large-zh-v1.5"
```

## ⚠️ 注意事项

1. **不要删除 models/ 目录**
   - 这是唯一必需的模型文件存储位置
   - 删除后需要重新下载约 2.43 GB

2. **不要移动模型文件**
   - 保持 `models/sentence_transformers/` 结构不变
   - 程序通过环境变量定位此目录

3. **离线运行验证**
   - 首次离线运行前建议运行 `test_offline.bat` 验证
   - 如果测试失败,检查模型文件完整性

4. **网络影响**
   - 即使有网络,程序也会优先使用本地模型
   - 设置 `HF_HUB_OFFLINE=1` 可强制完全离线

## 🛠️ 故障排查

### 问题1: 模型加载失败

**症状**:
```
ValueError: Unrecognized model... Should have a `model_type` key in its config.json
```

**解决**:
1. 检查 `models/sentence_transformers/models--BAAI--bge-large-zh-v1.5/` 目录是否存在
2. 运行 `verify_bge_offline.py` 检查文件完整性
3. 确认 `config.json` 文件存在且包含 `model_type` 字段

### 问题2: 找不到模型

**症状**:
```
OSError: Can't load tokenizer for 'BAAI/bge-large-zh-v1.5'
```

**解决**:
1. 确认 `SENTENCE_TRANSFORMERS_HOME` 环境变量设置正确
2. 检查 `models/sentence_transformers/` 目录权限
3. 运行 `test_offline.bat` 验证

### 问题3: 仍尝试联网

**症状**:
看到 `Retrying... thrown while requesting HEAD https://huggingface.co/` 等日志

**解决**:
1. 设置环境变量: `set HF_HUB_OFFLINE=1`
2. 或在代码中添加:
   ```python
   os.environ["HF_HUB_OFFLINE"] = "1"
   ```

## 📦 模型备份

建议定期备份 `models/` 目录:

```powershell
# 创建压缩备份
Compress-Archive -Path models\sentence_transformers -DestinationPath models_backup.zip

# 解压恢复
Expand-Archive -Path models_backup.zip -DestinationPath models\
```

## 🔄 更新模型

如需更新模型到新版本:

1. 删除旧模型目录:
   ```powershell
   Remove-Item models\sentence_transformers\models--BAAI--bge-large-zh-v1.5 -Recurse -Force
   ```

2. 运行程序,自动下载新版本:
   ```powershell
   python main.py
   ```

3. 等待下载完成(需要网络连接)

## 📞 技术支持

如有问题,请查看:
- 日志文件: `logs/smart_cut.log`
- 错误追踪: 运行 `python offline_model_test.py` 查看详细错误

## ✨ 特性

- ✅ 完全离线运行
- ✅ 中文语义理解优化
- ✅ 高性能(1024 维嵌入)
- ✅ 自动本地缓存
- ✅ 无需网络配置
