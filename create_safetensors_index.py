import json
from pathlib import Path

# 读取safetensors文件获取大小
safetensors_path = Path("models/sentence_transformers/models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/pytorch_model.safetensors")
total_size = safetensors_path.stat().st_size

# 读取原始的权重结构（从其他snapshot备份）
from safetensors import safe_open
keys = []
with safe_open(safetensors_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())

# 创建索引文件
index = {
    "metadata": {
        "total_size": total_size
    },
    "weight_map": {key: "pytorch_model.safetensors" for key in keys}
}

# 写入索引文件
output_path = Path("models/sentence_transformers/models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116/model.safetensors.index.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2, ensure_ascii=False)

print(f"Created index file: {output_path}")
print(f"Total keys: {len(keys)}")
print(f"Total size: {total_size}")
