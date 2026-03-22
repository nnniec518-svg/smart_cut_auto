# 检查环境
import sys
sys.path.insert(0, '.')

try:
    import torch
    print(f"torch: {torch.__version__}")
except Exception as e:
    print(f"torch 导入失败: {e}")

try:
    import transformers
    print(f"transformers: {transformers.__version__}")
except Exception as e:
    print(f"transformers 导入失败: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print(f"sentence-transformers: 已安装")
except Exception as e:
    print(f"sentence-transformers 导入失败: {e}")

try:
    import autoawq
    print(f"autoawq: {autoawq.__version__}")
except Exception as e:
    print(f"autoawq 导入失败: {e}")
