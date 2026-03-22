import sys
sys.path.insert(0, '.')

print("测试模型加载...")

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

try:
    from sentence_transformers import SentenceTransformer
    
    print("加载 BGE-M3 模型...")
    # 不指定 device，让 SentenceTransformer 自动选择
    model = SentenceTransformer("BAAI/bge-m3")
    
    print(f"模型设备: {model.device}")
    print(f"模型加载成功!")
    
    # 测试编码
    test_texts = ["你好世界", "测试"]
    embeddings = model.encode(test_texts)
    print(f"编码成功, shape: {embeddings.shape}")
    
except Exception as e:
    import traceback
    print(f"错误: {e}")
    traceback.print_exc()
