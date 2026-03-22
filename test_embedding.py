import sys
sys.path.insert(0, '.')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制使用 CPU

print("=== 测试向量模型加载 ===")

try:
    print("1. 测试 transformers...")
    from transformers import AutoTokenizer, AutoModel
    print("   OK")
except Exception as e:
    print(f"   失败: {e}")

try:
    print("2. 测试 BGE-M3...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3")
    model.eval()
    print("   OK")
except Exception as e:
    print(f"   失败: {e}")

try:
    print("3. 测试 SentenceTransformer...")
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer('BAAI/bge-m3', device='cpu')
    print("   OK")
except Exception as e:
    print(f"   失败: {e}")

print("\n=== 测试完成 ===")
