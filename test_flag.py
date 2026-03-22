import sys
sys.path.insert(0, '.')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制使用 CPU
os.environ['ROCM_VISIBLE_DEVICES'] = ''  # 禁用 ROCm

print("=== 测试 FlagEmbedding ===")

try:
    print("1. 测试 FlagEmbedding...")
    from FlagEmbedding import FlagModel
    print("   FlagModel 导入成功")
except Exception as e:
    print(f"   FlagModel 导入失败: {e}")
    import traceback
    traceback.print_exc()

try:
    print("2. 加载模型...")
    model = FlagModel("BAAI/bge-m3", use_fp16=False, device="cpu")
    print("   模型加载成功")
except Exception as e:
    print(f"   模型加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试完成 ===")
