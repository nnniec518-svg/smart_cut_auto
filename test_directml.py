import sys
sys.path.insert(0, '.')
import os

print("=" * 60)
print("测试 DirectML 和 AMF 硬件加速")
print("=" * 60)

# 测试 DirectML
print("\n[1] 测试 DirectML...")
try:
    from core.hardware import init_device, get_device_info
    
    device_info = get_device_info()
    print(f"  设备类型: {device_info['device_type']}")
    print(f"  设备名称: {device_info['device_name']}")
    print("  [OK] DirectML 测试通过")
except Exception as e:
    print(f"  [ERROR] {e}")

# 测试 SentenceTransformer with DirectML
print("\n[2] 测试 SentenceTransformer + DirectML...")
try:
    from core.planner import EmbeddingModel
    
    emb_model = EmbeddingModel("BAAI/bge-m3")
    test_texts = ["你好世界", "这是一个测试"]
    embeddings = emb_model.encode(test_texts)
    print(f"  向量维度: {embeddings.shape}")
    print("  [OK] SentenceTransformer 测试通过")
except Exception as e:
    print(f"  [ERROR] {e}")

# 测试 FFmpeg AMF 编码器
print("\n[3] 测试 FFmpeg AMF 编码器...")
try:
    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-encoders"],
        capture_output=True,
        text=True
    )
    if "h264_amf" in result.stdout:
        print("  [OK] h264_amf 编码器可用")
    else:
        print("  [WARNING] h264_amf 编码器不可用")
except Exception as e:
    print(f"  [ERROR] {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
