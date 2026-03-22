# -*- coding: utf-8 -*-
"""简单测试脚本"""
import sys
import os

# 设置路径
os.chdir(r"c:\Users\nnniec\Program\smart_cut_auto")
sys.path.insert(0, r"c:\Users\nnniec\Program\smart_cut_auto")

# 测试1: 导入模块
print("=" * 50)
print("测试1: 导入模块")
print("=" * 50)

try:
    from db.models import Database, Asset, Segment
    print("[OK] db.models")
except Exception as e:
    print(f"[FAIL] db.models: {e}")

try:
    from core.processor import VideoPurifier
    print("[OK] core.processor")
except Exception as e:
    print(f"[FAIL] core.processor: {e}")

try:
    from core.planner import SequencePlanner, EmbeddingModel
    print("[OK] core.planner")
except Exception as e:
    print(f"[FAIL] core.planner: {e}")

try:
    from core.auto_cutter import VideoAutoCutter
    print("[OK] core.auto_cutter")
except Exception as e:
    print(f"[FAIL] core.auto_cutter: {e}")

# 测试2: 初始化数据库
print("\n" + "=" * 50)
print("测试2: 初始化数据库")
print("=" * 50)

try:
    db = Database("storage/test_new.db")
    print("[OK] Database created")
    
    # 添加测试素材
    import time
    asset = Asset(
        file_path="test_video.MOV",
        file_name="test_video",
        track_type="A_ROLL",
        valid_start_offset=1.5,
        duration=30.0,
        has_audio=True,
        audio_db=-20.0,
        mtime=time.time(),
        asr_text="这是测试文本",
        transcript_json='{"text": "这是测试文本", "segments": []}'
    )
    saved_asset = db.add_asset(asset)
    print(f"[OK] Asset saved: id={saved_asset.id}")
    
    # 查询
    assets = db.get_assets_by_type("A_ROLL")
    print(f"[OK] Query A_ROLL: {len(assets)} records")
    
    db.close()
    print("[OK] Database closed")
except Exception as e:
    print(f"[FAIL] Database: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("测试完成")
print("=" * 50)
