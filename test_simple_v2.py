# -*- coding: utf-8 -*-
"""简单测试脚本 - 带文件输出"""
import sys
import os

# 设置路径
os.chdir(r"c:\Users\nnniec\Program\smart_cut_auto")
sys.path.insert(0, r"c:\Users\nnniec\Program\smart_cut_auto")

output = []

def log(msg):
    output.append(str(msg))

# 测试1: 导入模块
log("=" * 50)
log("测试1: 导入模块")
log("=" * 50)

try:
    from db.models import Database, Asset, Segment
    log("[OK] db.models")
except Exception as e:
    log(f"[FAIL] db.models: {e}")

try:
    from core.processor import VideoPurifier
    log("[OK] core.processor")
except Exception as e:
    log(f"[FAIL] core.processor: {e}")

try:
    from core.planner import SequencePlanner, EmbeddingModel
    log("[OK] core.planner")
except Exception as e:
    log(f"[FAIL] core.planner: {e}")

try:
    from core.auto_cutter import VideoAutoCutter
    log("[OK] core.auto_cutter")
except Exception as e:
    log(f"[FAIL] core.auto_cutter: {e}")

# 测试2: 初始化数据库
log("")
log("=" * 50)
log("测试2: 初始化数据库")
log("=" * 50)

try:
    db = Database("storage/test_new.db")
    log("[OK] Database created")
    
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
    log(f"[OK] Asset saved: id={saved_asset.id}")
    
    # 查询
    assets = db.get_assets_by_type("A_ROLL")
    log(f"[OK] Query A_ROLL: {len(assets)} records")
    
    db.close()
    log("[OK] Database closed")
except Exception as e:
    log(f"[FAIL] Database: {e}")
    import traceback
    log(traceback.format_exc())

log("")
log("=" * 50)
log("测试完成")
log("=" * 50)

# 写入文件
with open("test_output.log", "w", encoding="utf-8") as f:
    f.write("\n".join(output))
