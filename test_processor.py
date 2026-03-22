"""测试Processor - 完整测试"""
import os
os.remove("storage/materials_new.db") if os.path.exists("storage/materials_new.db") else None

from core.processor import Processor, CUE_WORDS
from pathlib import Path

print("CUE_WORDS:", CUE_WORDS)

processor = Processor("storage/materials_new.db")

# 测试单个素材 - 有"走"的
files = list(Path("storage/materials-all").glob("*.MOV"))
test_files = [f for f in files if "8802" in f.name or "8804" in f.name][:2]

for f in test_files:
    asset = processor.process_single(str(f), force_reprocess=False)
    print(f"\n=== {asset.name} ===")
    print(f"track_type: {asset.track_type}")
    print(f"asr_text: {asset.asr_text}")
    print(f"segments: {len(asset.segments)}")
    print(f"offset: {asset.valid_start_offset:.3f}s")
    if asset.segments:
        print(f"前3个segments:")
        for seg in asset.segments[:3]:
            print(f"  {seg['text']}: {seg['start']:.3f}s - {seg['end']:.3f}s")
