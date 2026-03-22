# -*- coding: utf-8 -*-
"""完整流程测试"""
import sys
import os
import logging

# 设置路径
os.chdir(r"c:\Users\nnniec\Program\smart_cut_auto")
sys.path.insert(0, r"c:\Users\nnniec\Program\smart_cut_auto")

output = []

def log(msg):
    output.append(str(msg))
    print(msg)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smart_cut")

log("=" * 60)
log("完整流程测试")
log("=" * 60)

# 测试文案
SCRIPT = """不用确定我们京东315活动3C、家电政府补贴至高15%
我再问一下
不用问,美的四大权益都可以享受
权益一
全屋智能家电套购
送至高1499元豪礼"""

try:
    # 导入模块
    log("\n[Step 1] 导入模块...")
    from core.auto_cutter import VideoAutoCutter
    log("[OK] VideoAutoCutter")
    
    # 创建控制器
    log("\n[Step 2] 创建控制器...")
    cutter = VideoAutoCutter(
        raw_folder="storage/materials-all",
        db_path="storage/materials.db",
        output_dir="temp"
    )
    log("[OK] VideoAutoCutter created")
    
    # 扫描素材（使用缓存，不强制处理）
    log("\n[Step 3] 扫描素材...")
    stats = cutter.scan_materials(force_reprocess=False)
    log(f"素材扫描完成: A_ROLL={stats['a_roll_count']}, B_ROLL={stats['b_roll_count']}")
    
    # 检查是否有素材
    if stats['a_roll_count'] == 0 and stats['b_roll_count'] == 0:
        log("[WARN] 没有素材，跳过匹配测试")
    else:
        # 执行规划
        log("\n[Step 4] 执行文案匹配...")
        edl = cutter.plan(SCRIPT)
        log(f"匹配完成: {len(edl)} 个片段")
        
        # 统计
        a_count = sum(1 for e in edl if e.get("track_type") == "A_ROLL")
        b_count = sum(1 for e in edl if e.get("track_type") == "B_ROLL")
        missing = sum(1 for e in edl if e.get("missing", False))
        log(f"匹配结果: A_ROLL={a_count}, B_ROLL={b_count}, 缺失={missing}")
        
        # 如果有匹配，执行渲染测试（只处理前3个片段）
        if a_count + b_count > 0:
            log("\n[Step 5] 渲染测试（仅测试前3个片段）...")
            test_edl = [e for e in edl if not e.get("missing")][:3]
            if test_edl:
                success = cutter.render(test_edl, "test_output.mp4", use_crossfade=False)
                if success:
                    log("[OK] 渲染测试成功")
                else:
                    log("[FAIL] 渲染测试失败")
            else:
                log("[WARN] 没有有效片段可渲染")
        else:
            log("[WARN] 没有匹配到素材，跳过渲染")
    
    cutter.close()
    log("\n[OK] 流程测试完成")
    
except Exception as e:
    log(f"\n[FAIL] 测试失败: {e}")
    import traceback
    log(traceback.format_exc())

# 写入文件
with open("test_full_output.log", "w", encoding="utf-8") as f:
    f.write("\n".join(output))
