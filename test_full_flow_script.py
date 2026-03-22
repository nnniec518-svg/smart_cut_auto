"""
完整流程自动化测试脚本
"""
import os
import sys
import json
import logging
import shutil
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 用户提供的文案
SCRIPT = """你确定
不用确定我们京东315活动3C、家电政府补贴至高15%，还能跟美的代金券叠加使用
我再问一下
不用问,美的四大权益都可以享受
权益一
全屋智能家电套购
送至高1499元豪礼
权益二
空气机及无风感系列空调一价全包
至高省1399元
权益三
购指定型号享最高
价值3200元局改服务
权益四
上抖音团购领
美的专属优惠券
活动时间：2026年3月1日-3月31日
美的火三月福利，今天带你们薅小天鹅洗烘套装的羊毛，错过等一年！
美的50代150 还可以叠加京东活动真的是太划算了
首选这套本色洗烘套装！蓝氧护色黑科技太懂女生了——白T恤越洗越亮，彩色衣服不串色，洗完烘完直接能穿，不用再担心晒不干有异味。现在领50代150的券，叠加京东315补贴，算下来比平时便宜大几百！
重点来了！京东315还有新房全屋全套5折抢，老房焕新补贴直接省到家。买小天鹅 先用券减100，再享政府补贴，双重优惠叠加，这便宜不占白不占！
我在京东电器旗舰店等你们！想买洗烘套装的赶紧来，记得领50代150的券，叠加补贴真的超划算～"""

def main():
    logger.info("="*60)
    logger.info("开始完整流程测试")
    logger.info("="*60)
    
    # Step 1: 清理旧缓存
    logger.info("Step 1: 清理旧缓存...")
    cache_dir = Path("storage/material_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("缓存已清理")
    
    # Step 2: 导入模块
    logger.info("Step 2: 导入模块...")
    from core.asr import ASR
    from core.matcher import Matcher
    from core.video_processor import VideoProcessor
    from core.audio_processor import AudioProcessor
    
    asr = ASR()
    audio_proc = AudioProcessor()
    matcher = Matcher()
    vp = VideoProcessor()
    logger.info("模块导入成功")
    
    # Step 3: 加载素材
    logger.info("Step 3: 加载素材...")
    materials_dir = Path("storage/materials-all")
    video_files = sorted(list(materials_dir.glob("*.MOV")))[:20]
    logger.info(f"找到 {len(video_files)} 个素材")
    
    materials = []
    for f in video_files:
        materials.append({
            "path": str(f),
            "name": f.name,
            "index": len(materials)
        })
    
    # Step 4: 分析素材 (使用matcher.process_single_material)
    logger.info("Step 4: 分析素材...")
    material_results = []
    
    for i, mat in enumerate(materials):
        logger.info(f"  处理素材 {i+1}/{len(materials)}: {mat['name']}")
        try:
            result = matcher.process_single_material(
                mat["path"],
                SCRIPT,
                asr,
                audio_proc,
                silence_threshold=1.5
            )
            material_results.append(result)
            
            if result:
                logger.info(f"    -> 相似度: {result.get('similarity', 0):.3f}")
            else:
                logger.warning(f"    -> 无匹配结果")
        except Exception as e:
            material_results.append(None)
            logger.error(f"    -> 错误: {e}")
    
    valid_count = sum(1 for r in material_results if r is not None)
    logger.info(f"有效素材数: {valid_count}/{len(materials)}")
    
    if valid_count == 0:
        logger.error("没有识别到任何素材，测试失败")
        return
    
    # Step 5: 文案匹配
    logger.info("Step 5: 文案匹配...")
    
    result = matcher.decide_best_materials(
        material_results,
        SCRIPT,
        single_threshold=0.85,
        sentence_threshold=0.55
    )
    
    segments = result.get("segments", [])
    logger.info(f"匹配模式: {result.get('mode')}")
    logger.info(f"匹配到 {len(segments)} 个片段")
    
    # 统计使用的素材
    used_materials = set()
    for seg in segments:
        if not seg.get("missing") and seg.get("material_index", -1) >= 0:
            used_materials.add(seg["material_index"])
    
    logger.info(f"使用的素材数: {len(used_materials)}")
    for idx in sorted(used_materials):
        logger.info(f"  - 素材 {idx}: {materials[idx]['name']}")
    
    if len(segments) == 0:
        logger.error("没有匹配到任何片段，测试失败")
        return
    
    # Step 6: 生成视频
    logger.info("Step 6: 生成视频...")
    os.makedirs("temp", exist_ok=True)
    
    cropped_files = []
    for i, seg in enumerate(segments):
        if seg.get("missing"):
            logger.warning(f"片段 {i} 缺失，跳过")
            continue
        
        mat_idx = seg.get("material_index", -1)
        if mat_idx < 0:
            continue
        
        input_path = materials[mat_idx]["path"]
        output_file = f"temp/cropped_{i}.mp4"
        
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        
        if end <= start:
            logger.warning(f"片段 {i} 时长无效")
            continue
        
        success = vp.crop_video(input_path, output_file, start, end)
        if success:
            cropped_files.append(output_file)
            logger.info(f"  裁剪片段 {i}: {output_file} ({start:.2f}s - {end:.2f}s)")
        else:
            logger.error(f"  裁剪片段 {i} 失败")
    
    if len(cropped_files) == 0:
        logger.error("没有可用的视频片段")
        return
    
    # 拼接
    output_path = "temp/final_output.mp4"
    if len(cropped_files) == 1:
        shutil.copy(cropped_files[0], output_path)
    else:
        vp.concat_videos(cropped_files, output_path)
    
    logger.info(f"视频已生成: {output_path}")
    
    # Step 7: 验证匹配率
    logger.info("Step 7: 验证匹配率...")
    try:
        asr_result = asr.transcribe(output_path)
        if asr_result and asr_result.get("text"):
            result_text = asr_result["text"]
            logger.info(f"视频ASR结果: {result_text[:200]}...")
            
            # 检查原文案中有多少句子出现在结果中
            matched_count = 0
            for sent in SCRIPT.split("\n"):
                sent = sent.strip()
                if sent and sent in result_text:
                    matched_count += 1
            
            total_sents = len([s for s in SCRIPT.split("\n") if s.strip()])
            match_rate = matched_count / total_sents * 100 if total_sents else 0
            logger.info(f"匹配率: {match_rate:.1f}% ({matched_count}/{total_sents})")
            
            if match_rate >= 80:
                logger.info("🎉 测试通过! 匹配率达到80%")
            else:
                logger.warning(f"⚠ 匹配率未达到80%，当前: {match_rate:.1f}%")
        else:
            logger.error("视频ASR失败")
    except Exception as e:
        logger.error(f"验证过程出错: {e}")
    
    logger.info("="*60)
    logger.info("测试完成")
    logger.info("="*60)

if __name__ == "__main__":
    main()
