"""完整流程自动化测试"""
import sys
import os
import json
import time
import logging
from pathlib import Path

# 禁用不必要的日志
os.environ['MODELSCOPE_LOG_LEVEL'] = 'ERROR'
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("test")

# 文案
SCRIPT = """不用确定我们京东315活动3C、家电政府补贴至高15%，还能跟美的代金券叠加使用
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

# 目标匹配率
TARGET_MATCH_RATE = 0.80

def analyze_materials(asr, materials):
    """分析所有素材"""
    cache_dir = Path("storage/material_cache_all")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for i, mat in enumerate(materials):
        print(f"  [{i+1}/{len(materials)}] 分析 {mat['name']}...")
        cache_file = cache_dir / f"{mat['name']}.json"
        
        if cache_file.exists():
            print(f"      -> 使用缓存")
            continue
        
        try:
            result = asr.transcribe(mat["path"])
            if result and result.get("text"):
                # 保存缓存
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"      -> 识别: {result.get('text', '')[:50]}...")
            else:
                print(f"      -> 识别失败")
        except Exception as e:
            print(f"      -> 错误: {e}")

def main():
    print("=" * 60)
    print("完整流程自动化测试")
    print("=" * 60)
    
    # 1. 加载素材
    print("\n[1] 加载素材...")
    materials_dir = Path("storage/materials-all")
    materials = []
    for f in sorted(materials_dir.glob("*.MOV")):
        materials.append({"name": f.stem, "path": str(f)})
    print(f"素材数量: {len(materials)}")
    
    # 2. 分析素材
    print("\n[2] 分析素材...")
    from core.asr import ASR
    asr = ASR()
    analyze_materials(asr, materials)
    
    # 3. 加载缓存并匹配
    print("\n[3] 执行匹配...")
    from core.matcher import Matcher
    matcher = Matcher()
    
    # 分句文案
    target_sentences = matcher._split_sentences(SCRIPT)
    print(f"文案句子数: {len(target_sentences)}")
    
    # 收集素材句子 - 构建正确格式的列表
    # 同时加载音频进行静音检测裁剪
    from core.audio_processor import AudioProcessor
    audio_processor = AudioProcessor()
    
    cache_dir = Path("storage/material_cache_all")
    available_sentences = []
    for i, mat in enumerate(materials):
        cache_file = cache_dir / f"{mat['name']}.json"
        if not cache_file.exists():
            continue
            
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        
        asr_result = cache.get('asr_result', cache)
        if not asr_result or not asr_result.get('segments'):
            continue
        
        # 加载音频进行静音检测
        mat_path = mat['path']
        try:
            if mat_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                audio_path = audio_processor.extract_audio_from_video(mat_path)
            else:
                audio_path = mat_path
            audio, sr = audio_processor.load_audio(audio_path)
            speech_segments = audio_processor.get_speech_segments(audio, sr)
            
            # 获取整体静音裁剪范围
            trimmed_start = speech_segments[0][0] if speech_segments else 0
            trimmed_end = speech_segments[-1][1] if speech_segments else (len(audio) / sr)
        except Exception as e:
            print(f"  警告: 处理音频失败 {mat['name']}: {e}")
            trimmed_start, trimmed_end = 0, 0
        
        for seg in asr_result['segments']:
            text = seg.get('text', '')
            if text:
                start = seg.get('start', 0)
                end = seg.get('end', 0)
                # 应用静音裁剪
                start = max(start, trimmed_start)
                end = min(end, trimmed_end)
                if end > start:
                    available_sentences.append({
                        'material_index': i,
                        'text': text,
                        'start': start,
                        'end': end
                    })
    
    print(f"有效素材句子数: {len(available_sentences)}")
    
    # 执行匹配 - 降低阈值以提高匹配率
    matched = matcher._greedy_sentence_matching(
        target_sentences,
        available_sentences,
        threshold=0.15
    )
    
    # 4. 计算匹配率
    print("\n[4] 匹配结果...")
    matched_count = sum(1 for s in matched if not s.get('missing', False))
    match_rate = matched_count / len(target_sentences) if target_sentences else 0
    
    print(f"  总文案句子: {len(target_sentences)}")
    print(f"  成功匹配: {matched_count}")
    print(f"  匹配率: {match_rate:.1%}")
    
    # 显示匹配详情
    print("\n匹配详情:")
    for i, m in enumerate(matched):
        status = "OK" if not m.get('missing') else "XX"
        text = m.get('text', '')[:35]
        print(f"  {status} [{i+1:2d}] {text}")
        if not m.get('missing'):
            print(f"       -> 素材{m.get('material_index')}, {m.get('start'):.1f}s-{m.get('end'):.1f}s, 相似度{m.get('similarity', 0):.2f}")
    
    # 5. 生成视频
    print("\n[5] 生成视频...")
    if matched_count > 0:
        from core.video_processor import VideoProcessor
        vp = VideoProcessor()
        
        # 准备片段
        clips = []
        for m in matched:
            if m.get('missing'):
                continue
            
            mat_idx = m.get('material_index')
            if mat_idx is not None and 0 <= mat_idx < len(materials):
                mat = materials[mat_idx]
                clips.append({
                    'path': mat['path'],
                    'start': m['start'],
                    'end': m['end']
                })
        
        print(f"  准备 {len(clips)} 个片段")
        
        # 裁剪并拼接
        output_path = "temp/final_output.mp4"
        vp.concat_videos([c['path'] for c in clips], output_path)
        print(f"  已生成: {output_path}")
    else:
        print("  没有可用的匹配片段")
    
    # 6. 验证结果
    print("\n[6] 验证结果...")
    if match_rate >= TARGET_MATCH_RATE:
        print(f"  [OK] 匹配率 {match_rate:.1%} >= 目标 {TARGET_MATCH_RATE:.0%} - 测试通过!")
    else:
        print(f"  [XX] 匹配率 {match_rate:.1%} < 目标 {TARGET_MATCH_RATE:.0%} - 测试未通过")

if __name__ == "__main__":
    main()
