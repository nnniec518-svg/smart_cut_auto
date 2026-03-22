"""
素材净化与分类模块 - VideoPurifier
核心功能：
1. 使用 ffprobe 检测音频流，无音轨或平均音量极低则标记为 B_ROLL
2. 调用 FunASR (带 timestamp=True) 获取词级时间戳
3. 正则切除逻辑：检查 ASR 前 3 秒，若匹配到 ["321", "走", "开始", "action"] 等词，记录最后一个提示词的 end_time 为 valid_start_offset
"""
from __future__ import annotations
import os
import re
import json
import time
import subprocess
import numpy as np
import yaml
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from sqlalchemy.orm import Session
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from db.models import Asset, Segment, Database
from core.logic_filter import LogicFilter, ClipAnalysis, ClipQuality
from core.clip_evaluator import ClipEvaluator, ClipData, ClipScore, ClipQuality as EvaluatorQuality

logger = logging.getLogger("smart_cut")

# 开拍提示词正则
CUE_PATTERN = re.compile(r"^(一|二|三|四|五|1|2|3|4|5|走|开始|准备|action|咔|好的|321|三二一)")

# 静音阈值 (dB) - 与 config.yaml 保持一致
SILENCE_THRESHOLD = -40


class VideoAsset:
    """视频素材实体类"""
    
    def __init__(self, path: str):
        self.path = path
        self.name = Path(path).stem
        self.track_type = "B_ROLL"  # 默认 B_ROLL
        self.valid_start_offset = 0.0
        self.segments = []  # 存储带时间戳的文本片段
        self.asr_text = ""  # 完整 ASR 文本
        self.duration = 0.0
        self.sample_rate = 16000
        self.has_audio = True
        self.audio_db = -100.0
        self.mtime = 0.0
        self.speech_ratio = 0.0  # 语音段时长比例
        # 新增：置信度和打分
        self.asr_confidence = 0.0
        self.a_roll_score = 0.0
        self.quality_status = "valid"
        
    def to_db_model(self) -> Asset:
        """转换为数据库模型"""
        # 获取配置版本
        from core.logic_filter import LogicFilter
        
        # 转换为 Python float (避免 numpy float32 JSON 序列化问题)
        confidence = float(self.asr_confidence) if hasattr(self.asr_confidence, '__float__') else self.asr_confidence
        a_roll_score = float(self.a_roll_score) if hasattr(self.a_roll_score, '__float__') else self.a_roll_score
        
        return Asset(
            file_path=self.path,
            file_name=self.name,
            track_type=self.track_type,
            valid_start_offset=self.valid_start_offset,
            duration=self.duration,
            has_audio=self.has_audio,
            audio_db=self.audio_db,
            mtime=self.mtime,
            asr_text=self.asr_text,
            transcript_json=json.dumps({
                "text": self.asr_text,
                "segments": self.segments,
                "confidence": confidence,
                "a_roll_score": a_roll_score,
                "quality_status": self.quality_status,
                # 缓存一致性检查字段
                "config_version": LogicFilter.DEFAULT_CONFIG["config_version"],
                "min_text_length": LogicFilter.DEFAULT_CONFIG["min_text_length"],
                "min_confidence": LogicFilter.DEFAULT_CONFIG["min_confidence"],
                "dedup_similarity_threshold": LogicFilter.DEFAULT_CONFIG["dedup_similarity_threshold"]
            }, ensure_ascii=False),
            asr_confidence=confidence,
            a_roll_score=a_roll_score,
            quality_status=self.quality_status
        )


class VideoPurifier:
    """
    素材净化器 - 负责素材分类、ASR处理、Offset计算
    """
    
    def __init__(self, db: Database):
        """
        初始化 VideoPurifier
        
        Args:
            db: Database 实例
        """
        self.db = db
        
        # 加载配置并初始化 LogicFilter
        self._init_logic_filter()
    
    def _init_logic_filter(self):
        """从配置文件加载参数并初始化 LogicFilter 和 ClipEvaluator"""
        try:
            config_path = Path(__file__).parent.parent / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # 提取 LogicFilter 需要的配置
                filter_config = {}
                
                # 从 filter 配置获取
                if 'filter' in config:
                    filter_config.update(config['filter'])
                
                # 从 vad 配置获取
                if 'vad' in config:
                    filter_config['min_confidence'] = config['vad'].get('speech_noise_thres', 0.8)
                
                # 从 classification 获取阈值
                if 'classification' in config:
                    filter_config['a_roll_threshold'] = config['classification'].get('a_roll_threshold', 70)
                
                self.logic_filter = LogicFilter(filter_config)
                logger.info(f"LogicFilter initialized with config version: {filter_config.get('config_version', 'unknown')}")
                
                # 初始化 ClipEvaluator (通用 A-Roll 评分系统)
                evaluator_config = {}
                if 'evaluator' in config:
                    evaluator_config = config['evaluator']
                
                self.evaluator = ClipEvaluator(evaluator_config)
                logger.info(f"ClipEvaluator initialized with config: {self.evaluator.get_config_hash()[:8]}...")
            else:
                self.logic_filter = LogicFilter()
                self.evaluator = ClipEvaluator()
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            self.logic_filter = LogicFilter()
            self.evaluator = ClipEvaluator()
        
    def _run_ffprobe(self, video_path: str) -> Dict:
        """
        使用 ffprobe 获取视频/音频信息
        
        Args:
            video_path: 视频路径
            
        Returns:
            包含音频信息的字典
        """
        result = {
            "has_audio": False,
            "duration": 0.0,
            "avg_db": -100.0,
            "sample_rate": 0,
            "codec": ""
        }
        
        try:
            # 获取视频时长
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.stdout.strip():
                result["duration"] = float(proc.stdout.strip())
            
            # 获取音频流信息
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type,codec_name,sample_rate",
                "-of", "json=c=1",
                video_path
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.stdout.strip():
                info = json.loads(proc.stdout.strip())
                audio_codecs = {"aac", "mp3", "opus", "vorbis", "ac3", "eac3", "pcm", "wav", "flac", "m4a"}
                
                for stream in info.get("streams", []):
                    codec_type = stream.get("codec_type", "")
                    codec_name = stream.get("codec_name", "").lower()
                    if codec_type == "audio" or codec_name in audio_codecs:
                        result["has_audio"] = True
                        result["sample_rate"] = int(stream.get("sample_rate", 16000))
                        result["codec"] = stream.get("codec_name", "")
                        break
            
            # 获取音频分贝 (EBU R128)
            if result["has_audio"]:
                try:
                    cmd = [
                        "ffmpeg", "-i", video_path,
                        "-af", "loudnorm=I=-24:TP=-1.5:LRA=11",
                        "-f", "null", "-"
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, stderr=subprocess.PIPE)
                    match = re.search(r"I:\s*([-\d.]+)\s*dB", proc.stderr)
                    if match:
                        result["avg_db"] = float(match.group(1))
                except Exception as e:
                    logger.debug(f"分贝检测失败: {e}")
                            
        except subprocess.TimeoutExpired:
            logger.warning(f"ffprobe 超时: {video_path}")
        except Exception as e:
            logger.warning(f"ffprobe 获取信息失败: {video_path}, {e}")
            
        return result
    
    def _detect_silence(self, video_path: str) -> Tuple[bool, float, float]:
        """
        检测是否为静音素材
        
        Args:
            video_path: 视频路径
            
        Returns:
            (是否静音, 平均分贝, 语音段时长比例)
        """
        info = self._run_ffprobe(video_path)
        
        if not info["has_audio"]:
            return True, -100.0
        
        # 使用 AudioProcessor 的 VAD 检测
        try:
            from core.audio_processor import AudioProcessor
            if not hasattr(self, '_audio_processor'):
                self._audio_processor = AudioProcessor()
            
            audio, sr = self._audio_processor.load_audio(video_path)
            
            # 计算平均分贝
            audio_float = audio.astype(np.float32)
            rms = np.sqrt(np.mean(audio_float**2))
            avg_db = 20 * np.log10(rms) if rms > 0 else -100.0
            
            # 检测语音段
            speech_segments = self._audio_processor.get_speech_segments(audio, sr)
            is_silence = len(speech_segments) == 0 or (len(audio) / sr) < 0.5
            
            # 计算语音段时长比例（用于置信度估算）
            total_duration = len(audio) / sr
            speech_duration = sum(end - start for start, end in speech_segments)
            speech_ratio = speech_duration / total_duration if total_duration > 0 else 0
            
            return is_silence, avg_db, speech_ratio
            
        except Exception as e:
            logger.warning(f"静音检测失败: {video_path}, {e}")
            return False, -50.0, 0.0
    
    def _load_asr_model(self):
        """加载 FunASR 模型"""
        from core.asr import ASR
        if not hasattr(self, '_asr_model'):
            # 尝试从配置文件加载 VAD 参数
            vad_params = None
            try:
                import yaml
                config_path = Path(__file__).parent.parent / "config.yaml"
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    if 'vad' in config:
                        vad_params = config['vad']
                        logger.info(f"从配置文件加载 VAD 参数: {vad_params}")
            except Exception as e:
                logger.warning(f"加载 VAD 配置失败，使用默认参数: {e}")
            
            self._asr_model = ASR(vad_params=vad_params)
        return self._asr_model
    
    def _find_cue_offset(self, segments: List[Dict], max_search_time: float = 3.0) -> float:
        """
        查找开拍提示词的偏移量
        
        Args:
            segments: ASR 结果段落列表
            max_search_time: 最大搜索时间 (秒)
            
        Returns:
            有效起始偏移量 (秒)
        """
        if not segments:
            return 0.0
        
        # 收集前 N 秒的所有词
        search_items = []
        
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").replace(" ", "")
            
            if start >= max_search_time:
                break
            
            # 词级时间戳
            timestamps = seg.get("timestamp", [])
            if timestamps:
                for i, ts in enumerate(timestamps):
                    if len(ts) >= 2:
                        word_start = ts[0] / 1000.0
                        word_end = ts[1] / 1000.0
                        if i < len(text):
                            search_items.append({
                                "text": text[i],
                                "start": word_start,
                                "end": word_end
                            })
            else:
                # 无时间戳，按字符均分
                for i, char in enumerate(text):
                    char_start = start + (end - start) * i / max(len(text), 1)
                    char_end = start + (end - start) * (i + 1) / max(len(text), 1)
                    search_items.append({
                        "text": char,
                        "start": char_start,
                        "end": char_end
                    })
        
        if not search_items:
            return 0.0
        
        # 检查前 5 个字/词
        check_limit = min(5, len(search_items))
        buffer_sec = 0.2
        
        valid_start_time = 0.0
        found_cue = False
        
        for i in range(check_limit):
            item = search_items[i]
            word = item["text"].strip()
            
            if CUE_PATTERN.match(word):
                valid_start_time = item["end"] + buffer_sec
                found_cue = True
                logger.info(f"找到提示词 '{word}' at {item['end']:.3f}s")
            else:
                break
        
        if found_cue:
            logger.info(f"检测到提示词，自动偏移起始点至: {valid_start_time:.3f}s")
        
        return valid_start_time
    
    def _asr_with_timestamp(self, video_path: str) -> Tuple[str, List[Dict], float]:
        """
        调用 FunASR 获取带时间戳的 ASR 结果
        
        Args:
            video_path: 视频路径
            
        Returns:
            (完整文本, 词级时间戳列表, 置信度)
        """
        try:
            asr_model = self._load_asr_model()
            
            with asr_model._model_lock:
                raw_result = asr_model.model.generate(
                    input=str(video_path),
                    batch_size_s=300,
                    timestamp=True
                )
            
            if not raw_result or len(raw_result) == 0:
                return "", [], 0.0
            
            item = raw_result[0]
            text = ""
            word_timestamps = []
            confidence = 0.0
            
            if isinstance(item, dict):
                text = item.get("text", "")
                word_timestamps = item.get("timestamp", [])
                # 尝试获取置信度
                segments_data = item.get("segments", [])
                if segments_data and len(segments_data) > 0:
                    scores = [seg.get("score", 0.0) for seg in segments_data if isinstance(seg, dict)]
                    if scores:
                        confidence = sum(scores) / len(scores)
            elif isinstance(item, list) and len(item) > 0:
                if isinstance(item[0], dict):
                    text = item[0].get("text", "")
                    word_timestamps = item[0].get("timestamp", [])
                    segments_data = item[0].get("segments", [])
                    if segments_data and len(segments_data) > 0:
                        scores = [seg.get("score", 0.0) for seg in segments_data if isinstance(seg, dict)]
                        if scores:
                            confidence = sum(scores) / len(scores)
                else:
                    text = str(item[0]) if item else ""
                    word_timestamps = []
            else:
                return "", [], 0.0
            
            # 如果没找到置信度，尝试从 tokens 中获取
            if confidence == 0.0 and isinstance(item, dict):
                tokens_data = item.get("tokens", [])
                if tokens_data:
                    scores = [t.get("score", 0.0) for t in tokens_data if isinstance(t, dict) and t.get("score", 0.0) > 0]
                    if scores:
                        confidence = sum(scores) / len(scores)
            
            # 转换为 segments 格式
            segments = []
            text_chars = list(text.replace(" ", ""))
            
            for i, ts in enumerate(word_timestamps):
                if len(ts) >= 2:
                    char = text_chars[i] if i < len(text_chars) else ""
                    segments.append({
                        "text": char,
                        "start": ts[0] / 1000.0,
                        "end": ts[1] / 1000.0,
                        "timestamp": [ts]
                    })
            
            logger.debug(f"ASR 置信度: {confidence:.4f}")
            return text, segments, confidence
            
        except Exception as e:
            logger.error(f"ASR 处理失败: {video_path}, {e}")
            return "", [], 0.0
    
    def purify(self, video_path: str, force_reprocess: bool = False) -> VideoAsset:
        """
        净化单个视频素材
        
        Args:
            video_path: 视频文件路径
            force_reprocess: 是否强制重新处理
            
        Returns:
            VideoAsset 对象
        """
        asset = VideoAsset(video_path)
        
        # 获取文件修改时间
        asset.mtime = os.path.getmtime(video_path)
        
        # 检查是否需要断点续传（强制重新处理时跳过缓存检查）
        if not force_reprocess and self.db.check_asset_fresh(video_path, asset.mtime):
            # 只在静默模式下记录缓存使用，不打印到主日志
            logger.debug(f"使用缓存: {asset.name}")
            db_asset = self.db.get_asset_by_path(video_path)
            if db_asset:
                asset.track_type = db_asset.track_type
                asset.valid_start_offset = db_asset.valid_start_offset
                asset.duration = db_asset.duration
                asset.has_audio = db_asset.has_audio
                asset.audio_db = db_asset.audio_db
                asset.asr_text = db_asset.asr_text
                transcript = json.loads(db_asset.transcript_json) if db_asset.transcript_json else {}
                asset.segments = transcript.get("segments", [])
                return asset
        
        # 1. 静音检测
        logger.info(f"检测静音: {asset.name}")
        is_silence, avg_db, speech_ratio = self._detect_silence(video_path)
        asset.has_audio = not is_silence
        asset.audio_db = avg_db
        
        if is_silence:
            asset.track_type = "B_ROLL"
            logger.info(f"素材 {asset.name} 判定为 B_ROLL (静音, db={avg_db:.1f})")
            self._save_to_db(asset)
            return asset
        
        # 2. 获取视频时长
        info = self._run_ffprobe(video_path)
        asset.duration = info.get("duration", 0.0)
        
        # 3. ASR 识别 (带时间戳)
        logger.info(f"ASR 识别: {asset.name}")
        asr_text, word_segments, asr_confidence = self._asr_with_timestamp(video_path)
        
        if asr_text:
            asset.asr_text = asr_text
            asset.segments = word_segments
            
            # ===== 使用 ClipEvaluator 通用评分系统 =====
            # 获取 VAD 语音段信息
            speech_segments = []
            try:
                if hasattr(self, '_audio_processor'):
                    audio, sr = self._audio_processor.load_audio(video_path)
                    speech_segments = self._audio_processor.get_speech_segments(audio, sr)
            except Exception as e:
                logger.warning(f"获取语音段失败: {e}")
            
            # 构建 ClipData
            clip_data = ClipData(
                clip_id=str(asset.name),
                file_path=video_path,
                text=asr_text,
                duration=asset.duration,
                audio_db=asset.audio_db,
                speech_segments=speech_segments,
                word_segments=word_segments,
                mtime=asset.mtime
            )
            
            # 计算评分 (使用 ClipEvaluator)
            # 先找 offset
            offset = self._find_cue_offset(word_segments)
            clip_score = self.evaluator.calculate_score(clip_data, offset_start=offset)
            
            # 物理安全检查
            if not self.evaluator.check_duration_safety(clip_score):
                clip_score.quality = EvaluatorQuality.INVALID_SHORT
            
            # 保存置信度和打分
            asset.asr_confidence = clip_score.total_score
            asset.a_roll_score = clip_score.total_score * 100  # 转换为百分制
            asset.quality_status = clip_score.quality.value
            
            # 根据评分判定 A/B Roll
            if clip_score.quality == EvaluatorQuality.VALID and clip_score.total_score >= self.evaluator.config["min_score"]:
                asset.track_type = "A_ROLL"
                logger.info(
                    f"素材 {asset.name} 判定为 A_ROLL "
                    f"(score={clip_score.total_score:.2f}, vad={clip_score.vad_ratio:.2f}, "
                    f"energy={clip_score.energy_score:.2f})"
                )
            else:
                asset.track_type = "B_ROLL"
                logger.info(
                    f"素材 {asset.name} 判定为 B_ROLL "
                    f"(score={clip_score.total_score:.2f}, quality={clip_score.quality.value})"
                )
            
            # 记录 offset (已在上面计算)
            if clip_score.quality == EvaluatorQuality.VALID:
                if offset > 0:
                    logger.info(f"素材 {asset.name} 有效起始偏移: {offset:.3f}s")
                else:
                    logger.info(f"素材 {asset.name} 无提示词，使用完整内容")
        else:
            # ASR 失败，降级为 B_ROLL
            logger.warning(f"ASR 失败，降级为 B_ROLL: {asset.name}")
            asset.track_type = "B_ROLL"
            asset.quality_status = "invalid_noise"
        
        # 5. 保存到数据库
        self._save_to_db(asset)
        
        return asset
    
    def _save_to_db(self, asset: VideoAsset):
        """保存到数据库"""
        db_asset = asset.to_db_model()
        saved_asset = self.db.add_asset(db_asset)
        
        # 如果是 A_ROLL，保存段落信息
        if asset.track_type == "A_ROLL" and asset.segments:
            session = self.db.get_session()
            try:
                # 删除旧的段落
                session.query(Segment).filter_by(video_id=saved_asset.id).delete()
                
                # 添加新段落
                for seg in asset.segments:
                    segment = Segment(
                        asset_id=saved_asset.id,
                        video_id=saved_asset.id,
                        start_time=seg.get("start", 0),
                        end_time=seg.get("end", 0),
                        valid_start_offset=asset.valid_start_offset,
                        asr_text=seg.get("text", ""),
                        timestamps_json=json.dumps(seg.get("timestamp", []))
                    )
                    session.add(segment)
                
                session.commit()
            finally:
                session.close()
    
    def purify_batch(self, video_dir: str, pattern: str = "*.MOV", force_reprocess: bool = False) -> List[VideoAsset]:
        """
        批量处理素材目录 - 多进程并行版本

        Args:
            video_dir: 视频目录
            pattern: 文件匹配模式
            force_reprocess: 是否强制重新处理

        Returns:
            处理后的素材列表
        """
        video_dir = Path(video_dir)
        video_files = sorted(video_dir.glob(pattern))

        logger.info(f"开始批量处理: {video_dir}, 共 {len(video_files)} 个文件，并发数: 8")

        # 多进程并行处理 - 使用 8 个 workers 利用 16 核 CPU
        assets = []
        num_workers = 8

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_path = {}
            for i, video_path in enumerate(video_files):
                future = executor.submit(
                    _process_single_video,
                    str(video_path),
                    force_reprocess,
                    i,
                    len(video_files)
                )
                future_to_path[future] = (i, video_path)

            # 收集结果
            for future in as_completed(future_to_path):
                i, video_path = future_to_path[future]
                try:
                    asset_data = future.result()
                    if asset_data:
                        # 从返回的数据创建 VideoAsset
                        asset = VideoAsset(asset_data['path'])
                        asset.track_type = asset_data['track_type']
                        asset.valid_start_offset = asset_data['valid_start_offset']
                        asset.duration = asset_data['duration']
                        asset.has_audio = asset_data['has_audio']
                        asset.audio_db = asset_data['audio_db']
                        asset.mtime = asset_data['mtime']
                        asset.asr_text = asset_data['asr_text']
                        asset.segments = asset_data['segments']
                        assets.append(asset)
                        logger.info(f"[{i+1}/{len(video_files)}] 完成: {video_path.name} -> {asset.track_type}")
                    else:
                        logger.warning(f"[{i+1}/{len(video_files)}] 跳过: {video_path.name}")
                except Exception as e:
                    logger.error(f"[{i+1}/{len(video_files)}] 处理失败: {video_path.name}, {e}")

        # 统计结果
        a_roll = sum(1 for a in assets if a.track_type == "A_ROLL")
        b_roll = sum(1 for a in assets if a.track_type == "B_ROLL")

        logger.info(f"处理完成: A_ROLL={a_roll}, B_ROLL={b_roll}")

        return assets
    
    def get_material_metadata(self, file_path: str) -> Optional[Dict]:
        """获取素材元数据"""
        db_asset = self.db.get_asset_by_path(file_path)
        if db_asset:
            return {
                "path": db_asset.file_path,
                "name": db_asset.file_name,
                "track_type": db_asset.track_type,
                "valid_start_offset": db_asset.valid_start_offset,
                "duration": db_asset.duration,
                "asr_text": db_asset.asr_text
            }
        return None


# ============ 多进程处理函数 ============
def _process_single_video(video_path: str, force_reprocess: bool, idx: int, total: int) -> Optional[Dict]:
    """
    多进程 worker 函数：处理单个视频素材

    注意：此函数在子进程中运行，不能访问主进程的数据库连接
    处理结果以字典形式返回，由主进程写入数据库
    """
    try:
        from core.audio_processor import AudioProcessor
        from core.asr import ASR

        logger.info(f"[{idx+1}/{total}] 处理: {Path(video_path).name}")

        # 加载 VAD 参数
        vad_params = None
        try:
            import yaml as yaml_module
            config_path = Path(__file__).parent.parent / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml_module.safe_load(f)
                if 'vad' in config:
                    vad_params = config['vad']
        except Exception:
            pass
        
        # 创建独立的处理器实例
        audio_processor = AudioProcessor()
        asr_model = ASR(vad_params=vad_params)

        # 获取文件信息
        mtime = os.path.getmtime(video_path)

        # 1. 静音检测
        is_silence, avg_db = _detect_silence_sync(video_path, audio_processor)

        # 2. 获取视频信息
        info = _get_video_info_sync(video_path)
        duration = info.get("duration", 0.0)

        # 3. ASR 识别
        asr_text, word_segments = _asr_with_timestamp_sync(video_path, asr_model)

        if asr_text:
            track_type = "A_ROLL"
            # 查找提示词偏移
            valid_start_offset = _find_cue_offset_sync(word_segments)
        elif is_silence:
            track_type = "B_ROLL"
            valid_start_offset = 0.0
        else:
            track_type = "B_ROLL"
            valid_start_offset = 0.0

        return {
            "path": video_path,
            "track_type": track_type,
            "valid_start_offset": valid_start_offset,
            "duration": duration,
            "has_audio": True,
            "audio_db": avg_db,
            "mtime": mtime,
            "asr_text": asr_text,
            "segments": word_segments
        }

    except Exception as e:
        logger.error(f"处理失败 {video_path}: {e}")
        return None


def _detect_silence_sync(video_path: str, audio_processor: AudioProcessor) -> Tuple[bool, float]:
    """同步版本的静音检测"""
    try:
        import tempfile
        import soundfile as sf

        # 提取音频
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-ac', '1', '-ar', '16000',
            '-f', 'wav', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            return True, -100.0

        # 读取音频
        import io
        audio, sr = sf.read(io.BytesIO(result.stdout), dtype='float32')

        if len(audio) == 0:
            return True, -100.0

        # 计算音量
        audio_float = audio.astype(np.float32)
        rms = np.sqrt(np.mean(audio_float**2))
        avg_db = 20 * np.log10(rms) if rms > 0 else -100.0

        # 检测语音段
        speech_segments = audio_processor.get_speech_segments(audio, sr)
        is_silence = len(speech_segments) == 0 or (len(audio) / sr) < 0.5

        return is_silence, avg_db

    except Exception as e:
        logger.warning(f"静音检测失败: {video_path}, {e}")
        return True, -100.0


def _get_video_info_sync(video_path: str) -> Dict:
    """同步获取视频信息"""
    try:
        import json
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return {}

        info = json.loads(result.stdout)
        duration = 0.0
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                duration = float(info.get('format', {}).get('duration', 0))
                break

        return {"duration": duration}
    except Exception:
        return {}


def _asr_with_timestamp_sync(video_path: str, asr_model: ASR) -> Tuple[str, List[Dict]]:
    """同步版本的 ASR 识别"""
    try:
        result = asr_model.transcribe(video_path, word_timestamps=True)
        text = result.get("text", "")

        # 转换 segments 格式
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "text": seg.get("text", ""),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0)
            })

        return text, segments

    except Exception as e:
        logger.error(f"ASR 处理失败: {video_path}, {e}")
        return "", []


def _find_cue_offset_sync(segments: List[Dict], max_search_time: float = 3.0) -> float:
    """同步版本的提示词查找"""
    if not segments:
        return 0.0

    search_items = []
    for seg in segments:
        start = seg.get("start", 0)
        if start >= max_search_time:
            break
        text = seg.get("text", "").replace(" ", "")
        if text:
            search_items.append((start, seg.get("end", 0), text))

    # 倒序查找最后一个匹配项
    for start, end, text in reversed(search_items):
        if CUE_PATTERN.search(text):
            return end

    return 0.0


# ============ 断点续传检查 ============
def should_reprocess(video_path: str, db: Database, force_reprocess: bool = False) -> bool:
    """
    检查素材是否需要重新处理
    
    Args:
        video_path: 视频文件路径
        db: Database 实例
        force_reprocess: 是否强制重新处理
        
    Returns:
        True 表示需要重新处理
    """
    if force_reprocess:
        logger.info(f"Force reprocess: {video_path}")
        return True
    
    # 1. 检查文件修改时间
    current_mtime = os.path.getmtime(video_path)
    if not db.check_asset_fresh(video_path, current_mtime):
        return True
    
    # 2. 增强缓存一致性检查：检查配置版本
    try:
        from core.logic_filter import LogicFilter
        session = db.get_session()
        asset = session.query(Asset).filter_by(file_path=video_path).first()
        
        if asset:
            # 检查配置版本
            transcript = asset.transcript_json
            if transcript:
                import json
                data = json.loads(transcript)
                cached_config_version = data.get("config_version", "0")
                current_config_version = LogicFilter.get_config_version()
                
                if cached_config_version != current_config_version:
                    logger.info(
                        f"Config version mismatch for {video_path}: "
                        f"cached={cached_config_version}, current={current_config_version}"
                    )
                    return True
                    
                # 检查关键参数是否变化
                key_params = ['min_text_length', 'min_confidence', 'dedup_similarity_threshold']
                for param in key_params:
                    cached_val = data.get(param)
                    current_val = LogicFilter.DEFAULT_CONFIG.get(param)
                    if cached_val != current_val:
                        logger.info(f"Config parameter changed: {param}")
                        return True
                        
    except Exception as e:
        logger.warning(f"Cache consistency check failed: {e}")
    
    return False


if __name__ == "__main__":
    # 测试代码
    import logging
    logging.basicConfig(level=logging.INFO)
    
    db = Database("storage/materials.db")
    purifier = VideoPurifier(db)
    
    # 批量处理素材
    assets = purifier.purify_batch("storage/materials-all")
    
    # 打印统计
    a_roll = [a for a in assets if a.track_type == "A_ROLL"]
    b_roll = [a for a in assets if a.track_type == "B_ROLL"]
    
    print(f"\n=== 处理结果 ===")
    print(f"总计: {len(assets)}")
    print(f"A_ROLL: {len(a_roll)}")
    print(f"B_ROLL: {len(b_roll)}")
    
    print(f"\n=== A_ROLL 素材 Offset ===")
    for a in a_roll[:10]:
        print(f"{a.name}: offset={a.valid_start_offset:.3f}s")
