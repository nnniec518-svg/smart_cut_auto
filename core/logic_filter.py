"""
逻辑过滤模块 - AudioAnalysis & LogicFilter
负责音频分析、置信度过滤、语义去重、A/B Roll打分分类
"""
import logging
import difflib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger("smart_cut")


class ClipQuality(Enum):
    """片段质量等级"""
    VALID = "valid"              # 有效口播
    INVALID_NOISE = "invalid_noise"    # 噪声/背景音
    INVALID_SHORT = "invalid_short"    # 太短
    DUPLICATE = "duplicate"      # 重复片段


@dataclass
class ClipAnalysis:
    """片段分析结果"""
    clip_id: str
    text: str
    confidence: float = 0.0      # 置信度 0-1
    no_speech_prob: float = 0.0  # 无语音概率 0-1
    audio_energy: float = 0.0    # 音频能量 dB
    duration: float = 0.0        # 时长 秒
    quality: ClipQuality = ClipQuality.VALID
    a_roll_score: float = 0.0   # A-Roll 打分
    is_duplicate: bool = False
    
    # 原始ASR数据
    word_segments: List[Dict] = field(default_factory=list)
    timestamps: List[Tuple[float, float]] = field(default_factory=list)


class LogicFilter:
    """
    逻辑过滤器 - 核心决策引擎
    实现：双重阈值过滤、语义去重、A/B Roll打分分类
    """
    
    # 默认阈值配置
    DEFAULT_CONFIG = {
        # 双重阈值过滤
        "min_text_length": 3,           # 最少文字数
        "min_confidence": 0.8,           # 最低置信度
        "max_no_speech_prob": 0.5,       # 最大无语音概率
        
        # 语义去重
        "dedup_similarity_threshold": 0.8,  # 相似度阈值
        
        # 物理安全边界
        "min_final_duration": 0.4,       # 最小最终时长(秒)
        
        # A/B Roll 打分权重
        "score_weights": {
            "text_length": 0.30,         # 文本长度权重
            "confidence": 0.40,           # 置信度权重
            "audio_energy": 0.30         # 音频能量权重
        },
        "a_roll_threshold": 70,         # A-Roll 入围分数
        
        # 缓存一致性
        "config_version": "1.0"          # 配置版本
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化逻辑过滤器
        
        Args:
            config: 配置字典，为None则使用默认配置
        """
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)
        
        # 加载依赖
        self._load_dependencies()
        
        logger.info(f"LogicFilter initialized with config: {self.config}")
    
    def _load_dependencies(self):
        """加载可选依赖"""
        self.levenshtein = None
        try:
            from Levenshtein import ratio as levenshtein_ratio
            self.levenshtein = levenshtein_ratio
        except ImportError:
            logger.warning("Levenshtein not installed, using difflib for similarity")
    
    # ========== 1. 双重阈值过滤 ==========
    
    def is_valid_speech(self, clip: ClipAnalysis) -> bool:
        """
        判断是否为有效语音（双重阈值）
        
        判断标准：
        1. 文字数 >= min_text_length
        2. 置信度 >= min_confidence
        3. no_speech_prob <= max_no_speech_prob
        
        Args:
            clip: 片段分析结果
            
        Returns:
            是否为有效语音
        """
        cfg = self.config
        
        # 检查文字长度
        text_len = len(clip.text.replace(" ", ""))
        if text_len < cfg["min_text_length"]:
            logger.debug(f"Clip {clip.clip_id}: text too short ({text_len} chars)")
            clip.quality = ClipQuality.INVALID_SHORT
            return False
        
        # 检查置信度
        if clip.confidence < cfg["min_confidence"]:
            logger.debug(f"Clip {clip.clip_id}: low confidence ({clip.confidence:.2f})")
            clip.quality = ClipQuality.INVALID_NOISE
            return False
        
        # 检查无语音概率
        if clip.no_speech_prob > cfg["max_no_speech_prob"]:
            logger.debug(f"Clip {clip.clip_id}: high no_speech_prob ({clip.no_speech_prob:.2f})")
            clip.quality = ClipQuality.INVALID_NOISE
            return False
        
        return True
    
    def analyze_clip(self, clip_data: Dict) -> ClipAnalysis:
        """
        分析单个片段
        
        Args:
            clip_data: 包含以下字段的字典:
                - clip_id: 片段ID
                - text: ASR识别文字
                - confidence: 置信度 (可选)
                - no_speech_prob: 无语音概率 (可选)
                - audio_energy: 音频能量dB (可选)
                - duration: 时长秒 (可选)
                - word_segments: 词级时间戳 (可选)
                
        Returns:
            ClipAnalysis 对象
        """
        clip = ClipAnalysis(
            clip_id=clip_data.get("clip_id", ""),
            text=clip_data.get("text", ""),
            confidence=clip_data.get("confidence", 0.9),  # 默认高置信度
            no_speech_prob=clip_data.get("no_speech_prob", 0.0),
            audio_energy=clip_data.get("audio_energy", -30.0),
            duration=clip_data.get("duration", 0.0),
            word_segments=clip_data.get("word_segments", [])
        )
        
        # 双重阈值判定
        clip.quality = ClipQuality.VALID if self.is_valid_speech(clip) else clip.quality
        
        # 计算 A-Roll 打分
        clip.a_roll_score = self._calculate_a_roll_score(clip)
        
        return clip
    
    # ========== 2. 语义模糊去重 ==========
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度 0-1
        """
        # 预处理：去除空格，统一大小写
        t1 = re.sub(r'\s+', '', text1).lower()
        t2 = re.sub(r'\s+', '', text2).lower()
        
        if self.levenshtein:
            # 使用 Levenshtein（更准确）
            return self.levenshtein(t1, t2)
        else:
            # 使用 difflib（无需额外安装）
            return difflib.SequenceMatcher(None, t1, t2).ratio()
    
    def deduplicate_clips(self, clips: List[ClipAnalysis]) -> List[ClipAnalysis]:
        """
        对片段进行语义去重
        
        逻辑：
        - 按时间排序
        - 比较相邻片段的文本相似度
        - 相似度 > 阈值时，保留最新的，标记其他的为重复
        
        Args:
            clips: 片段列表（已按时间排序）
            
        Returns:
            去重后的片段列表
        """
        if not clips or len(clips) <= 1:
            return clips
        
        threshold = self.config["dedup_similarity_threshold"]
        
        # 按时间排序（假设已有顺序）
        sorted_clips = sorted(clips, key=lambda x: x.duration)
        
        # 标记重复
        to_remove = set()
        
        for i in range(len(sorted_clips)):
            if i in to_remove:
                continue
                
            for j in range(i + 1, len(sorted_clips)):
                if j in to_remove:
                    continue
                    
                sim = self.calculate_similarity(
                    sorted_clips[i].text, 
                    sorted_clips[j].text
                )
                
                if sim > threshold:
                    # 相似度高，标记旧的为重复
                    to_remove.add(i)
                    sorted_clips[j].is_duplicate = True
                    logger.debug(f"Deduplication: clip {i} similar to {j} (sim={sim:.2f})")
                    break  # 只处理最近的重复
        
        # 过滤重复
        result = [c for idx, c in enumerate(sorted_clips) if idx not in to_remove]
        
        removed_count = len(clips) - len(result)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate clips")
        
        return result
    
    # ========== 3. 物理安全边界校验 ==========
    
    def validate_duration(self, clip: ClipAnalysis, 
                          offset_start: float = 0.0, 
                          offset_end: float = 0.0) -> bool:
        """
        校验片段应用偏移后的物理时长
        
        Args:
            clip: 片段分析结果
            offset_start: 起始偏移量（秒）
            offset_end: 结束偏移量（秒）
            
        Returns:
            时长是否满足要求
        """
        min_duration = self.config["min_final_duration"]
        
        final_duration = clip.duration - (offset_start + offset_end)
        
        if final_duration < min_duration:
            logger.warning(
                f"Clip {clip.clip_id}: duration too short "
                f"({final_duration:.2f}s < {min_duration}s), "
                f"original={clip.duration}s, offset={offset_start}+{offset_end}"
            )
            return False
        
        return True
    
    def filter_short_clips(self, clips: List[ClipAnalysis],
                           offset_start: float = 0.0,
                           offset_end: float = 0.0) -> List[ClipAnalysis]:
        """
        过滤时长不足的片段
        
        Args:
            clips: 片段列表
            offset_start: 起始偏移
            offset_end: 结束偏移
            
        Returns:
            过滤后的列表
        """
        valid = []
        for clip in clips:
            if self.validate_duration(clip, offset_start, offset_end):
                valid.append(clip)
            else:
                clip.quality = ClipQuality.INVALID_SHORT
        
        removed = len(clips) - len(valid)
        if removed > 0:
            logger.info(f"Removed {removed} clips due to insufficient duration")
        
        return valid
    
    # ========== 4. A/B Roll 打分分类器 ==========
    
    def _calculate_a_roll_score(self, clip: ClipAnalysis) -> float:
        """
        计算 A-Roll 打分
        
        打分维度：
        - 文本长度 (30%): 越长越可能是有效口播
        - 置信度 (40%): 越高越可能是有效口播
        - 音频能量 (30%): 越高越可能是有效口播
        
        Args:
            clip: 片段分析结果
            
        Returns:
            打分 0-100
        """
        weights = self.config["score_weights"]
        
        # 1. 文本长度得分 (0-100)
        # 假设有效口播至少5个字，20个字以上满分
        text_len = len(clip.text.replace(" ", ""))
        text_score = min(100, (text_len / 20) * 100) if text_len > 0 else 0
        
        # 2. 置信度得分 (0-100)
        confidence_score = clip.confidence * 100
        
        # 3. 音频能量得分 (0-100)
        # 假设 -45dB 为0分，-10dB 为100分（提高阈值以过滤低能量素材）
        energy = clip.audio_energy
        if energy >= -10:
            energy_score = 100
        elif energy <= -45:
            energy_score = 0
        else:
            energy_score = ((energy + 45) / 35) * 100
        
        # 加权求和
        score = (
            text_score * weights["text_length"] +
            confidence_score * weights["confidence"] +
            energy_score * weights["audio_energy"]
        )
        
        return round(score, 2)
    
    def classify_ab_roll(self, clips: List[ClipAnalysis]) -> Tuple[List[ClipAnalysis], List[ClipAnalysis]]:
        """
        A/B Roll 分类
        
        Args:
            clips: 片段列表
            
        Returns:
            (A_Roll列表, B_Roll列表)
        """
        threshold = self.config["a_roll_threshold"]
        
        a_roll = []
        b_roll = []
        
        for clip in clips:
            # 只有有效语音才参与分类
            if clip.quality == ClipQuality.VALID:
                if clip.a_roll_score >= threshold:
                    a_roll.append(clip)
                else:
                    b_roll.append(clip)
            else:
                # 无效片段直接归入 B-Roll
                b_roll.append(clip)
        
        logger.info(
            f"A/B Roll classification: A={len(a_roll)}, B={len(b_roll)}, "
            f"threshold={threshold}"
        )
        
        return a_roll, b_roll
    
    # ========== 5. 缓存一致性检查 ==========
    
    @staticmethod
    def get_config_version() -> str:
        """获取当前配置版本"""
        return LogicFilter.DEFAULT_CONFIG["config_version"]
    
    def check_cache_valid(self, cache_data: Dict) -> bool:
        """
        检查缓存是否有效
        
        Args:
            cache_data: 缓存数据字典
            
        Returns:
            缓存是否有效
        """
        if not cache_data:
            return False
        
        # 检查版本
        cached_version = cache_data.get("config_version")
        current_version = self.get_config_version()
        
        if cached_version != current_version:
            logger.warning(
                f"Cache version mismatch: cached={cached_version}, "
                f"current={current_version}"
            )
            return False
        
        # 检查关键配置参数
        key_params = [
            "min_text_length", "min_confidence", 
            "dedup_similarity_threshold", "min_final_duration"
        ]
        
        for param in key_params:
            cached_val = cache_data.get(param)
            current_val = self.config.get(param)
            
            if cached_val != current_val:
                logger.warning(
                    f"Config parameter changed: {param} "
                    f"(cached={cached_val}, current={current_val})"
                )
                return False
        
        return True


# ========== 辅助函数 ==========

def create_clip_from_asr(asr_result: Dict, clip_id: str, duration: float) -> Dict:
    """
    从 ASR 结果创建片段数据
    
    Args:
        asr_result: ASR 返回结果
        clip_id: 片段ID
        duration: 视频时长
        
    Returns:
        片段数据字典
    """
    text = asr_result.get("text", "")
    
    # 尝试提取置信度（FunASR可能返回）
    confidence = 0.9  # 默认
    if "segments" in asr_result and asr_result["segments"]:
        # 从段落中尝试获取
        for seg in asr_result["segments"]:
            if "confidence" in seg:
                confidence = seg["confidence"]
                break
    
    return {
        "clip_id": clip_id,
        "text": text,
        "confidence": confidence,
        "no_speech_prob": 0.0,
        "duration": duration,
        "word_segments": asr_result.get("segments", [])
    }
