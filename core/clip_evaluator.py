"""
通用 A-Roll 评分系统 - ClipEvaluator
基于物理特征（能量、时长）和语义特征（VAD占比）自动判定 A-Roll 合法性
不依赖特定 ASR 模型
"""
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import difflib

logger = logging.getLogger("smart_cut")


class ClipQuality(Enum):
    """片段质量等级"""
    VALID = "valid"              # 有效口播
    INVALID_NOISE = "invalid_noise"    # 噪声/背景音
    INVALID_SHORT = "invalid_short"   # 太短
    INVALID_DUPLICATE = "invalid_duplicate"  # 重复片段
    INVALID_ENERGY = "invalid_energy"      # 能量不足


@dataclass
class ClipData:
    """片段原始数据"""
    clip_id: str
    file_path: str
    text: str                    # ASR 识别文本
    duration: float              # 总时长 (秒)
    audio_db: float              # 音频平均分贝
    speech_segments: List[Tuple[float, float]] = field(default_factory=list)  # VAD 语音段
    word_segments: List[Dict] = field(default_factory=list)  # 词级时间戳
    mtime: float = 0.0           # 文件修改时间


@dataclass
class ClipScore:
    """片段评分结果"""
    clip_id: str
    vad_ratio: float = 0.0       # VAD 占比
    energy_score: float = 0.0   # 能量得分
    length_penalty: float = 0.0 # 长度惩罚
    total_score: float = 0.0   # 总分
    quality: ClipQuality = ClipQuality.VALID
    is_duplicate: bool = False
    duplicate_of: str = ""
    final_duration: float = 0.0 # 考虑 offset 后的最终时长
    
    # 详情
    text_length: int = 0
    speech_duration: float = 0.0
    

class ClipEvaluator:
    """
    通用 A-Roll 评分器
    基于多维特征计算评分，不依赖特定 ASR 模型
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        # 评分权重
        "score_weights": {
            "vad_ratio": 0.40,      # VAD 占比权重
            "energy": 0.30,         # 能量得分权重
            "length": 0.30          # 长度得分权重
        },
        
        # 阈值设置
        "min_score": 0.65,         # 最低入围分数
        "min_text_length": 3,       # 最少文字数
        "min_text_duration": 0.5,    # 最少文字时长 (秒)
        "min_final_duration": 0.4,  # 最小最终时长 (秒)
        "min_audio_db": -45,         # 最低音频分贝
        
        # 去重配置
        "dedup_similarity_threshold": 0.85,  # 相似度阈值
        "dedup_min_score": 0.65,    # 参与去重的最低分数
        
        # 能量归一化参数
        "energy_min_db": -50,       # -50dB = 0分
        "energy_max_db": -10,       # -10dB = 100分
        
        # 缓存配置
        "config_version": "2.0"      # 配置版本 (变更需重算)
    }
    
    def __init__(self, config: Optional[Dict] = None, cache_path: str = "storage/clip_cache.json"):
        """
        初始化评分器
        
        Args:
            config: 配置字典
            cache_path: 缓存文件路径
        """
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)
        
        self.cache_path = Path(cache_path)
        self._cache = self._load_cache()
        
        # 加载相似度计算函数
        self._load_similarity_func()
        
        logger.info(f"ClipEvaluator initialized with config: {self._get_config_hash()[:8]}...")
    
    def _load_similarity_func(self):
        """加载相似度计算函数"""
        self.similarity_func = None
        try:
            from Levenshtein import ratio as levenshtein_ratio
            self.similarity_func = levenshtein_ratio
        except ImportError:
            logger.warning("Levenshtein not installed, using difflib")
    
    def _get_config_hash(self) -> str:
        """计算配置哈希"""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _load_cache(self) -> Dict:
        """加载缓存"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
        return {"clips": {}, "config_hash": ""}
    
    def _save_cache(self):
        """保存缓存"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def is_cache_valid(self, clip_id: str, mtime: float) -> bool:
        """检查缓存是否有效"""
        cached = self._cache.get("clips", {}).get(clip_id)
        if not cached:
            return False
        
        # 检查文件修改时间
        if cached.get("mtime") != mtime:
            return False
        
        # 检查配置是否变化
        if cached.get("config_hash") != self._get_config_hash():
            return False
        
        return True
    
    def calculate_score(self, clip_data: ClipData, 
                        offset_start: float = 0.0, 
                        offset_end: float = 0.0) -> ClipScore:
        """
        计算片段综合评分
        
        公式:
        - vad_ratio: 语音时长 / 总时长
        - energy_score: 归一化能量得分
        - length_penalty: 长度惩罚 (字数少且时长短时扣分)
        - total_score = vad_ratio * 0.4 + energy_score * 0.3 + length_score * 0.3
        
        Args:
            clip_data: 片段数据
            offset_start: 起始偏移 (秒)
            offset_end: 结束偏移 (秒)
            
        Returns:
            ClipScore 评分结果
        """
        cfg = self.config
        weights = cfg["score_weights"]
        
        # 1. 计算 VAD 占比
        total_duration = clip_data.duration
        if total_duration <= 0:
            return ClipScore(
                clip_id=clip_data.clip_id,
                quality=ClipQuality.INVALID_SHORT,
                total_score=0.0
            )
        
        speech_duration = sum(
            end - start 
            for start, end in clip_data.speech_segments
            if start >= offset_start and end <= (total_duration - offset_end)
        )
        vad_ratio = speech_duration / max(0.1, total_duration - offset_start - offset_end)
        vad_ratio = min(1.0, vad_ratio)
        
        # 2. 计算能量得分
        audio_db = clip_data.audio_db
        energy_min = cfg["energy_min_db"]
        energy_max = cfg["energy_max_db"]
        
        if audio_db >= energy_max:
            energy_score = 1.0
        elif audio_db <= energy_min:
            energy_score = 0.0
        else:
            energy_score = (audio_db - energy_min) / (energy_max - energy_min)
        
        # 3. 计算长度得分/惩罚
        text_len = len(clip_data.text.replace(" ", ""))
        text_length = text_len
        min_len = cfg["min_text_length"]
        min_dur = cfg["min_text_duration"]
        
        # 有效文字时长
        valid_speech = sum(
            min(end, total_duration - offset_end) - max(start, offset_start)
            for start, end in clip_data.speech_segments
        )
        
        if text_len < min_len or valid_speech < min_dur:
            # 太短，扣分
            length_penalty = -0.3
            length_score = 0.0
        else:
            # 正常长度，得分与字数成正比
            length_score = min(1.0, text_len / 20)
            length_penalty = 0.0
        
        # 4. 计算总分
        total_score = (
            vad_ratio * weights["vad_ratio"] +
            energy_score * weights["energy"] +
            length_score * weights["length"] +
            length_penalty
        )
        total_score = max(0.0, min(1.0, total_score))
        
        # 5. 质量判定
        quality = ClipQuality.VALID
        if text_len < min_len:
            quality = ClipQuality.INVALID_SHORT
        elif audio_db < cfg["min_audio_db"]:
            quality = ClipQuality.INVALID_ENERGY
        elif vad_ratio < 0.2:
            quality = ClipQuality.INVALID_NOISE
        elif total_score < cfg["min_score"]:
            quality = ClipQuality.INVALID_NOISE
        
        # 6. 计算最终时长 (考虑 offset)
        final_duration = total_duration - offset_start - offset_end
        
        score = ClipScore(
            clip_id=clip_data.clip_id,
            vad_ratio=vad_ratio,
            energy_score=energy_score,
            length_penalty=length_penalty,
            total_score=total_score,
            quality=quality,
            final_duration=final_duration,
            text_length=text_length,
            speech_duration=speech_duration
        )
        
        return score
    
    def check_duration_safety(self, score: ClipScore) -> bool:
        """
        物理安全检查
        
        Args:
            score: 评分结果
            
        Returns:
            是否通过安全检查
        """
        min_dur = self.config["min_final_duration"]
        if score.final_duration < min_dur:
            logger.warning(
                f"片段 {score.clip_id} 时长不足: {score.final_duration:.2f}s < {min_dur}s"
            )
            return False
        return True
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        t1 = re.sub(r'\s+', '', text1).lower()
        t2 = re.sub(r'\s+', '', text2).lower()
        
        if not t1 or not t2:
            return 0.0
        
        if self.similarity_func:
            return self.similarity_func(t1, t2)
        else:
            return difflib.SequenceMatcher(None, t1, t2).ratio()
    
    def deduplicate(self, clips: List[ClipScore], texts: List[str]) -> List[ClipScore]:
        """
        语义去重
        
        对 Score >= min_score 的相邻片段进行相似度计算，
        若相似度 > threshold，仅保留 Score 最高的那段
        
        Args:
            clips: 评分结果列表 (已按时间排序)
            texts: 对应的文本列表
            
        Returns:
            去重后的评分列表
        """
        cfg = self.config
        threshold = cfg["dedup_similarity_threshold"]
        min_score = cfg["dedup_min_score"]
        
        if len(clips) <= 1:
            return clips
        
        # 按 final_duration 排序（假设已经按时间顺序）
        sorted_clips = sorted(enumerate(clips), key=lambda x: x[1].final_duration)
        
        keep = set()
        remove = set()
        
        for i in range(len(sorted_clips)):
            idx_i, clip_i = sorted_clips[i]
            if idx_i in remove:
                continue
            if clip_i.total_score < min_score:
                keep.add(idx_i)
                continue
                
            for j in range(i + 1, len(sorted_clips)):
                idx_j, clip_j = sorted_clips[j]
                if idx_j in remove:
                    continue
                if clip_j.total_score < min_score:
                    continue
                
                sim = self.calculate_similarity(texts[idx_i], texts[idx_j])
                if sim > threshold:
                    # 相似，保留分数高的
                    if clip_i.total_score >= clip_j.total_score:
                        remove.add(idx_j)
                        clip_j.is_duplicate = True
                        clip_j.duplicate_of = clip_i.clip_id
                    else:
                        remove.add(idx_i)
                        clip_i.is_duplicate = True
                        clip_i.duplicate_of = clip_j.clip_id
                        break
        
        # 构建结果
        result = []
        for i, clip in enumerate(clips):
            if i not in remove:
                result.append(clip)
            else:
                clip.quality = ClipQuality.INVALID_DUPLICATE
        
        return result
    
    def evaluate_clips(self, clips: List[ClipData], 
                       offsets: Optional[List[Tuple[float, float]]] = None) -> List[ClipScore]:
        """
        批量评估片段
        
        Args:
            clips: 片段数据列表
            offsets: 偏移量列表 [(start, end), ...]，可选
            
        Returns:
            评分结果列表
        """
        if offsets is None:
            offsets = [(0.0, 0.0)] * len(clips)
        
        scores = []
        for clip, (off_start, off_end) in zip(clips, offsets):
            # 检查缓存
            if self.is_cache_valid(clip.clip_id, clip.mtime):
                cached = self._cache["clips"].get(clip.clip_id, {})
                score = ClipScore(**cached)
                scores.append(score)
                continue
            
            # 计算评分
            score = self.calculate_score(clip, off_start, off_end)
            
            # 物理安全检查
            if not self.check_duration_safety(score):
                score.quality = ClipQuality.INVALID_SHORT
            
            # 保存到缓存
            self._cache["clips"][clip.clip_id] = asdict(score)
            scores.append(score)
        
        # 更新配置哈希
        self._cache["config_hash"] = self._get_config_hash()
        self._save_cache()
        
        return scores
    
    def classify_ab_roll(self, scores: List[ClipScore], texts: List[str]) -> Tuple[List[ClipScore], List[ClipScore]]:
        """
        A/B Roll 分类
        
        Args:
            scores: 评分列表
            texts: 文本列表
            
        Returns:
            (A_ROLL 列表, B_ROLL 列表)
        """
        cfg = self.config
        min_score = cfg["min_score"]
        
        # 先去重
        deduplicated = self.deduplicate(scores, texts)
        
        a_roll = []
        b_roll = []
        
        for score in deduplicated:
            # 必须通过质量检查
            if score.quality != ClipQuality.VALID:
                b_roll.append(score)
                continue
            
            # 必须达到最低分数
            if score.total_score >= min_score:
                a_roll.append(score)
            else:
                b_roll.append(score)
        
        logger.info(
            f"A/B Roll 分类: A={len(a_roll)}, B={len(b_roll)}, "
            f"threshold={min_score}"
        )
        
        return a_roll, b_roll
    
    def get_config_hash(self) -> str:
        """获取当前配置哈希"""
        return self._get_config_hash()


# 辅助函数
import re
