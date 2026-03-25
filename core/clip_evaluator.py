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

from .config import config

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
    
    # 新增：语义完整性
    semantic_integrity: float = 1.0  # 语义完整性得分 (0-1)
    has_end_punctuation: bool = True  # 是否有终止标点
    char_per_second: float = 0.0     # 语速 (字/秒)
    has_stutter: bool = False        # 是否有卡顿/重复词
    stutter_count: int = 0           # 卡顿次数
    
    # 详情
    text_length: int = 0
    speech_duration: float = 0.0
    

class ClipEvaluator:
    """
    通用 A-Roll 评分器
    基于多维特征计算评分，不依赖特定 ASR 模型
    """
    

    
    def __init__(self, evaluator_config: Optional[Dict] = None, cache_path: Optional[str] = None):
        """
        初始化评分器

        Args:
            evaluator_config: 评分配置字典（可选，默认使用全局配置）
            cache_path: 缓存文件路径（可选）
        """
        # 使用全局配置或传入的配置
        if evaluator_config is None:
            self.config = config.evaluator_config.copy()
        else:
            self.config = evaluator_config.copy()

        # 设置缓存路径
        if cache_path is None:
            cache_path = "storage/clip_cache.json"
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
        - semantic_integrity: 语义完整性（标点 + 语速）
        - stutter_penalty: 卡顿惩罚
        - total_score = vad_ratio * 0.3 + energy_score * 0.25 + length_score * 0.2 + semantic_integrity * 0.25 + stutter_penalty
        
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
        
        # 4. 计算语义完整性得分
        # 使用有效语音时长（考虑 offset）
        effective_duration = max(0.1, valid_speech)
        semantic_integrity, semantic_details = calculate_semantic_integrity(
            clip_data.text, effective_duration
        )
        has_end_punctuation = semantic_details["has_end_punc"]
        char_per_second = semantic_details["char_per_second"]
        
        # 5. 计算卡顿惩罚
        stutter_penalty = calculate_stutter_penalty(clip_data.text)
        has_stutter = stutter_penalty < 0
        stutter_count = 1 if has_stutter else 0
        
        # 6. 计算总分
        total_score = (
            vad_ratio * weights["vad_ratio"] +
            energy_score * weights["energy"] +
            length_score * weights["length"] +
            semantic_integrity * weights.get("semantic_integrity", 0.25) +
            stutter_penalty +
            length_penalty
        )
        total_score = max(0.0, min(1.0, total_score))
        
        # 7. 质量判定
        quality = ClipQuality.VALID
        if text_len < min_len:
            quality = ClipQuality.INVALID_SHORT
        elif audio_db < cfg["min_audio_db"]:
            quality = ClipQuality.INVALID_ENERGY
        elif vad_ratio < 0.2:
            quality = ClipQuality.INVALID_NOISE
        elif total_score < cfg["min_score"]:
            quality = ClipQuality.INVALID_NOISE
        
        # 8. 计算最终时长 (考虑 offset)
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
            speech_duration=speech_duration,
            # 新增字段
            semantic_integrity=semantic_integrity,
            has_end_punctuation=has_end_punctuation,
            char_per_second=char_per_second,
            has_stutter=has_stutter,
            stutter_count=stutter_count
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


# ========== 语义完整性评分 ==========

def calculate_semantic_integrity(text: str, duration: float) -> Tuple[float, Dict]:
    """
    计算语义完整性得分
    
    评估维度：
    1. 标点检查：片段末尾是否有终止标点（。！？）
    2. 字数/时长比：字符数/秒，异常则扣分
    
    Args:
        text: ASR 文本
        duration: 片段时长（秒）
        
    Returns:
        (完整性得分 0-1, 详情字典)
    """
    details = {
        "has_end_punc": False,
        "char_per_second": 0.0,
        "speed_rating": "normal"
    }
    
    if not text or duration <= 0:
        return 0.0, details
    
    # 1. 标点检查
    # 移除空格后的文本
    clean_text = text.strip()
    has_end_punc = bool(re.search(r'[。！？]$', clean_text))
    details["has_end_punc"] = has_end_punc
    
    # 2. 字数/时长比
    char_count = len(clean_text)
    char_per_second = char_count / duration
    details["char_per_second"] = char_per_second
    
    # 3. 评分计算
    score = 0.0
    
    # 标点得分：如果有终止标点，得 0.5 分
    if has_end_punc:
        score += 0.5
    
    # 语速得分：正常范围 3-8 字/秒
    # 过快：>8 字/秒（抢话）；过慢：<3 字/秒（长停顿/卡顿）
    if 3 <= char_per_second <= 8:
        score += 0.5  # 正常语速满分
    elif char_per_second < 3:
        # 过慢，扣分
        score += max(0, 0.5 * (char_per_second / 3))
    else:
        # 过快，扣分
        score += max(0, 0.5 * (8 / char_per_second))
    
    # 速度评级
    if char_per_second < 3:
        details["speed_rating"] = "slow"
    elif char_per_second > 8:
        details["speed_rating"] = "fast"
    else:
        details["speed_rating"] = "normal"
    
    return score, details


def detect_stutter(text: str) -> Tuple[bool, List[Dict]]:
    """
    检测卡顿与重复词（通用方法）
    
    检测逻辑：
    - 方法1：逗号分隔的重复词 "我、我、我"
    - 方法2：空格分隔的重复词 "就是就是就是"
    - 方法3：口吃型单字重复 "那那那"
    - 方法4：无标点连续重复词 "不要不要" "一遍一遍"
    - 方法5：带语气词的重复 "就是就是嘛"
    - 方法6：混合标点重复 "不要，不要，不要"
    - 方法7：数字/量词重复 "一次一次"
    - 方法8：间隔重复短语 "今天带你们和...今天带你们和"（口播重复）
    
    Args:
        text: ASR 文本
        
    Returns:
        (是否有卡顿, 卡顿详情列表)
    """
    if not text:
        return False, []
    
    stutters = []
    clean_text = text.strip()
    
    # 记录已检测到的位置，避免重复
    detected_positions = set()
    
    def add_stutter(stype, word, match_obj):
        """添加卡顿检测结果，避免重复位置"""
        pos = match_obj.start()
        # 允许小范围内重叠（如"不要不要"和"不要"）
        for existing_pos in detected_positions:
            if abs(pos - existing_pos) < len(word):
                return  # 跳过重复位置
        detected_positions.add(pos)
        stutters.append({
            "type": stype,
            "word": word,
            "match": match_obj.group(0),
            "position": pos
        })
    
    # 方法1：检测逗号分隔的重复词 "我、我、我"
    repeat_pattern = re.compile(r'([\u4e00-\u9fa5a-zA-Z0-9]{1,3})[、,，]\1(?:[、,，]\1)+')
    for match in repeat_pattern.finditer(clean_text):
        add_stutter("comma_repeat", match.group(1), match)
    
    # 方法2：检测空格分隔的重复词 "就是就是就是"（连续出现3次及以上）
    space_repeat_pattern = re.compile(r'(\S+)(?:\s+\1){2,}')
    for match in space_repeat_pattern.finditer(clean_text):
        word = match.group(1)
        if len(word) >= 1 and not re.match(r'^[。！？，、；：]$', word):
            add_stutter("space_repeat", word, match)
    
    # 方法3：检测口吃型单字重复 "那那那"（3个及以上相同字）
    stutter_pattern = re.compile(r'([\u4e00-\u9fa5a-zA-Z0-9])\1{2,}')
    for match in stutter_pattern.finditer(clean_text):
        char = match.group(1)
        full_match = match.group(0)
        if len(full_match) >= 3:
            add_stutter("stutter", char, match)
    
    # 方法4：无标点连续重复词 "不要不要" "一遍一遍" "可以吗可以吗"
    # 匹配：词/短语直接连续重复2次及以上（无标点分隔）
    no_punc_repeat = re.compile(r'([\u4e00-\u9fa5a-zA-Z0-9]{1,6})\1+')
    for match in no_punc_repeat.finditer(clean_text):
        word = match.group(1)
        full_match = match.group(0)
        # 排除过短匹配，确保是真正的重复
        if len(word) >= 2 or (len(word) == 1 and len(full_match) >= 3):
            add_stutter("no_punc_repeat", word, match)
    
    # 方法5：带语气词的重复 "就是就是嘛" "不要不要啊"
    # 匹配：词 + 语气词/助词 + 重复
    modal_repeat = re.compile(r'([\u4e00-\u9fa5a-zA-Z0-9]{1,4})(?:吗|啊|呢|吧|呀|哦|嗯|哈|嘿|嘛)[的个着过]?\1')
    for match in modal_repeat.finditer(clean_text):
        add_stutter("modal_repeat", match.group(1), match)
    
    # 方法6：带连接符的重复 "不要-不要" "one-one-one"
    connector_repeat = re.compile(r'([\u4e00-\u9fa5a-zA-Z0-9]+)[-–—]\1(?:[-–—]\1)*')
    for match in connector_repeat.finditer(clean_text):
        add_stutter("connector_repeat", match.group(1), match)
    
    # 方法7：检测数字+量词重复 "一次一次" "一遍一遍" "一下一下"
    # 匹配：数词/量词直接连续重复
    num_measure_repeat = re.compile(r'([一二三四五六七八九十0-9]+[次遍个下]){2,}')
    for match in num_measure_repeat.finditer(clean_text):
        add_stutter("num_measure_repeat", match.group(1), match)
    
    # 方法8：检测间隔重复短语 "今天带你们和...今天带你们和"（口播时重复说话）
    # 去除空格后检测：短语重复（间隔5个字符以内）
    text_no_space = clean_text.replace(" ", "")
    for length in range(4, 12):  # 短语长度 4-11 个字符
        for i in range(len(text_no_space) - length * 2):
            substr = text_no_space[i:i+length]
            next_pos = text_no_space.find(substr, i + length)
            if next_pos != -1:
                gap = next_pos - (i + length)
                # 间隔不超过5个字符，认为是重复
                if gap <= 5:
                    # 创建虚拟匹配对象
                    class FakeMatch:
                        def __init__(self, start, end, group0):
                            self._start = start
                            self._end = end
                            self._group0 = group0
                        def start(self):
                            return self._start
                        def end(self):
                            return self._end
                        def group(self, n=0):
                            return self._group0 if n == 0 else substr
                    
                    # 检查是否已检测过
                    add_stutter("interval_repeat", substr, FakeMatch(i, next_pos + length, f"...{substr}..."))
    
    has_stutter = len(stutters) > 0
    
    if has_stutter:
        logger.debug(f"检测到卡顿: {stutters}")
    
    return has_stutter, stutters


def calculate_stutter_penalty(text: str) -> float:
    """
    计算卡顿惩罚分数
    
    有卡顿的片段会降低评分
    
    Args:
        text: ASR 文本
        
    Returns:
        惩罚分数（负值），0 表示无惩罚
    """
    has_stutter, stutters = detect_stutter(text)
    
    if not has_stutter:
        return 0.0
    
    # 根据卡顿次数扣分
    # 1次卡顿：-0.1
    # 2次卡顿：-0.2
    # 3次及以上：-0.3
    count = len(stutters)
    penalty = -min(0.3, count * 0.1)
    
    return penalty


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串之间的编辑距离（Levenshtein Distance）
    
    Args:
        s1: 字符串1
        s2: 字符串2
        
    Returns:
        编辑距离
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_text_similarity(s1: str, s2: str) -> float:
    """
    使用编辑距离计算两个字符串的相似度
    
    Args:
        s1: 字符串1
        s2: 字符串2
        
    Returns:
        相似度（0-1），1 表示完全相同
    """
    if not s1 or not s2:
        return 0.0
    
    # 去除空格比较
    s1 = s1.replace(" ", "")
    s2 = s2.replace(" ", "")
    
    if s1 == s2:
        return 1.0
    
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    
    distance = levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


def calculate_text_cleanliness(text: str) -> Tuple[float, Dict]:
    """
    计算文本清洁度得分（基于滑动窗口+编辑距离）
    
    使用滑动窗口算法检测片段内部的语义重复：
    - 移动两个窗口，如果相似度超过阈值，判定为重复
    - 计算重复比例作为清洁度得分
    
    Args:
        text: ASR 文本（带空格）
        
    Returns:
        (清洁度得分 0-1, 详情字典)
    """
    details = {
        "repeat_count": 0,
        "repeat_ratio": 0.0,
        "max_repeat_length": 0,
        "repeats": []
    }
    
    if not text or len(text) < 8:
        return 1.0, details
    
    # 去除空格
    text_no_space = text.replace(" ", "")
    length = len(text_no_space)
    
    if length < 8:
        return 1.0, details
    
    # 滑动窗口参数
    min_window_size = 4
    max_window_size = min(length // 2, 12)  # 窗口最大为文本长度的一半，上限12
    similarity_threshold = 0.7  # 相似度阈值
    max_gap = 5  # 最大间隔字符数
    
    repeats = []
    
    # 对每个窗口大小进行检测
    for window_size in range(min_window_size, max_window_size + 1):
        for i in range(length - window_size * 2):
            window_a = text_no_space[i:i + window_size]
            # 在间隔范围内查找相似窗口
            search_start = i + window_size
            search_end = min(search_start + max_gap + window_size, length - window_size)
            
            for j in range(search_start, search_end):
                window_b = text_no_space[j:j + window_size]
                
                # 计算相似度
                sim = calculate_text_similarity(window_a, window_b)
                
                if sim >= similarity_threshold:
                    repeats.append({
                        "text": window_a,
                        "pos1": i,
                        "pos2": j,
                        "length": window_size,
                        "similarity": sim,
                        "gap": j - (i + window_size)
                    })
                    break  # 只记录第一次发现
    
    # 去重：合并重叠的重复
    if repeats:
        repeats = _merge_overlapping_repeats(repeats)
    
    details["repeat_count"] = len(repeats)
    details["repeats"] = repeats
    
    if repeats:
        max_len = max(r.get("length", 0) for r in repeats)
        details["max_repeat_length"] = max_len
        # 重复比例：重复字符数 / 总字符数
        total_repeat_chars = sum(r.get("length", 0) for r in repeats)
        details["repeat_ratio"] = min(1.0, total_repeat_chars / length)
    
    # 清洁度得分：根据重复次数和比例计算
    # 无重复 = 1.0，有重复则降低
    if details["repeat_count"] == 0:
        cleanliness = 1.0
    else:
        # 基础得分
        base_score = 1.0
        # 每次重复扣0.15
        repeat_penalty = details["repeat_count"] * 0.15
        # 重复比例扣分
        ratio_penalty = details["repeat_ratio"] * 0.3
        cleanliness = max(0.0, base_score - repeat_penalty - ratio_penalty)
    
    return cleanliness, details


def _merge_overlapping_repeats(repeats: List[Dict]) -> List[Dict]:
    """合并重叠的重复检测结果"""
    if not repeats:
        return []
    
    # 按位置排序
    sorted_repeats = sorted(repeats, key=lambda x: x["pos1"])
    merged = []
    
    for repeat in sorted_repeats:
        if not merged:
            merged.append(repeat)
            continue
        
        last = merged[-1]
        # 如果有重叠，保留较长的
        if repeat["pos1"] < last["pos1"] + last["length"]:
            if repeat["length"] > last["length"]:
                merged[-1] = repeat
        else:
            merged.append(repeat)
    
    return merged


def find_last_repeat_point(text: str) -> Optional[Dict]:
    """
    查找最后一次重复的起始位置（用于裁剪）
    
    Args:
        text: ASR 文本
        
    Returns:
        重复信息字典，包含起始位置、重复文本等，如果没有重复返回 None
    """
    if not text or len(text) < 8:
        return None
    
    text_no_space = text.replace(" ", "")
    length = len(text_no_space)
    
    # 查找所有重复位置
    repeats = []
    min_window_size = 4
    max_window_size = min(length // 2, 12)
    similarity_threshold = 0.7
    max_gap = 5
    
    for window_size in range(min_window_size, max_window_size + 1):
        for i in range(length - window_size * 2):
            window_a = text_no_space[i:i + window_size]
            search_start = i + window_size
            search_end = min(search_start + max_gap + window_size, length - window_size)
            
            for j in range(search_start, search_end):
                window_b = text_no_space[j:j + window_size]
                sim = calculate_text_similarity(window_a, window_b)
                
                if sim >= similarity_threshold:
                    repeats.append({
                        "text": window_a,
                        "start_pos": i,
                        "end_pos": i + window_size,
                        "repeat_start": j,
                        "repeat_end": j + window_size,
                        "length": window_size,
                        "similarity": sim,
                        "gap": j - (i + window_size)
                    })
                    break
    
    if not repeats:
        return None
    
    # 选择最佳重复：优先选择位置最靠后的（最后一次重复）
    # 这样可以保留最多有效内容：A-B-B' 模式保留 B' 及之后
    best_repeat = max(repeats, key=lambda x: x["repeat_start"])
    return best_repeat


def self_heal_stutter(text: str, word_timestamps: List[Dict] = None, 
                       min_duration: float = 0.5) -> Tuple[str, Optional[float], Dict]:
    """
    自动修复卡顿文本（回溯裁剪）- 修正版
    
    核心逻辑：
    1. 使用 rfind 找到重复短语最后一次出现的位置
    2. 保留从该位置到结尾的所有内容（A-B-B' 模式保留 B' 及之后）
    3. 语义锚点保护：确保保留完整句子
    4. 时长校验：裁剪后时长不足则放弃
    
    Args:
        text: ASR 文本（带空格）
        word_timestamps: 词级时间戳列表 [{"text": "字", "start": 0.0, "end": 0.1}, ...]
        min_duration: 裁剪后最短时长（秒），默认0.5秒
        
    Returns:
        (裁剪后的文本, 新的起始时间(秒), 详情字典)
    """
    details = {
        "was_trimmed": False,
        "original_length": len(text),
        "trimmed_length": 0,
        "trimmed_chars": 0,
        "repeat_info": None,
        "new_start_time": 0.0,
        "new_end_time": None,
        "trimmed_duration": 0.0,
        "kept_duration": 0.0,
        "should_drop": False,
        "drop_reason": ""
    }
    
    if not text:
        return text, None, details
    
    # 去除空格后的文本
    text_no_space = text.replace(" ", "")
    
    # 查找重复信息
    # 直接在文本上查找精确重复的子串
    # 策略：优先选择较长的重复（更有意义），其次选择位置靠后的
    
    all_repeats = []
    
    for length in range(8, 3, -1):  # 优先找较长的重复
        for i in range(len(text_no_space) - length * 2):
            substr = text_no_space[i:i+length]
            # 检查这个子串是否在后面重复
            next_pos = text_no_space.find(substr, i + length)
            if next_pos != -1:
                gap = next_pos - (i + length)
                if gap <= 5:  # 间隔小于5个字符
                    all_repeats.append({
                        "text": substr,
                        "first_pos": i,
                        "repeat_start": next_pos,
                        "length": length,
                        "gap": gap
                    })
    
    if not all_repeats:
        # 如果没找到精确重复，回退到使用 calculate_text_cleanliness
        cleanliness, clean_details = calculate_text_cleanliness(text)
        if clean_details.get("repeat_count", 0) > 0:
            repeats = clean_details.get("repeats", [])
            # 选择位置最靠后的
            best_repeat_info = max(repeats, key=lambda x: x["pos1"])
            best_repeat = best_repeat_info["text"]
            last_occurrence_idx = text_no_space.rfind(best_repeat)
        else:
            return text, None, details
    else:
        # 优先选择较长的重复（更有意义），长度相同则选择位置靠后的
        # 这样可以保留完整的词组
        best_repeat_info = max(all_repeats, key=lambda x: (x["length"], x["repeat_start"]))
        # 保留从第二次重复开始的所有内容
        last_occurrence_idx = best_repeat_info["repeat_start"]
        best_repeat = best_repeat_info["text"]
    
    if last_occurrence_idx == -1:
        return text, None, details
    
    # 保留从最后一次重复开始到结尾的所有内容
    kept_text = text_no_space[last_occurrence_idx:]
    
    details["was_trimmed"] = True
    details["trimmed_chars"] = last_occurrence_idx
    details["trimmed_length"] = len(text_no_space) - len(kept_text)
    details["repeat_info"] = best_repeat
    
    # ===== 语义锚点保护：确保保留完整句子 =====
    # 检查片段末尾是否有终止标点
    has_end_punc = bool(re.search(r'[。！？]$', text_no_space))
    
    if has_end_punc:
        # 末尾有完整句子，保留从最后一次重复开始到末尾
        # 不需要额外处理
        pass
    else:
        # 末尾没有终止标点，尝试找到最后一个逗号/顿号，保留之后的内容
        # 这确保我们从一个完整的小句子开始
        for punc in "，、；：":
            punc_pos = kept_text.rfind(punc)
            if punc_pos != -1 and punc_pos < len(kept_text) - 1:
                kept_text = kept_text[punc_pos + 1:]
                details["trimmed_chars"] += punc_pos + 1
                details["trimmed_length"] += punc_pos + 1
                break
    
    # ===== 计算新的时间轴 =====
    new_start_time = 0.0
    new_end_time = None
    
    if word_timestamps and len(word_timestamps) > 0:
        # 原始片段时长
        original_duration = word_timestamps[-1].get("end", 0.0) - word_timestamps[0].get("start", 0.0)
        details["trimmed_duration"] = original_duration
        
        # 计算保留部分的时长比例
        char_ratio = len(kept_text) / len(text_no_space) if text_no_space else 0
        kept_duration = original_duration * char_ratio
        details["kept_duration"] = kept_duration
        
        # 找到保留文本第一个字符在原始文本中的位置
        target_char = kept_text[0] if kept_text else ""
        
        # 尝试精确匹配
        char_found = False
        for i, ts in enumerate(word_timestamps):
            if ts.get("text", "").replace(" ", "") == target_char:
                new_start_time = ts.get("start", 0.0)
                char_found = True
                break
        
        # 如果没找到精确匹配，用比例估算
        if not char_found:
            new_start_time = word_timestamps[0].get("start", 0.0) + original_duration * (1 - char_ratio)
        
        # 保留结束时间不变
        new_end_time = word_timestamps[-1].get("end", 0.0)
    else:
        # 没有时间戳，用字符比例估算
        original_length = len(text_no_space)
        char_ratio = len(kept_text) / original_length if original_length > 0 else 0
        details["kept_duration"] = char_ratio  # 比例作为估算
    
    details["new_start_time"] = new_start_time
    details["new_end_time"] = new_end_time
    
    # ===== 裁剪后时长校验 =====
    kept_len = len(kept_text)
    if kept_len < 4:
        # 保留内容太少，可能是误报
        details["should_drop"] = True
        details["drop_reason"] = f"裁剪后文本过短({kept_len}字符)，可能是误报"
    elif details.get("kept_duration", 0) > 0 and details["kept_duration"] < min_duration:
        # 裁剪后时长不足
        details["should_drop"] = True
        details["drop_reason"] = f"裁剪后时长不足({details['kept_duration']:.2f}秒 < {min_duration}秒)"
    
    return kept_text, new_start_time, details


def detect_stutter_extended(text: str, word_timestamps: List[Dict] = None) -> Tuple[bool, List[Dict], Dict]:
    """
    增强版卡顿检测（返回详细信息）
    
    在 detect_stutter 基础上添加：
    - 文本清洁度得分
    - 自动裁剪建议
    
    Args:
        text: ASR 文本
        word_timestamps: 词级时间戳（可选）
        
    Returns:
        (是否有卡顿, 卡顿详情列表, 额外信息)
    """
    # 基本卡顿检测
    has_stutter, stutters = detect_stutter(text)
    
    # 额外信息
    extra = {
        "text_cleanliness": 1.0,
        "cleanliness_details": {},
        "trim_suggestion": None,
        "trim_details": {},
        "should_drop": False,  # 是否应该丢弃该片段
        "drop_reason": ""
    }
    
    # 计算文本清洁度
    cleanliness, cleanliness_details = calculate_text_cleanliness(text)
    extra["text_cleanliness"] = cleanliness
    extra["cleanliness_details"] = cleanliness_details
    
    # 检查是否需要裁剪
    trim_text, trim_time, trim_details = self_heal_stutter(text, word_timestamps)
    extra["trim_suggestion"] = trim_text
    extra["trim_details"] = trim_details
    
    # 判断是否应该丢弃
    # 规则1：卡顿次数超过2次且无法精准裁剪
    stutter_count = len(stutters)
    if stutter_count >= 3:
        extra["should_drop"] = True
        extra["drop_reason"] = f"卡顿次数过多({stutter_count}次)且无法精准裁剪"
    # 规则2：文本清洁度过低
    elif cleanliness < 0.3:
        extra["should_drop"] = True
        extra["drop_reason"] = f"文本清洁度过低({cleanliness:.2f})"
    # 规则3：裁剪后时长不足
    elif trim_details.get("should_drop", False):
        extra["should_drop"] = True
        extra["drop_reason"] = trim_details.get("drop_reason", "裁剪后时长不足")
    # 规则4：有间隔重复但无法有效裁剪
    elif trim_details.get("was_trimmed", False) and trim_details.get("trimmed_chars", 0) > len(text.replace(" ", "")) * 0.7:
        # 裁剪太多，丢弃
        extra["should_drop"] = True
        extra["drop_reason"] = "裁剪比例过大，保留内容不足"
    
    return has_stutter, stutters, extra


# 辅助函数
import re
