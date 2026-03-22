"""
视频拼接组装模块 - Assembler
负责：
1. 标准脚本对照逻辑 (align_to_script)
2. 多版本去重与择优 (select_best_version)
3. 逻辑断句检查 (Sequence Guard)
4. 排序和去重
"""
import re
import difflib
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger("smart_cut")


@dataclass
class AssembledClip:
    """组装后的片段"""
    clip_id: str
    video_path: str
    video_id: int
    start: float
    end: float
    text: str                    # 匹配的目标文案
    matched_text: str            # 素材实际文本
    similarity: float
    track_type: str              # A_ROLL / B_ROLL
    score: float                 # 综合评分 (能量+VAD)
    is_b_roll: bool = False
    original_index: int = 0      # 原始素材索引（用于排序）
    file_order: int = 0          # 文件内顺序


class Assembler:
    """
    视频组装器
    实现脚本驱动排序、去重、择优
    """
    
    # 去重相似度阈值
    DEDUP_SIMILARITY_THRESHOLD = 0.85
    
    def __init__(self, cache_path: str = "storage/assembler_cache.json"):
        """
        初始化组装器
        
        Args:
            cache_path: 缓存文件路径
        """
        self.cache_path = Path(cache_path)
        self._cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """加载缓存"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"used_clips": set(), "config_hash": ""}
    
    def _save_cache(self):
        """保存缓存"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache["used_clips"] = list(self._cache.get("used_clips", set()))
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def reset_used_clips(self):
        """重置已使用的片段记录"""
        self._cache["used_clips"] = set()
        self._save_cache()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        t1 = re.sub(r'\s+', '', text1).lower()
        t2 = re.sub(r'\s+', '', text2).lower()
        if not t1 or not t2:
            return 0.0
        return difflib.SequenceMatcher(None, t1, t2).ratio()
    
    # ========== 1. 标准脚本对照逻辑 ==========
    
    def align_to_script(self, edl: List[Dict], script: str) -> List[Dict]:
        """
        按脚本顺序对齐素材
        
        逻辑：
        1. 如果提供了标准脚本，按脚本句子顺序排序
        2. 如果没有脚本，按素材原始时间轴排序
        
        Args:
            edl: 原始 EDL 列表
            script: 标准脚本
            
        Returns:
            按脚本顺序排序的 EDL
        """
        if not edl:
            return []
        
        # 分句
        script_sentences = self._split_sentences(script)
        
        if not script_sentences:
            # 无脚本，按原始顺序
            return sorted(edl, key=lambda x: (x.get("video_id", 0), x.get("start", 0)))
        
        # 为每个 EDL 条目计算与脚本句子的匹配度
        def get_script_order(item):
            """获取该素材应该对应的脚本句子索引"""
            text = item.get("matched_text", item.get("text", ""))
            best_idx = 0
            best_sim = 0
            
            for idx, sent in enumerate(script_sentences):
                sim = self.calculate_similarity(text, sent)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx
            
            return best_idx
        
        # 排序：先按脚本顺序，再按原始素材顺序
        sorted_edl = sorted(edl, key=lambda x: (get_script_order(x), x.get("video_id", 0), x.get("start", 0)))
        
        logger.info(f"脚本对照排序完成: {len(sorted_edl)} 个片段")
        
        return sorted_edl
    
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        # 按标点和换行分句
        sentences = re.split(r'[。！？\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    # ========== 2. 多版本去重与择优 ==========
    
    def select_best_version(self, clips: List[Dict]) -> List[Dict]:
        """
        从重复片段中选择最佳版本
        
        判定标准：
        1. 使用 SequenceMatcher 识别文本相似度 >= 0.85 的片段组
        2. 从每组中选择 Score 最高的片段
        3. 确保每个语义片段只出现一次
        
        Args:
            clips: 片段列表
            
        Returns:
            去重后的最佳片段列表
        """
        if len(clips) <= 1:
            return clips
        
        # 按视频和时间排序（保持原始顺序）
        sorted_clips = sorted(clips, key=lambda x: (
            x.get("video_id", 0), 
            x.get("start", 0)
        ))
        
        # 聚类：相似片段分组
        groups = self._cluster_similar_clips(sorted_clips)
        
        # 从每组选择最佳
        selected = []
        for group in groups:
            if len(group) == 1:
                selected.append(group[0])
            else:
                # 按评分排序，选择最高的
                best = max(group, key=lambda x: x.get("score", x.get("similarity", 0)))
                selected.append(best)
                # 记录被舍弃的
                for c in group:
                    if c is not best:
                        c["_dropped"] = True
                        logger.info(f"去重舍弃: {c.get('video_path', 'unknown')} (已有更高分版本)")
        
        logger.info(f"去重完成: {len(clips)} -> {len(selected)} 片段")
        
        return selected
    
    def _cluster_similar_clips(self, clips: List[Dict]) -> List[List[Dict]]:
        """将相似片段聚类分组"""
        if not clips:
            return []
        
        groups = []
        used = set()
        
        for i, clip in enumerate(clips):
            if i in used:
                continue
            
            # 当前组
            current_group = [clip]
            used.add(i)
            
            # 找相似的
            for j in range(i + 1, len(clips)):
                if j in used:
                    continue
                
                sim = self.calculate_similarity(
                    clip.get("matched_text", ""),
                    clips[j].get("matched_text", "")
                )
                
                if sim >= self.DEDUP_SIMILARITY_THRESHOLD:
                    current_group.append(clips[j])
                    used.add(j)
            
            groups.append(current_group)
        
        return groups
    
    # ========== 3. 逻辑断句检查 (Sequence Guard) ==========
    
    def check_sequence_guard(self, clips: List[Dict]) -> List[Dict]:
        """
        检查片段衔接是否完整
        
        逻辑：
        1. 同一视频连续片段：保持原样
        2. 不同视频片段：检查"半句话"问题
        3. 需要 crossfade 时标记
        
        Args:
            clips: 排序后的片段列表
            
        Returns:
            处理后的片段列表
        """
        if len(clips) <= 1:
            return clips
        
        result = []
        
        for i, clip in enumerate(clips):
            # 检查是否需要 crossfade
            need_crossfade = False
            is_incomplete = False
            
            if i > 0:
                prev = clips[i - 1]
                
                # 来自不同视频或不是连续时间点
                if prev.get("video_id") != clip.get("video_id"):
                    need_crossfade = True
                    
                    # 检查是否是"半句话"（文本不完整）
                    # 简单判断：前一句结尾没有句号，后一句开头是常见连接词
                    prev_text = prev.get("matched_text", "")
                    curr_text = clip.get("matched_text", "")
                    
                    # 如果前一句不以标点结尾，可能是半句话
                    if prev_text and not re.search(r'[。！？]$', prev_text):
                        # 检查当前句是否可能是承接句
                        if curr_text and curr_text[0] in '但是在而且所以因为':
                            is_incomplete = True
                            logger.warning(
                                f"检测到半句话: {prev.get('video_path')} -> {clip.get('video_path')}"
                            )
            
            # 添加标记
            clip["need_crossfade"] = need_crossfade
            clip["is_incomplete"] = is_incomplete
            
            # 如果是不完整的半句话，尝试合并或跳过
            if is_incomplete:
                # 标记前一帧需要延尾
                if result:
                    result[-1]["extend_tail"] = True
            
            result.append(clip)
        
        return result
    
    # ========== 4. 综合组装流程 ==========
    
    def assemble(self, edl: List[Dict], script: str = "", 
                 apply_dedup: bool = True,
                 apply_sequence_guard: bool = True) -> List[Dict]:
        """
        综合组装流程
        
        流程：
        1. 排序：按脚本顺序或原始时间轴
        2. 去重：选择最佳版本
        3. 断句检查：处理半句话和 crossfade
        
        Args:
            edl: 原始 EDL
            script: 标准脚本
            apply_dedup: 是否应用去重
            apply_sequence_guard: 是否应用断句检查
            
        Returns:
            组装后的 EDL
        """
        if not edl:
            return []
        
        result = edl
        
        # 1. 脚本对照排序
        if script:
            result = self.align_to_script(result, script)
            logger.info(f"脚本对照排序: {len(result)} 片段")
        
        # 2. 去重择优
        if apply_dedup:
            result = self.select_best_version(result)
        
        # 3. 断句检查
        if apply_sequence_guard:
            result = self.check_sequence_guard(result)
        
        # 4. 确保不重复使用同一素材
        result = self._ensure_unique_usage(result)
        
        logger.info(f"组装完成: {len(result)} 片段")
        
        return result
    
    def _ensure_unique_usage(self, clips: List[Dict]) -> List[Dict]:
        """确保每个素材只使用一次"""
        used_videos = set()
        result = []
        
        for clip in clips:
            video_id = clip.get("video_id")
            video_path = clip.get("video_path", "")
            
            # 如果这个视频已经在之前使用过
            if video_id in used_videos:
                # 尝试找替代
                logger.warning(f"素材重复使用，跳过: {Path(video_path).name}")
                continue
            
            used_videos.add(video_id)
            result.append(clip)
        
        if len(result) < len(clips):
            logger.info(f"去重: {len(clips)} -> {len(result)} 片段")
        
        return result
    
    # ========== 5. 最终排序（防止多线程乱序）==========
    
    def sort_by_timestamp(self, clips: List[Dict]) -> List[Dict]:
        """
        按时间戳排序
        
        防止多线程处理导致的乱序问题
        
        Args:
            clips: 片段列表
            
        Returns:
            排序后的列表
        """
        return sorted(clips, key=lambda x: (
            x.get("video_id", 0),
            x.get("start", 0),
            x.get("original_index", 0)
        ))
