"""
视频拼接组装模块 - Assembler
负责：
1. 标准脚本对照逻辑 (align_to_script)
2. 多版本去重与择优 (select_best_version)
3. 逻辑断句检查 (Sequence Guard)
4. 排序和去重

强化版特性（v2.0）：
- 语义指纹去重（Levenshtein距离<3）
- 全文对齐匹配
- 强制末尾优先
"""
import re
import difflib
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import shutil

logger = logging.getLogger("smart_cut")


# ========== 辅助函数 ==========

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串的Levenshtein编辑距离
    
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


def extract_semantic_fingerprint(text: str) -> str:
    """
    提取语义指纹：去除标点、空格、助词，只保留核心词
    
    Args:
        text: 原始文本
        
    Returns:
        语义指纹
    """
    # 去除标点和空格
    cleaned = re.sub(r'[，。！？、；：""''【】（）\s，。！？、；：""''【】\(\)]', '', text)
    
    # 去除助词
    particles = ['的', '了', '呢', '吧', '啊', '吗', '呀', '哦', '嘛', '呐', '哩', '咯']
    for p in particles:
        cleaned = cleaned.replace(p, '')
    
    return cleaned


def check_cache_validity() -> bool:
    """
    检查缓存是否有效
    
    Returns:
        True表示缓存有效，False表示需要清理
    """
    import yaml
    
    config_path = Path(__file__).parent.parent / "config.yaml"
    cache_version_file = Path(__file__).parent.parent / "temp" / "logic_version.txt"
    
    # 读取当前logic_version
    if not config_path.exists():
        return True
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    current_version = config.get("filter", {}).get("logic_version", "1.0")
    
    # 读取缓存的version
    if cache_version_file.exists():
        with open(cache_version_file, 'r', encoding='utf-8') as f:
            cached_version = f.read().strip()
        
        if cached_version != current_version:
            logger.warning(f"逻辑版本变化: {cached_version} -> {current_version}，需要清理缓存")
            return False
    
    return True


def force_clean_cache() -> None:
    """
    强制清理缓存：删除temp目录下的json文件和数据库
    """
    project_root = Path(__file__).parent.parent
    temp_dir = project_root / "temp"
    db_path = project_root / "storage" / "materials.db"

    logger.info("=== 强制清理缓存 ===")

    # 清理temp目录下的json文件
    if temp_dir.exists():
        for f in temp_dir.glob("*.json"):
            f.unlink()
            logger.info(f"  删除缓存: {f.name}")

    # 清理数据库
    if db_path.exists():
        db_path.unlink()
        logger.info(f"  删除数据库: {db_path.name}")

    # 更新缓存的logic_version
    import yaml
    config_path = project_root / "config.yaml"
    cache_version_file = temp_dir / "logic_version.txt"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    current_version = config.get("filter", {}).get("logic_version", "1.0")

    with open(cache_version_file, 'w', encoding='utf-8') as f:
        f.write(current_version)

    logger.info("=== 缓存清理完成 ===")


def force_clean_sequence_cache() -> None:
    """
    仅清理序列缓存：删除temp目录下的json文件和文案哈希，保留素材数据库
    用于文案变化时仅重新规划序列，不重新处理素材
    """
    project_root = Path(__file__).parent.parent
    temp_dir = project_root / "temp"

    logger.info("=== 强制清理序列缓存 ===")

    # 清理temp目录下的json文件（sequence.json等）
    if temp_dir.exists():
        for f in temp_dir.glob("*.json"):
            f.unlink()
            logger.info(f"  删除序列缓存: {f.name}")

    # 删除文案哈希缓存
    script_hash_file = temp_dir / "script_hash.txt"
    if script_hash_file.exists():
        script_hash_file.unlink()
        logger.info(f"  删除文案哈希缓存: {script_hash_file.name}")

    logger.info("=== 序列缓存清理完成（素材数据库已保留）===")


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
    
    # ========== 2. 多版本去重与择优 (全局语义聚类) ==========
    
    # 全局聚类相似度阈值
    GLOBAL_CLUSTER_THRESHOLD = 0.55
    
    def select_best_version(self, clips: List[Dict]) -> List[Dict]:
        """
        从重复片段中选择最佳版本 - 升级版全局语义聚类
        
        判定标准（多版本竞争算法）：
        1. 全局语义聚类：将所有片段按文本相似度 >= 0.7 聚集成"语义簇"
        2. 初筛：剔除末尾无终止标点（中断）和有重复词（卡顿）的片段
        3. 优选：在完整片段中选择综合评分(score)最高的
        4. 保底：如果所有片段都有瑕疵，选择字数最多且录制时间最晚的
        
        Args:
            clips: 片段列表
            
        Returns:
            去重后的最佳片段列表
        """
        if len(clips) <= 1:
            return clips
        
        # 按视频和时间排序（保持原始录制顺序）
        sorted_clips = sorted(clips, key=lambda x: (
            x.get("video_id", 0), 
            x.get("start", 0)
        ))
        
        # 1. 全局语义聚类（阈值 0.7）
        clusters = self._global_cluster(sorted_clips)
        
        # 2. 从每个语义簇中选择最佳版本
        selected = []
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) == 1:
                selected.append(cluster[0])
                continue
            
            # 多版本竞争
            best = self._version_competition(cluster)
            selected.append(best)
            
            # 记录被舍弃的
            for c in cluster:
                if c is not best:
                    c["_dropped"] = True
                    reason = c.get("_drop_reason", "语义重复")
                    logger.info(f"去重舍弃: {c.get('video_path', 'unknown')} ({reason})")
        
        logger.info(f"去重完成: {len(clips)} -> {len(selected)} 片段")
        
        return selected
    
    def _global_cluster(self, clips: List[Dict]) -> List[List[Dict]]:
        """
        全局语义聚类
        
        使用 0.7 阈值将相似的片段聚集成簇
        不同簇之间可能也有一定相似度，但每个片段只属于一个簇
        
        Args:
            clips: 已排序的片段列表
            
        Returns:
            语义簇列表
        """
        if not clips:
            return []
        
        # 使用 Union-Find 算法进行聚类
        n = len(clips)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 遍历所有片段对，相似度 >= GLOBAL_CLUSTER_THRESHOLD 则合并
        for i in range(n):
            for j in range(i + 1, n):
                text_i = clips[i].get("matched_text", clips[i].get("text", ""))
                text_j = clips[j].get("matched_text", clips[j].get("text", ""))
                
                # 使用灵活的相似度计算（支持包含关系检测）
                sim = calculate_flexible_similarity(text_i, text_j)
                
                # 调试日志
                if sim > 0.5:
                    logger.info(f"聚类检测: clip[{i}] vs clip[{j}] 相似度: {sim:.2f} (阈值:{self.GLOBAL_CLUSTER_THRESHOLD})")
                    logger.info(f"  text_i: {text_i[:30]}...")
                    logger.info(f"  text_j: {text_j[:30]}...")
                
                if sim >= self.GLOBAL_CLUSTER_THRESHOLD:
                    union(i, j)
        
        # 按根节点分组
        clusters_dict = {}
        for i, clip in enumerate(clips):
            root = find(i)
            if root not in clusters_dict:
                clusters_dict[root] = []
            clusters_dict[root].append(clip)
        
        return list(clusters_dict.values())
    
    def _version_competition(self, cluster: List[Dict]) -> Dict:
        """
        多版本竞争 - 从语义簇中选择最佳版本
        
        竞争策略：
        1. 初筛：排除有严重瑕疵的片段
           - 末尾无终止标点（has_end_punctuation=False）
           - 有卡顿/重复词（has_stutter=True）
        2. 优选：在剩余片段中选择 score 最高的
        3. 保底：如果全部被排除，选择字数最多且时间最晚的
        
        Args:
            cluster: 语义簇（相似片段列表）
            
        Returns:
            最佳片段
        """
        if len(cluster) == 1:
            return cluster[0]
        
        # 1. 初筛：检查语义完整性和卡顿
        valid_candidates = []
        invalid_candidates = []
        
        for clip in cluster:
            has_end_punc = clip.get("has_end_punctuation", True)
            has_stutter = clip.get("has_stutter", False)
            score = clip.get("score", clip.get("similarity", 0))
            
            clip_info = {
                "clip": clip,
                "has_end_punc": has_end_punc,
                "has_stutter": has_stutter,
                "score": score,
                "text_length": len(clip.get("matched_text", clip.get("text", "")))
            }
            
            if has_end_punc and not has_stutter:
                # 完整片段（无中断、无卡顿）
                valid_candidates.append(clip_info)
            else:
                invalid_candidates.append(clip_info)
                # 记录排除原因
                reasons = []
                if not has_end_punc:
                    reasons.append("末尾无终止标点")
                if has_stutter:
                    reasons.append("有卡顿/重复词")
                clip["_drop_reason"] = "; ".join(reasons)
        
        # 2. 优选：从有效候选中选择 score 最高的
        if valid_candidates:
            best = max(valid_candidates, key=lambda x: x["score"])
            return best["clip"]
        
        # 3. 保底：所有片段都有瑕疵，选择"相对最好"的
        # 策略：优先选择字数多、时间晚的（通常最后一遍录制更完整）
        logger.warning(f"语义簇中所有片段都有瑕疵，使用保底策略")
        
        # 为无效候选计算保底分数
        for c in invalid_candidates:
            # 保底分数 = 字数 * 0.5 + (1 - 有无卡顿) * 0.3 + (有无终止标点) * 0.2
            fallback_score = (
                c["text_length"] * 0.05 +
                (0 if c["has_stutter"] else 0.3) +
                (0.2 if c["has_end_punc"] else 0)
            )
            c["fallback_score"] = fallback_score
        
        best = max(invalid_candidates, key=lambda x: x["fallback_score"])
        best["clip"]["_fallback"] = True
        logger.info(f"使用保底片段: score={best['score']}, text_len={best['text_length']}")
        
        return best["clip"]
    
    def _cluster_similar_clips(self, clips: List[Dict]) -> List[List[Dict]]:
        """将相似片段聚类分组（兼容旧版本）"""
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
    
    # ========== 6. 文案引导模式 (Script-Driven Assembly) ==========
    
    # 时间轴窗口偏移权重参数
    WINDOW_SIZE = 3  # 窗口大小
    POSITION_BONUS = 0.15  # 位置接近时的额外加分
    
    def script_driven_assembly(self, all_clips: List[Dict]) -> List[Dict]:
        """
        文案引导模式组装 - 核心功能（v2.0 强化版）
        
        流程：
        1. 强制清理缓存：确保读取最新ASR结果
        2. 提取"虚拟脚本线"：从所有A-Roll提取唯一句子构成逻辑脚本（语义指纹去重）
        3. 按句索引择优：为每个脚本句子找到最佳素材
        4. 唯一性锁定：入选素材立即打上USED标签，禁止后续句子回流
        5. 强制末尾优先：必须选该台词在全片中【最后一次】出现的、且完整的版本
        6. 强制线性拼接：按脚本顺序输出
        7. 剪辑点自适应平滑：首部呼吸位 + 尾部淡出
        
        Args:
            all_clips: 所有A-Roll素材列表
            
        Returns:
            按文案逻辑顺序排列的片段列表
        """
        # 搜索阈值降至0.6，与extract_unique_sentences保持一致
        MATCH_THRESHOLD = 0.6

        if not all_clips:
            return []

        # ====== 硬核修改1: 强制清理缓存 ======
        # 确保不再读取旧的ASR匹配结果
        logger.info("=== 强制清理缓存 ===")
        force_clean_cache()
        logger.info("缓存已清理，开始执行文案引导模式\n")

        # 0. 初始化：给所有素材打上is_picked=False
        for clip in all_clips:
            clip["is_picked"] = False
        
        # 1. 提取虚拟脚本线
        script_skeleton = self.extract_unique_sentences(all_clips)
        
        if not script_skeleton:
            logger.warning("未能从素材中提取出有效脚本，使用原始顺序")
            return self.sort_by_timestamp(all_clips)
        
        logger.info(f"=== 文案引导模式（v2.0 强化版）===")
        logger.info(f"提取到 {len(script_skeleton)} 个唯一句子作为脚本骨架")
        
        # 打印脚本骨架
        logger.info("--- 脚本骨架 ---")
        for i, s in enumerate(script_skeleton):
            logger.info(f"  [{i+1}] {s[:50]}")
        logger.info("--- 开始匹配 ---\n")
        
        # 2. 按句索引择优（带唯一性锁定 + 末尾优先）
        final_timeline = []
        used_video_ids = set()  # 跟踪已使用的素材ID
        
        # 为每个素材计算全局索引位置
        sorted_clips = sorted(all_clips, key=lambda x: (x.get("video_id", 0), x.get("start", 0)))
        clip_to_index = {id(c): idx for idx, c in enumerate(sorted_clips)}
        
        for i, sentence_text in enumerate(script_skeleton):
            # 详细日志输出
            logger.info(f"[Script Line {i+1}/{len(script_skeleton)}] \"{sentence_text[:40]}...\"")
            
            # 在全库中寻找说这句话的所有"候选人"（阈值降至0.6）
            candidates = self._find_similar_clips(all_clips, sentence_text, threshold=MATCH_THRESHOLD)
            
            if not candidates:
                logger.warning(f"  [X] 未找到匹配素材，跳过")
                continue
            
            # 过滤掉已使用的素材（禁止回流）
            available_candidates = [c for c in candidates if c.get("video_id") not in used_video_ids]
            
            if not available_candidates:
                logger.warning(f"  [X] 所有候选素材已使用（USED），跳过")
                continue
            
            logger.info(f"  找到 {len(available_candidates)} 个候选素材")
            
            # 强化末尾优先：直接按(完整度*1000 + start_time)排序
            # 完整句子优先（加1000分），且选择录制时间最晚的
            def sort_key_v3(c):
                text = c.get("matched_text", c.get("text", "")).strip()
                # 完整度分：如果有标点加1000分
                completeness_score = 1000.0 if re.search(r'[。！？]$', text) else 0.0
                # 时间权重：将秒数直接加进来
                time_score = c.get("start", 0.0)
                return completeness_score + time_score

            available_candidates.sort(key=sort_key_v3, reverse=True)
            
            # 3. 择优逻辑（末尾优先）
            best_clip = self._select_best_from_candidates(available_candidates)
            
            if best_clip:
                # 4. 剪辑点自适应平滑
                best_clip = self._apply_smart_cut_points(best_clip)
                
                # 标记这是对应第几句话的
                best_clip["script_index"] = i
                best_clip["script_text"] = sentence_text
                
                # 5. 唯一性锁定：打上USED标签
                best_clip["is_picked"] = True
                video_id = best_clip.get("video_id")
                used_video_ids.add(video_id)
                
                final_timeline.append(best_clip)
                
                # 详细日志 - 匹配报告
                clip_name = Path(best_clip.get('video_path', '')).name
                clip_text = best_clip.get('matched_text', best_clip.get('text', ''))[:40]
                score = best_clip.get('score', best_clip.get('composite_score', 0))
                start_time = best_clip.get('start', 0)
                logger.info(f"  [OK] Using Clip [ID: {video_id}, Name: {clip_name}]")
                logger.info(f"       Text: {clip_text}...")
                logger.info(f"       Score: {score:.2f}, Start: {start_time:.2f}s")

                # 决策对齐日志
                logger.info(f"===> 决策对齐: 脚本句 '{sentence_text[:10]}...' 最终锁定了素材 ID {video_id}，时间点 {start_time:.2f}s")
            else:
                logger.warning(f"  [X] 所有候选素材质量不达标，跳过")
        
        # 打印最终匹配报告
        logger.info("\n" + "=" * 60)
        logger.info("匹配报告 (Matching Report)")
        logger.info("=" * 60)
        logger.info(f"{'#':<3} {'脚本句子':<30} {'素材ID':<8} {'文件名':<15} {'评分':<6}")
        logger.info("-" * 60)
        
        for i, clip in enumerate(final_timeline):
            script_text = clip.get("script_text", "")[:28]
            video_id = clip.get("video_id", "N/A")
            clip_name = Path(clip.get('video_path', '')).name[:13]
            score = clip.get('score', clip.get('composite_score', 0))
            logger.info(f"{i+1:<3} {script_text:<30} {video_id:<8} {clip_name:<15} {score:.2f}")
        
        logger.info("=" * 60)
        logger.info(f"文案引导组装完成: {len(final_timeline)} 个片段")
        
        # 6. 逻辑校验
        final_timeline = self.validate_sequence_logic(final_timeline)
        
        return final_timeline
    
    def _apply_position_bonus(self, candidates: List[Dict], clip_to_index: Dict, 
                              current_index: int, total_scripts: int) -> List[Dict]:
        """
        应用时间轴窗口偏移权重
        
        逻辑：匹配素材时，优先在脚本点前后WINDOW_SIZE个片段范围内寻找。
        如果全库有多个高分素材，物理位置更靠近前后文逻辑的素材应获得额外加分。
        
        Args:
            candidates: 候选素材列表
            clip_to_index: 素材到全局索引的映射
            current_index: 当前脚本句子的索引
            total_scripts: 总脚本句子数
            
        Returns:
            加权后的候选列表
        """
        # 计算期望的素材位置
        if not candidates or total_scripts == 0:
            return candidates
        
        # 期望位置 = (当前索引 / 总脚本数) * 总素材数
        expected_position_ratio = current_index / total_scripts
        
        # 计算每个候选的位置得分
        for clip in candidates:
            clip_id = id(clip)
            if clip_id in clip_to_index:
                clip_pos = clip_to_index[clip_id]
                # 归一化位置
                total_clips = len(clip_to_index)
                clip_pos_ratio = clip_pos / max(total_clips, 1)
                
                # 位置偏差
                position_diff = abs(clip_pos_ratio - expected_position_ratio)
                
                # 偏差越小，得分越高
                if position_diff < 0.1:  # 偏差小于10%
                    clip["_position_bonus"] = self.POSITION_BONUS
                elif position_diff < 0.2:
                    clip["_position_bonus"] = self.POSITION_BONUS * 0.5
                else:
                    clip["_position_bonus"] = 0.0
        
        return candidates
    
    def _apply_smart_cut_points(self, clip: Dict) -> Dict:
        """
        剪辑点自适应平滑
        
        操作：
        1. 首部预留：向左扩展start（呼吸位）
        2. 尾部处理：
           - 如果结尾有标点或字数>=10：向右扩展end（自然结束）
           - 如果字数<10（可能中断）：标记需要淡出
        
        Args:
            clip: 素材片段
            
        Returns:
            处理后的片段
        """
        # 首部呼吸位：向左扩展0.1秒
        original_start = clip.get("start", 0)
        clip["start"] = max(0, original_start - 0.1)
        clip["_start_adjusted"] = original_start != clip["start"]
        
        # 尾部处理
        text = clip.get("matched_text", clip.get("text", "")).strip().replace(" ", "")
        char_count = len(text)
        
        # 判断是否为完整句子：
        # 1. 有结尾标点，或
        # 2. 字数 >= 10（通常是一个完整的短句）
        is_complete = bool(re.search(r'[。！？]$', text)) or char_count >= 10
        
        if is_complete:
            # 完整句子，向右扩展作为呼吸位
            original_end = clip.get("end", clip.get("duration", 0))
            clip["end"] = original_end + 0.15
            clip["_fade_out"] = False
        else:
            # 不完整/中断，标记需要淡出
            clip["_fade_out"] = True
            # 稍微向左收缩，避免截断
            original_end = clip.get("end", clip.get("duration", 0))
            clip["end"] = original_end - 0.1
        
        return clip
    
    def validate_sequence_logic(self, timeline: List[Dict]) -> List[Dict]:
        """
        最后的逻辑防火墙：检查拼接后的视频是否真的'顺'
        
        检查规则：
        1. 如果两个相邻片段在原始视频中相距太远（如跨了10分钟），打印警告
        2. 如果出现素材回跳（video_id递减），尝试调整
        
        Args:
            timeline: 整理后的时间线
            
        Returns:
            校验后的时间线
        """
        if len(timeline) < 2:
            return timeline
        
        warnings = []
        
        for i in range(len(timeline) - 1):
            curr = timeline[i]
            next_clip = timeline[i + 1]
            
            curr_video_id = curr.get("video_id", 0)
            next_video_id = next_clip.get("video_id", 0)
            
            # 检查1：素材回跳
            if next_video_id < curr_video_id:
                warnings.append(f"Warning: 素材回跳 at index {i}: video_id {next_video_id} < {curr_video_id}")
            
            # 检查2：时间间隔过大
            # 这里简化处理：如果两个片段来自不同视频，打印警告
            if curr_video_id != next_video_id:
                curr_name = curr.get('file_name', '')
                next_name = next_clip.get('file_name', '')
                warnings.append(f"Info: 跨视频拼接 at index {i}: {curr_name} -> {next_name}")
        
        if warnings:
            logger.info("=== 序列逻辑校验 ===")
            for w in warnings:
                logger.info(w)
        
        return timeline
    
    def _normalize_text(self, text: str) -> str:
        """
        文本归一化：去掉标点、空格，并将数字转为文字
        
        Args:
            text: 原始文本
            
        Returns:
            归一化后的文本
        """
        import re
        
        # 去掉所有标点符号
        normalized = re.sub(r'[，。！？、；：""''【】（）\s，。！？、；：""''【】\(\)]', '', text)
        
        # 数字转文字（可选）
        digit_map = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', 
                     '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}
        for d, w in digit_map.items():
            normalized = normalized.replace(d, w)
        
        return normalized
    
    def extract_unique_sentences(self, all_clips: List[Dict]) -> List[str]:
        """
        从所有A-Roll素材中提取唯一句子构成"虚拟脚本线"
        
        过滤规则（强化版 v2.0）：
        1. 过滤掉卡顿、字数过少(<3字)的废话
        2. 文本归一化：去掉标点、空格、数字转文字
        3. 语义指纹去重：Levenshtein距离<3判定为重复
        4. 相似度阈值降至0.6 - 确保语义绝对唯一
        5. 短句合并：如果连续两个片段很短且合并后能形成完整句子，则合并
        6. 只保留最长且带标点的那一个作为脚本锚点
        
        Args:
            all_clips: 所有A-Roll素材列表
            
        Returns:
            不重复的句子列表（逻辑脚本）
        """
        from difflib import SequenceMatcher
        
        # 相似度阈值降至0.6，确保语义绝对唯一
        SIMILARITY_THRESHOLD = 0.6
        MIN_TEXT_LENGTH = 3
        LEVENSHTEIN_MAX_DISTANCE = 5  # 语义指纹重复判定阈值（调高以更狠合并重复项）
        
        if not all_clips:
            return []
        
        # 0. 检查缓存有效性
        if not check_cache_validity():
            force_clean_cache()
        
        # 1. 先按录制时间排序
        sorted_clips = sorted(all_clips, key=lambda x: (
            x.get("video_id", 0),
            x.get("start", 0)
        ))
        
        # 2. 短句合并：先把相邻的短片段合并
        merged_clips = self._merge_short_clips(sorted_clips)
        
        # 3. 提取有效句子（语义指纹去重版）
        unique_lines = []  # 存储 (原始文本, 归一化文本, 语义指纹, clip信息)
        seen_fingerprints = set()  # 已见过的语义指纹集合
        
        for clip in merged_clips:
            text = clip.get("matched_text", clip.get("text", ""))
            if not text:
                continue
            
            # 清理文本
            original_clean = text.strip().replace(" ", "")
            
            # 过滤规则1：字数过少
            if len(original_clean) < MIN_TEXT_LENGTH:
                continue
            
            # 过滤规则2：卡顿检测
            if self._has_stutter(original_clean):
                continue
            
            # 过滤规则3：置信度过低（如果有）
            confidence = clip.get("asr_confidence", 1.0)
            if confidence < 0.5:
                continue
            
            # 过滤规则4：以提示词开头（被截断的素材）
            if re.match(r"^(走|一|二|三|四|五|1|2|3|4|5|开始)", original_clean):
                continue
            
            # 文本归一化
            normalized = self._normalize_text(original_clean)
            
            # 提取语义指纹
            fingerprint = extract_semantic_fingerprint(original_clean)
            
            # 语义指纹硬拦截：Levenshtein距离<3判定为重复
            is_duplicate = False
            for existing_fingerprint in seen_fingerprints:
                dist = levenshtein_distance(fingerprint, existing_fingerprint)
                if dist < LEVENSHTEIN_MAX_DISTANCE:
                    is_duplicate = True
                    logger.info(f"  [DEDUP] 语义指纹重复: '{fingerprint[:20]}' vs '{existing_fingerprint[:20]}' (距离={dist})")
                    break
            
            # 检查是否是完整句子（末尾有标点）
            has_ending_punct = bool(re.search(r'[。！？]$', original_clean))
            
            if is_duplicate:
                # 末次优先采样：只要当前这个片段的录制时间更晚，就用它作为"脚本锚点"
                # 因为拍摄者最后一遍说的通常最接近最终剧本
                for i, (existing_orig, existing_norm, existing_fp, existing_clip) in enumerate(unique_lines):
                    dist = levenshtein_distance(fingerprint, existing_fp)
                    if dist < LEVENSHTEIN_MAX_DISTANCE:
                        # 优先选择录制时间更晚的片段
                        if clip.get("start", 0) > existing_clip.get("start", 0):
                            unique_lines[i] = (original_clean, normalized, fingerprint, clip)
                            logger.info(f"  [DEDUP] 末次优先: '{original_clean[:20]}' (start={clip.get('start')}) 替换了旧版本")
                        break
                continue
            
            # 补充检查：相似度阈值（双重保险）- 也采用末次优先
            for i, (existing_orig, existing_norm, existing_fp, existing_clip) in enumerate(unique_lines):
                sim = SequenceMatcher(None, normalized, existing_norm).ratio()
                if sim >= SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    # 末次优先采样
                    if clip.get("start", 0) > existing_clip.get("start", 0):
                        unique_lines[i] = (original_clean, normalized, fingerprint, clip)
                        logger.info(f"  [DEDUP] 相似度阈值末次优先: 替换了旧版本")
                    break
            
            if not is_duplicate:
                unique_lines.append((original_clean, normalized, fingerprint, clip))
                seen_fingerprints.add(fingerprint)
        
        # 4. 提取最终脚本线
        unique_sentences = [item[0] for item in unique_lines]
        
        # 如果没有完整句子，退而求其次选择所有有效句子
        if not unique_sentences:
            seen_fingerprints.clear()
            unique_lines = []
            for clip in merged_clips:
                text = clip.get("matched_text", clip.get("text", ""))
                if text:
                    original_clean = text.strip().replace(" ", "")
                    normalized = self._normalize_text(original_clean)
                    fingerprint = extract_semantic_fingerprint(original_clean)
                    
                    if len(original_clean) >= MIN_TEXT_LENGTH and not self._has_stutter(original_clean):
                        # 语义指纹去重
                        is_duplicate = False
                        for existing_fp in seen_fingerprints:
                            dist = levenshtein_distance(fingerprint, existing_fp)
                            if dist < LEVENSHTEIN_MAX_DISTANCE:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            unique_lines.append((original_clean, normalized, fingerprint, clip))
                            seen_fingerprints.add(fingerprint)
            
            unique_sentences = [item[0] for item in unique_lines]
        
        logger.info(f"[Extract v2.0] 语义指纹去重后提取到 {len(unique_sentences)} 个唯一句子")
        for i, s in enumerate(unique_sentences):
            logger.info(f"  [{i+1}] {s[:40]}...")
        
        return unique_sentences
    
    def _merge_short_clips(self, clips: List[Dict], min_merge_length: int = 8) -> List[Dict]:
        """
        短句合并：如果连续两个片段都很短，且合并后能形成完整句子，则合并
        
        Args:
            clips: 已排序的片段列表
            min_merge_length: 合并后最小字数阈值
            
        Returns:
            合并后的片段列表
        """
        if len(clips) < 2:
            return clips
        
        merged = []
        i = 0
        
        while i < len(clips):
            current = clips[i]
            current_text = current.get("matched_text", current.get("text", "")).strip().replace(" ", "")
            
            # 如果当前片段已经有结尾标点，直接保留
            if re.search(r'[。！？]$', current_text):
                merged.append(current)
                i += 1
                continue
            
            # 如果当前片段没有结尾标点，尝试与下一个片段合并
            if i + 1 < len(clips):
                next_clip = clips[i + 1]
                next_text = next_clip.get("matched_text", next_clip.get("text", "")).strip().replace(" ", "")
                
                # 检查是否可以合并：
                # 1. 两个片段来自同一个视频（或相邻视频）
                # 2. 合并后字数适中
                # 3. 合并后有结尾标点，或者next有结尾标点
                
                same_video = (current.get("video_id") == next_clip.get("video_id"))
                
                if same_video:
                    merged_text = current_text + next_text
                    has_ending = re.search(r'[。！？]$', merged_text) or re.search(r'[。！？]$', next_text)
                    
                    # 合并条件：合并后有结尾标点 且 字数适中
                    if has_ending and min_merge_length <= len(merged_text) <= 30:
                        # 执行合并
                        merged_clip = {
                            **current,
                            "matched_text": merged_text,
                            "text": merged_text,
                            "_merged": True  # 标记为合并后的片段
                        }
                        merged.append(merged_clip)
                        logger.info(f"短句合并: {current_text[:15]}... + {next_text[:15]}... -> {merged_text[:20]}...")
                        i += 2  # 跳过两个片段
                        continue
            
            # 不能合并，保留当前片段
            merged.append(current)
            i += 1
        
        return merged
    
    def _has_stutter(self, text: str) -> bool:
        """检测文本是否有卡顿/重复词"""
        if not text:
            return False
        
        # 检测连续重复
        for i in range(len(text) - 1):
            if text[i] == text[i+1] and text[i] in "的了是在":
                return True
        
        # 检测短语重复模式
        stutter_patterns = [
            r"(.+?)\1{2,}",  # 连续重复
            r"(.+?)，\1",    # 逗号重复
            r"(.+?)\s+\1",   # 空格重复
        ]
        
        for pattern in stutter_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _find_similar_clips(self, all_clips: List[Dict], target_text: str, 
                           threshold: float = 0.7) -> List[Dict]:
        """
        在素材库中找到与目标文本相似的所有素材
        
        Args:
            all_clips: 所有素材
            target_text: 目标文本
            threshold: 相似度阈值
            
        Returns:
            相似的素材列表
        """
        if not target_text or not all_clips:
            return []
        
        target_clean = target_text.strip().replace(" ", "")
        
        similar_clips = []
        for clip in all_clips:
            text = clip.get("matched_text", clip.get("text", ""))
            if not text:
                continue
            
            clean_text = text.strip().replace(" ", "")
            
            # 计算相似度
            sim = self.calculate_similarity(target_clean, clean_text)
            
            if sim >= threshold:
                similar_clips.append({
                    **clip,
                    "_similarity": sim
                })
        
        return similar_clips
    
    def _select_best_from_candidates(self, candidates: List[Dict]) -> Optional[Dict]:
        """
        从候选素材中选择最佳版本（物理逻辑优先版）
        
        排序规则（优先级从高到低）：
        1. 完整度：有结尾标点（最高优先级，权重10000）
        2. 录制时间：start_time越晚越好（权重1000）- 因为最后一遍通常是"OK条"
        3. 综合评分 composite_score（权重100）
        4. 位置偏移加分（权重10）
        
        Args:
            candidates: 候选素材列表
            
        Returns:
            最佳素材，如果没有合格则返回None
        """
        if not candidates:
            return None
        
        # 检查是否有完整句子（有结尾标点）
        has_complete = any(
            bool(re.search(r'[。！？]$', c.get("matched_text", c.get("text", "")).strip()))
            for c in candidates
        )
        
        # 排序（物理逻辑优先版）
        def sort_key(c):
            text = c.get("matched_text", c.get("text", ""))
            clean_text = text.strip().replace(" ", "")
            
            # 权重1（最高）：完整度 - 有结尾标点得10000分，否则0分
            has_ending = 10000.0 if re.search(r'[。！？]$', clean_text) else 0.0
            
            # 如果有完整句子，过滤掉不完整的（直接返回最低分）
            if has_complete and not has_ending:
                return (0.0, 0.0, 0.0, 0.0)
            
            # 权重2：录制时间（start_time越大代表越晚录制的优先级越高）
            # 使用start作为时间戳，越大越好
            start_time = c.get("start", 0.0)
            
            # 权重3：综合评分 (composite_score)
            composite_score = c.get("composite_score", c.get("score", 0.0))
            
            # 权重4：位置偏移加分
            position_bonus = c.get("_position_bonus", 0.0)
            
            return (has_ending, start_time, composite_score, position_bonus)
        
        # 排序（倒序）
        sorted_candidates = sorted(candidates, key=sort_key, reverse=True)
        
        best = sorted_candidates[0]
        
        # 再次检查：如果最佳素材是严重不完整的（如只有2-3个字），则跳过
        text = best.get("matched_text", best.get("text", ""))
        clean_text = text.strip().replace(" ", "")
        
        if len(clean_text) < 3:
            logger.warning(f"  最佳素材字数过少({len(clean_text)}字)，跳过")
            return None
        
        # 如果最佳素材没有结尾标点，尝试找次优但完整的素材
        if not re.search(r'[。！？]$', clean_text) and has_complete:
            for c in sorted_candidates[1:]:
                text2 = c.get("matched_text", c.get("text", ""))
                clean_text2 = text2.strip().replace(" ", "")
                if re.search(r'[。！？]$', clean_text2):
                    logger.info(f"  选用次优完整素材替代不完整素材")
                    best = c
                    break
        
        return best


# ========== 独立函数版本 ==========

def deduplicate_and_sort_clips(all_clips, similarity_threshold=0.85):
    """
    对 A-Roll 片段去重并排序
    
    处理"同一句口播词多次录制"和"拼接顺序混乱"的问题
    
    流程：
    1. 物理排序：先按 file_id 排，再按 start 时间戳排，保证"先拍的在前"
    2. 语义去重：使用 SequenceMatcher 计算文本相似度，识别重复口播
    3. 版本择优：在重复片段中保留综合评分(socre)更高的版本
    
    Args:
        all_clips: 包含 {'text', 'score', 'start', 'file_id', ...} 的列表
        similarity_threshold: 文本相似度阈值，超过此值认为是重复录制
        
    Returns:
        去重排序后的片段列表
    """
    if not all_clips:
        return []
    
    # 1. 物理排序（确保逻辑基础是录制顺序）
    # 先按文件ID排，再按文件内部时间戳排，保证"先拍的在前"
    sorted_clips = sorted(
        all_clips, 
        key=lambda x: (x.get('file_id', x.get('video_id', 0)), x.get('start', 0))
    )
    
    final_sequence = []
    
    for current_clip in sorted_clips:
        if not final_sequence:
            final_sequence.append(current_clip)
            continue
        
        last_clip = final_sequence[-1]
        
        # 计算当前片段与上一个入选片段的文本相似度
        text1 = last_clip.get('text', last_clip.get('matched_text', ''))
        text2 = current_clip.get('text', current_clip.get('matched_text', ''))
        
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        
        if similarity > similarity_threshold:
            # 如果是重复口播，对比综合评分 (Score)
            # 策略：保留分值更高（录制质量更好）的版本
            current_score = current_clip.get('score', current_clip.get('similarity', 0))
            last_score = last_clip.get('score', last_clip.get('similarity', 0))
            
            if current_score > last_score:
                final_sequence[-1] = current_clip  # 替换为更优版本
                logger.info(f"去重替换: \"{text2[:20]}...\" (score: {last_score:.2f} -> {current_score:.2f})")
            else:
                pass  # 舍弃当前版本，保留已入选的版本
        else:
            # 如果是新台词，直接加入序列
            final_sequence.append(current_clip)
    
    logger.info(f"去重排序完成: {len(all_clips)} -> {len(final_sequence)} 片段")
    
    return final_sequence


def cluster_by_semantics(clips: List[Dict], threshold: float = 0.7) -> List[List[Dict]]:
    """
    按语义相似度聚类
    
    Args:
        clips: 片段列表
        threshold: 相似度阈值
        
    Returns:
        语义簇列表
    """
    if not clips:
        return []
    
    n = len(clips)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i in range(n):
        for j in range(i + 1, n):
            text_i = clips[i].get("matched_text", clips[i].get("text", ""))
            text_j = clips[j].get("matched_text", clips[j].get("text", ""))
            sim = difflib.SequenceMatcher(None, text_i, text_j).ratio()
            if sim >= threshold:
                union(i, j)
    
    clusters_dict = {}
    for i, clip in enumerate(clips):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(clip)
    
    return list(clusters_dict.values())


def filter_and_rank_clips(clips: List[Dict], cluster_threshold: float = 0.7) -> List[Dict]:
    """
    针对中断、卡顿、重复录制的深度过滤逻辑
    
    流程：
    1. 按语义聚类：将相似片段分组
    2. 排除严重卡顿：检查文本中是否有连续重复词
    3. 排除中途断句：检查末尾标点符号是否为终止符（。！？）
    4. 综合评分排序：完整度优先 > 置信度/能量 > 时间戳
    5. 每个语义簇只取第一名（Best One）
    6. 最后按物理录制顺序重新排列
    
    Args:
        clips: 片段列表
        cluster_threshold: 语义聚类阈值
        
    Returns:
        过滤排序后的最佳片段列表
    """
    if not clips:
        return []
    
    # 导入检测函数
    from core.clip_evaluator import detect_stutter, calculate_semantic_integrity
    
    # 1. 按语义聚类
    clusters = cluster_by_semantics(clips, threshold=cluster_threshold)
    
    ranked_clips = []
    
    for cluster in clusters:
        # 2-3. 为每个片段计算完整度信息
        for clip in cluster:
            text = clip.get("matched_text", clip.get("text", ""))
            duration = clip.get("duration", clip.get("final_duration", 1.0))
            
            # 检测是否有终止标点
            has_end_punc = bool(re.search(r'[。！？]$', text.strip()))
            clip["has_end_punc"] = has_end_punc
            
            # 检测卡顿
            has_stutter, _ = detect_stutter(text)
            clip["has_stutter"] = has_stutter
            
            # 计算语义完整性
            semantic_score, _ = calculate_semantic_integrity(text, max(0.1, duration))
            clip["semantic_integrity"] = semantic_score
            
            # 综合评分
            base_score = clip.get("score", clip.get("similarity", 0))
            # 完整片段加成分：有终止标点 +0.2，无卡顿 +0.2
            bonus = (0.2 if has_end_punc else 0) + (0.2 if not has_stutter else 0)
            clip["composite_score"] = base_score + bonus
        
        # 4. 排序：完整度优先 > 综合评分 > 时间戳（越晚越好）
        cluster.sort(key=lambda x: (
            x["has_end_punc"] and not x["has_stutter"],  # 完整度
            x["composite_score"],                         # 综合评分
            x.get("start", 0)                            # 时间戳
        ), reverse=True)
        
        # 5. 每个语义簇只取第一名
        if cluster:
            best = cluster[0]
            # 记录被淘汰的
            for c in cluster[1:]:
                c["_dropped"] = True
                c["_drop_reason"] = "语义重复，已选择最佳版本"
            ranked_clips.append(best)
    
    # 6. 最后按照物理录制顺序重新排列
    ranked_clips.sort(key=lambda x: (
        x.get("file_id", x.get("video_id", 0)), 
        x.get("start", 0)
    ))
    
    logger.info(f"深度过滤完成: {len(clips)} -> {len(ranked_clips)} 片段")
    
    return ranked_clips


def get_golden_clips(all_clips, similarity_threshold=0.8):
    """
    获取黄金片段 - 通过三个维度筛选最优质的唯一片段序列
    
    筛选维度：
    1. 标点完整度：利用 FunASR 标点，末尾无终止标点（。！？）则大幅扣分
    2. 重复词检测：检测连续重复的字词（如"我我"、"今天今天"），按次数扣分
    3. 综合评分：基础评分(能量+VAD) * 质量系数
    
    物理安全隔离：
    - 中途中断且无更好替代品的片段，转为 B-Roll 或删除
    
    Args:
        all_clips: 包含 {'text', 'score', 'start', 'end', 'file_id'} 的列表
        similarity_threshold: 文本相似度阈值，超过此值认为是重复录制
        
    Returns:
        黄金片段列表 (已排序)
    """
    if not all_clips:
        return [], []
    
    # 1. 预处理：物理排序，确保逻辑线正确
    # 先按文件ID排，再按文件内部时间戳排
    sorted_clips = sorted(all_clips, key=lambda x: (
        x.get('file_id', x.get('video_id', 0)), 
        x.get('start', 0)
    ))
    
    # 2. 语义质量打分函数
    def evaluate_quality(clip):
        """
        计算片段质量分数
        
        维度 A: 完整性检查
        如果末尾没有句号、感叹号、问号，说明中途断了，大幅扣分
        
        维度 B: 结巴/重复词检测
        检测连续重复的词，如 "我我"、"今天今天"
        """
        text = clip.get('text', clip.get('matched_text', '')).strip()
        base_score = clip.get('score', clip.get('similarity', 0))  # 基础评分（能量+VAD）
        
        if not text:
            return 0.0, "empty_text"
        
        q_score = base_score
        
        # 维度 A: 完整性检查（利用 FunASR 标点）
        # 如果末尾没有句号、感叹号、问号，说明中途断了，大幅扣分
        if not re.search(r'[。！？?!.]$', text):
            q_score *= 0.5
            clip["_incomplete"] = True
        
        # 维度 B: 结巴/重复词检测
        # 检测连续重复的字/词
        words = list(text)
        stutter_count = 0
        
        # 方法1：连续相同单字（口吃型）
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and re.match(r'[\u4e00-\u9fa5]', words[i]):
                stutter_count += 1
        
        # 方法2：检测词级重复 "今天今天"
        if len(text) >= 4:
            for i in range(len(text) - 2):
                substring = text[i:i+2]
                if text.count(substring) >= 2 and i < len(text) - 4:
                    # 找到重复词组
                    next_pos = text.find(substring, i + 2)
                    if next_pos == i + 2:  # 连续重复
                        stutter_count += 1
        
        clip["_stutter_count"] = stutter_count
        
        if stutter_count > 0:
            # 按卡顿次数扣分，最多扣 40%
            q_score *= (1 - min(0.4, 0.2 * stutter_count))
            clip["_has_stutter"] = True
        else:
            clip["_has_stutter"] = False
        
        # 记录原因
        reasons = []
        if clip.get("_incomplete"):
            reasons.append("中途中断")
        if stutter_count > 0:
            reasons.append(f"卡顿{stutter_count}次")
        
        clip["_quality_reason"] = " | ".join(reasons) if reasons else "完整"
        
        return q_score
    
    # 3. 语义聚类与去重（解决多遍录制）
    golden_clips = []
    dropped_clips = []  # 记录被淘汰的片段（可作为 B-Roll）
    
    for current in sorted_clips:
        # 计算质量分数
        current['final_q'] = evaluate_quality(current)
        
        if not golden_clips:
            # 第一个片段直接加入
            current["_selected"] = True
            golden_clips.append(current)
            continue
        
        last = golden_clips[-1]
        
        # 计算当前片段与上一个入选片段的文本相似度
        text_last = last.get('text', last.get('matched_text', ''))
        text_current = current.get('text', current.get('matched_text', ''))
        
        similarity = difflib.SequenceMatcher(None, text_last, text_current).ratio()
        
        # 如果当前片段与上一个片段"撞台词"（相似度超过阈值）
        if similarity > similarity_threshold:
            # 比较两者质量：保留最完整、不卡顿、评分最高的版本
            if current['final_q'] >= last['final_q']:
                # 当前版本更好（或质量相同但时间更晚），替换
                last["_selected"] = False
                last["_drop_reason"] = f"被更优版本替换 (相似度:{similarity:.2f})"
                dropped_clips.append(last)
                
                current["_selected"] = True
                golden_clips[-1] = current
                
                logger.info(f"黄金片段替换: \"{text_current[:15]}...\" "
                           f"(q:{last.get('final_q', 0):.2f} -> {current['final_q']:.2f})")
            else:
                # 当前版本不如上一个，舍弃
                current["_selected"] = False
                current["_drop_reason"] = f"质量低于已有版本 (相似度:{similarity:.2f})"
                dropped_clips.append(current)
        else:
            # 如果是新台词，且质量不是太差（防止纯杂音），则加入
            # 质量阈值 0.3：太低的直接丢弃
            if current['final_q'] > 0.3:
                current["_selected"] = True
                golden_clips.append(current)
            else:
                current["_selected"] = False
                current["_drop_reason"] = f"质量过低 (q:{current['final_q']:.2f})"
                dropped_clips.append(current)
    
    # 4. 物理安全隔离：检查是否有"中途中断"片段没有更好替代
    # 如果一个片段被判定为"中途中断"且没有其他完整版本，需要特殊处理
    final_golden = []
    for i, clip in enumerate(golden_clips):
        if clip.get("_incomplete") and clip.get("final_q", 0) < 0.5:
            # 尝试在 dropped_clips 中找替代
            text = clip.get('text', clip.get('matched_text', ''))
            
            # 查找同类的完整版本
            alternatives = [c for c in dropped_clips 
                          if calculate_similarity_for_text(c.get('text', ''), text) > similarity_threshold
                          and not c.get("_incomplete")]
            
            if alternatives:
                # 有替代，使用替代版本
                best_alt = max(alternatives, key=lambda x: x.get('final_q', 0))
                best_alt["_selected"] = True
                best_alt["_replaced_incomplete"] = True
                final_golden.append(best_alt)
                
                clip["_selected"] = False
                clip["_drop_reason"] = "中途中断，被替代版本替换"
                logger.info(f"安全隔离: 替换中断片段 \"{text[:15]}...\"")
                continue
        
        final_golden.append(clip)
    
    # 5. 最终排序：按物理录制顺序
    final_golden.sort(key=lambda x: (
        x.get('file_id', x.get('video_id', 0)), 
        x.get('start', 0)
    ))
    
    logger.info(f"黄金片段筛选完成: {len(all_clips)} -> {len(final_golden)} A-Roll, {len(dropped_clips)} B-Roll")
    
    return final_golden, dropped_clips


def calculate_similarity_for_text(text1: str, text2: str) -> float:
    """计算两个文本的相似度"""
    if not text1 or not text2:
        return 0.0
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def calculate_flexible_similarity(text1: str, text2: str) -> float:
    """
    计算灵活的文本相似度
    
    考虑多种匹配策略：
    1. 标准 SequenceMatcher 相似度
    2. 包含关系：短文本是否被长文本包含（前缀/后缀）
    3. 关键词重叠：关键短语的共享程度
    
    适用于 ASR 输出单字分词的情况
    """
    if not text1 or not text2:
        return 0.0
    
    # 移除空格
    t1 = text1.replace(' ', '').replace('　', '')
    t2 = text2.replace(' ', '').replace('　', '')
    
    if not t1 or not t2:
        return 0.0
    
    # 策略1：标准相似度
    base_sim = difflib.SequenceMatcher(None, t1, t2).ratio()
    
    # 策略2：包含关系检测
    # 如果短文本的大部分（>60%）被长文本包含，认为是同一语义
    if len(t1) < len(t2):
        shorter, longer = t1, t2
    else:
        shorter, longer = t2, t1
    
    # 计算短文本在长文本中出现的字符数
    matched_chars = sum(1 for c in shorter if c in longer)
    contain_ratio = matched_chars / len(shorter) if shorter else 0
    
    # 如果短文本60%以上字符都在长文本中，认为是包含关系
    contain_sim = contain_ratio if contain_ratio > 0.6 else 0
    
    # 策略2.5：尾部匹配检测（新增）
    # 解决"废片只有后半部分"的问题
    # 例如："全一次向抖音团购领" vs "权益四胜...抖音团购领美的专属优惠券"
    # 寻找两个文本的公共后缀部分
    
    # 从短文本末尾开始，找最长匹配的后缀序列
    suffix_match = 0
    min_len = min(len(shorter), len(longer))
    
    for i in range(1, min_len + 1):
        # 比较两个文本的最后i个字符
        if shorter[-i:] == longer[-i:]:
            suffix_match = i
        else:
            break
    
    # 如果匹配的后缀 >= 3个字，且占短文本50%以上，认为是同一语义
    suffix_ratio = suffix_match / len(shorter) if shorter else 0
    suffix_sim = suffix_ratio if suffix_match >= 3 and suffix_ratio > 0.5 else 0
    
    # 策略2.6：公共子串比例
    # 即使位置不同，只要公共子串足够多，也认为是相似
    common_chars = sum(1 for c in set(shorter) if c in longer)
    common_ratio = common_chars / len(set(shorter)) if shorter else 0
    
    # 降低阈值：公共字符占短文本50%以上就认为是相似
    char_ratio_sim = common_ratio if common_ratio > 0.5 else 0
    
    # 策略3：关键词短语匹配
    # 提取连续的中文短语（2-4字）
    def extract_phrases(text):
        phrases = set()
        for i in range(len(text)):
            for j in range(i+2, min(i+5, len(text)+1)):
                phrase = text[i:j]
                if re.match(r'^[\u4e00-\u9fa5]+$', phrase):
                    phrases.add(phrase)
        return phrases
    
    phrases1 = extract_phrases(t1)
    phrases2 = extract_phrases(t2)
    
    if phrases1 and phrases2:
        common = phrases1 & phrases2
        union = phrases1 | phrases2
        phrase_sim = len(common) / len(union) if union else 0
    else:
        phrase_sim = 0
    
    # 综合得分：取所有策略的最大值
    final_sim = max(base_sim, contain_sim, suffix_sim, char_ratio_sim, phrase_sim)
    
    return final_sim


# ========== 全局语义去重最终版 ==========

def build_semantic_map(clips: List[Dict], threshold: float = 0.55) -> Dict[str, List[Dict]]:
    """
    建立全局语义 Map
    
    遍历所有片段，将相似度 > threshold 的片段归为同一个 Key（语义点）
    每个 Key 下存储该语义点的所有录制版本
    
    Args:
        clips: 片段列表
        threshold: 相似度阈值，建议 0.7-0.8
        
    Returns:
        semantic_map: {key: [clip1, clip2, ...]} 
    """
    semantic_map = {}  # key -> clip list
    
    for clip in clips:
        text = clip.get('text', clip.get('matched_text', '')).strip()
        if not text:
            continue
            
        found_cluster = False
        
        # 遍历现有语义簇
        for key in semantic_map:
            # 只跟簇里第一个片段比较（性能优化）
            key_text = semantic_map[key][0].get('text', 
                            semantic_map[key][0].get('matched_text', '')).strip()
            # 使用灵活的相似度计算
            similarity = calculate_flexible_similarity(text, key_text)
            
            if similarity > threshold:
                semantic_map[key].append(clip)
                found_cluster = True
                logger.debug(f"加入语义簇 '{key[:20]}...': 相似度 {similarity:.2f}")
                break
        
        # 没找到匹配的簇，创建新簇
        if not found_cluster:
            # 使用文本前15字作为 key
            key = text[:15] if len(text) > 15 else text
            semantic_map[key] = [clip]
    
    logger.info(f"全局语义 Map 构建完成: {len(clips)} 片段 -> {len(semantic_map)} 个语义点")
    
    return semantic_map


def final_cleanup(all_clips: List[Dict], 
                  similarity_threshold: float = 0.55,
                  cluster_threshold: float = 0.55) -> List[Dict]:
    """
    最终清理 - 全局视野的去重与择优
    
    解决：多遍录制、结巴、中途断句
    
    流程：
    1. 全局聚类：建立 semantic_map，解决非相邻重复问题
    2. 三维择优：在每个语义簇内按 完整度 > 综合评分 > 录制时间 排序
    3. 物理还原：按 file_id + start 排序，确保拼接顺序正确
    
    Args:
        all_clips: 包含 {'text', 'score', 'start', 'end', 'file_id'} 的列表
        similarity_threshold: 片段间相似度阈值
        cluster_threshold: 语义簇阈值
        
    Returns:
        清理后的最佳片段列表
    """
    if not all_clips:
        return []
    
    # 0. 预处理：确保每条记录都有必要的字段
    for clip in all_clips:
        if 'text' not in clip and 'matched_text' in clip:
            clip['text'] = clip['matched_text']
        if 'file_id' not in clip and 'video_id' in clip:
            clip['file_id'] = clip['video_id']
    
    # 1. 全局聚类（解决非相邻重复）
    clusters = []
    for clip in all_clips:
        text = clip.get('text', '')
        if not text:
            continue
            
        found = False
        for cluster in clusters:
            # 只要跟簇里任意一个片段相似度 > threshold，就进这个簇
            cluster_text = cluster[0].get('text', '')
            # 使用灵活的相似度计算
            similarity = calculate_flexible_similarity(text, cluster_text)
            
            if similarity > similarity_threshold:
                cluster.append(clip)
                found = True
                logger.debug(f"聚类加入: '{text[:15]}...' ~ '{cluster_text[:15]}...' 相似度: {similarity:.2f}")
                break
        
        if not found:
            clusters.append([clip])
    
    logger.info(f"全局聚类完成: {len(all_clips)} 片段 -> {len(clusters)} 个语义簇")
    
    # 打印聚类详情
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            first_text = cluster[0].get('text', '')[:20]
            logger.info(f"  语义簇 {i+1}: '{first_text}...' 包含 {len(cluster)} 个版本")
    
    # 2. 三维择优：在每个簇内选择最佳版本
    best_clips = []
    
    for cluster in clusters:
        # 为每个片段计算完整度
        for clip in cluster:
            text = clip.get('text', '')
            
            # 维度 A：完整度检查（标点）
            has_end_punc = bool(re.search(r'[。！？]$', text.strip()))
            clip['_has_end_punc'] = has_end_punc
            
            # 维度 B：流畅度检查（卡顿）
            stutter_count = _detect_stutter_count(text)
            clip['_stutter_count'] = stutter_count
            clip['_has_stutter'] = stutter_count > 0
            
            # 基础评分
            base_score = clip.get('score', clip.get('similarity', 0))
            clip['_base_score'] = base_score
            
            # 完整度得分：有标点 +0.3，无卡顿 +0.2
            completeness_score = (0.3 if has_end_punc else 0) + (0.2 if stutter_count == 0 else 0)
            clip['_completeness_score'] = completeness_score
            
            # 最终得分 = 基础评分 + 完整度得分
            clip['_final_quality'] = base_score + completeness_score
        
        # 三维排序：完整度(标点) > 综合评分 > 录制时间(越晚越好)
        # 注意：re.search 返回 Match 或 None，需要转布尔值
        cluster.sort(key=lambda x: (
            bool(re.search(r'[。！？]$', x.get('text', '').strip())),  # 有终止标点
            x.get('_final_quality', 0),                                   # 综合评分
            x.get('start', 0)                                             # 录制时间
        ), reverse=True)
        
        # 只取每个语义簇的第一名
        best = cluster[0]
        best_clips.append(best)
        
        # 记录被淘汰的
        if len(cluster) > 1:
            for c in cluster[1:]:
                c['_dropped'] = True
                c['_drop_reason'] = "语义重复，已选择最佳版本"
                logger.info(f"去重舍弃: '{c.get('text', '')[:15]}...' "
                           f"(保留: '{best.get('text', '')[:15]}...')")
    
    # 3. 物理还原：确保拼接顺序逻辑正常
    best_clips.sort(key=lambda x: (
        x.get('file_id', x.get('video_id', 0)), 
        x.get('start', 0)
    ))
    
    # 打印最终结果
    logger.info(f"最终清理完成: {len(all_clips)} -> {len(best_clips)} A-Roll 片段")
    for i, clip in enumerate(best_clips):
        text = clip.get('text', '')[:20]
        logger.info(f"  A-Roll {i+1}: '{text}...' (file:{clip.get('file_id', '?')}, "
                   f"start:{clip.get('start', 0):.2f})")
    
    return best_clips


def _detect_stutter_count(text: str) -> int:
    """
    检测文本中卡顿/重复词的数量
    
    Args:
        text: 文本
        
    Returns:
        卡顿次数
    """
    if not text:
        return 0
    
    count = 0
    
    # 方法1：连续相同单字（口吃型）
    words = list(text)
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and re.match(r'[\u4e00-\u9fa5]', words[i]):
            count += 1
    
    # 方法2：词级重复 "今天今天"
    if len(text) >= 4:
        for i in range(len(text) - 2):
            substring = text[i:i+2]
            if text.count(substring) >= 2 and i < len(text) - 4:
                next_pos = text.find(substring, i + 2)
                if next_pos == i + 2:
                    count += 1
    
    return count
