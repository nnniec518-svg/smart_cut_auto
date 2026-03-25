"""
全局匹配规划器 - SequencePlanner
实现带状态的全局匹配算法，输出 FFmpeg 剪辑列表（EDL）

核心功能：
1. 向量检索：使用 BGE-M3 模型将脚本的每一句转化为向量
2. 候选过滤：主轨仅从 A_ROLL 素材中召回
3. 禁止重复：维护 used_video_ids 集合
4. 单调性约束：同一 video_id 连续使用时，必须满足 start > 上一片段的 end
"""
import os
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import sqlite3
import logging
import json

logger = logging.getLogger("smart_cut")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class EmbeddingModel:
    """向量嵌入模型封装 - 支持 BGE-M3 或降级到 Sentence-Transformer"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        初始化嵌入模型
        
        Args:
            model_name: 模型名称，默认为 BGE-M3
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            # 尝试加载 BGE-M3
            from FlagEmbedding import FlagModel
            
            logger.info(f"Loading BGE-M3 model: {self.model_name}")
            self.model = FlagModel(self.model_name, use_fp16=True)
            logger.info("BGE-M3 model loaded successfully")
            
        except ImportError:
            logger.warning("FlagEmbedding not available, falling back to Sentence-Transformer")
            self._load_sentence_transformer()
        except Exception as e:
            logger.warning(f"Failed to load BGE-M3: {e}, falling back to Sentence-Transformer")
            self._load_sentence_transformer()
    
    def _load_sentence_transformer(self):
        """降级到 Sentence-Transformer"""
        try:
            from sentence_transformers import SentenceTransformer
            import os
            
            # 强制使用离线模式
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            model_path = MODELS_DIR / "sentence_transformers" / "models--all-MiniLM-L6-v2"
            if model_path.exists():
                cache_folder = str(MODELS_DIR / "sentence_transformers")
                self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_folder)
            else:
                cache_folder = str(MODELS_DIR / "sentence_transformers")
                self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_folder)
            
            logger.info(f"Fallback model loaded: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            self.model = None
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            向量数组 (n, dim)
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            # BGE-M3 使用 encode 方法
            if hasattr(self.model, 'encode'):
                embeddings = self.model.encode(
                    texts, 
                    batch_size=batch_size,
                    normalize=True  # BGE-M3 需要显式归一化
                )
                return np.array(embeddings)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
        
        # 降级方案
        return self._fallback_encode(texts)
    
    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """降级编码方法"""
        if self.model is None:
            raise RuntimeError("No embedding model available")
        
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)
    
    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        计算两个文本的余弦相似度
        
        Args:
            text_a: 文本A
            text_b: 文本B
            
        Returns:
            相似度分数 (-1 到 1)
        """
        embeddings = self.encode([text_a, text_b])
        
        # 余弦相似度
        vec_a, vec_b = embeddings[0], embeddings[1]
        dot_product = np.dot(vec_a, vec_b)
        
        return float(dot_product)  # 已归一化，直接点积即为余弦相似度


class SequencePlanner:
    """全局匹配规划器"""

    # 惩罚参数
    PENALTY_REPEAT = 0.8           # 非连续重复惩罚
    PENALTY_TIME_REVERSE = 0.5    # 时间倒流惩罚（增强）
    PENALTY_HARD_TIME_REVERSE = 1.0  # 严重时间回溯惩罚（同视频内回跳）
    REWARD_SEQUENCE = 0.2         # 顺序保持奖励
    PENALTY_NUMBER_MISMATCH = 0.4  # 数字不匹配惩罚

    # 阈值
    A_ROLL_THRESHOLD = 0.3       # A_ROLL 最低相似度阈值
    B_ROLL_THRESHOLD = 0.1       # B_ROLL 最低相似度阈值

    # 平局决胜参数
    TIEBREAK_THRESHOLD = 0.05   # 平局判定阈值（分数差距小于此值触发平局决胜）
    LATEST_RECORDING_BONUS = 0.01 # "末次优先"奖励分数（平局决胜时使用）

    # 动态窗口参数
    WINDOW_INITIAL_PERCENT = 0.20  # 初始窗口覆盖前20%
    WINDOW_EXPANSION = 1.5         # 窗口扩展系数
    WINDOW_MIN_SIZE = 10          # 最小窗口大小（句数）

    def __init__(self, db_path: str = "storage/materials.db",
                 embedding_model: Optional[EmbeddingModel] = None):
        """
        初始化 SequencePlanner
        
        Args:
            db_path: SQLite 数据库路径
            embedding_model: 嵌入模型实例
        """
        self.db_path = db_path
        self.model = embedding_model or EmbeddingModel()
        
        # 状态管理
        self.used_video_ids: set = set()  # 已使用的视频ID
        self.last_match: Dict = {         # 上一个匹配结果
            "video_id": None,
            "end_time": 0.0,
            "video_path": None,
            "material_index": 0           # 素材在缓存中的索引
        }

        # 动态窗口状态
        self.window_start_idx = 0         # 窗口起始索引
        self.window_end_idx = 0           # 窗口结束索引
        self.progress_ratio = 0.0         # 当前进度比例

        # 加载素材数据
        self.materials_cache: Dict[str, Dict] = {}
        self.materials_list: List[Dict] = []  # 素材列表（按顺序）
        self._load_materials()
    
    def _load_materials(self):
        """从数据库加载素材信息"""
        if not os.path.exists(self.db_path):
            logger.warning(f"Database not found: {self.db_path}")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT file_path, file_name, track_type, valid_start_offset, 
                       duration, transcript_json
                FROM materials
            """)
            
            for row in cursor.fetchall():
                file_path = row["file_path"]
                
                # 解析 ASR 结果
                segments = []
                asr_text = ""
                transcript = row["transcript_json"]
                if transcript:
                    try:
                        data = json.loads(transcript)
                        asr_text = data.get("text", "")
                        segments = data.get("segments", [])
                    except:
                        pass
                
                self.materials_cache[file_path] = {
                    "video_id": file_path,
                    "path": file_path,
                    "name": row["file_name"],
                    "track_type": row["track_type"],
                    "valid_start_offset": row["valid_start_offset"],
                    "duration": row["duration"],
                    "asr_text": asr_text,
                    "segments": segments
                }
            
            conn.close()
            logger.info(f"Loaded {len(self.materials_cache)} materials from database")
            
        except Exception as e:
            logger.error(f"Failed to load materials: {e}")
    
    def get_a_roll_segments(self) -> List[Dict]:
        """
        获取所有 A_ROLL 素材的片段
        
        Returns:
            A_ROLL 片段列表
        """
        segments = []
        
        for video_id, mat in self.materials_cache.items():
            if mat["track_type"] != "A_ROLL":
                continue
            
            # 如果有 ASR 片段，使用片段级别
            mat_segments = mat.get("segments", [])
            if mat_segments:
                for seg in mat_segments:
                    text = seg.get("text", "")
                    if text and len(text.strip()) > 0:
                        segments.append({
                            "video_id": video_id,
                            "video_path": mat["path"],
                            "video_name": mat["name"],
                            "track_type": "A_ROLL",
                            "text": text,
                            "start": seg.get("start", 0),
                            "end": seg.get("end", 0),
                            "valid_offset": mat.get("valid_start_offset", 0.0)
                        })
            else:
                # 没有片段，使用整个素材
                text = mat.get("asr_text", "")
                if text:
                    segments.append({
                        "video_id": video_id,
                        "video_path": mat["path"],
                        "video_name": mat["name"],
                        "track_type": "A_ROLL",
                        "text": text,
                        "start": 0,
                        "end": mat.get("duration", 0),
                        "valid_offset": mat.get("valid_start_offset", 0.0)
                    })
        
        logger.info(f"Total A_ROLL segments: {len(segments)}")
        return segments

    @staticmethod
    def _extract_numbers(text: str) -> set:
        """
        提取文本中的所有数字（阿拉伯数字和中文数字）

        Args:
            text: 输入文本

        Returns:
            数字集合（规范化为阿拉伯数字）
        """
        numbers = set()

        # 阿拉伯数字
        arabic_numbers = re.findall(r'\d+', text)
        numbers.update(arabic_numbers)

        # 中文数字
        chinese_nums = {
            '零': '0', '一': '1', '二': '2', '两': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
            '壹': '1', '贰': '2', '叁': '3', '肆': '4', '伍': '5', '陆': '6',
            '柒': '7', '捌': '8', '玖': '9', '拾': '10', '佰': '100', '仟': '1000'
        }

        for cn, ar in chinese_nums.items():
            if cn in text:
                numbers.add(ar)

        # 检测"第X"、"权益X"等模式
        patterns = [
            r'第[零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+',
            r'权益[零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+',
            r'[零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+号'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # 转换为数字
                num = match[-1]  # 取最后一个字符
                if num in chinese_nums:
                    numbers.add(chinese_nums[num])

        return numbers

    def _check_number_match(self, text_a: str, text_b: str) -> bool:
        """
        检查两段文本的数字是否匹配

        Args:
            text_a: 文案
            text_b: 素材ASR文本

        Returns:
            数字是否匹配
        """
        nums_a = self._extract_numbers(text_a)
        nums_b = self._extract_numbers(text_b)

        # 如果文案有数字而素材没有，直接不匹配
        if nums_a and not nums_b:
            return False

        # 如果双方都有数字，检查是否有交集
        if nums_a and nums_b:
            # 检查是否有共同数字
            common_nums = nums_a & nums_b
            if not common_nums:
                # 没有共同数字，不匹配
                return False

        return True

    def _update_dynamic_window(self, current_idx: int, total_sentences: int):
        """
        更新动态滑动窗口

        Args:
            current_idx: 当前文案索引
            total_sentences: 总文案数量
        """
        # 计算进度
        self.progress_ratio = current_idx / max(total_sentences, 1)

        # 获取素材总数
        total_materials = len(self.materials_list)

        # 计算窗口大小
        window_size = max(
            self.WINDOW_MIN_SIZE,
            int(total_materials * self.WINDOW_INITIAL_PERCENT * self.WINDOW_EXPANSION)
        )

        # 计算窗口中心位置
        window_center = int(total_materials * self.progress_ratio)

        # 计算窗口边界
        self.window_start_idx = max(0, window_center - window_size // 2)
        self.window_end_idx = min(total_materials, window_center + window_size // 2)

        logger.debug(f"窗口更新: [{self.window_start_idx}, {self.window_end_idx}), "
                    f"中心={window_center}, 进度={self.progress_ratio:.2%}")

    def _filter_by_window(self, candidates: List[Dict]) -> List[Dict]:
        """
        根据动态窗口过滤候选

        Args:
            candidates: 原始候选列表

        Returns:
            过滤后的候选列表
        """
        if not candidates:
            return candidates

        # 获取窗口内的素材索引
        window_indices = set(range(self.window_start_idx, self.window_end_idx))

        # 过滤候选
        filtered = []
        for cand in candidates:
            material_idx = cand.get("material_index", -1)
            if material_idx in window_indices or material_idx == -1:
                # 在窗口内或无索引信息
                filtered.append(cand)
            else:
                # 距离窗口中心的距离
                center = (self.window_start_idx + self.window_end_idx) // 2
                distance = abs(material_idx - center)
                total = len(self.materials_list)

                # 如果距离过远（超过素材总数的30%），严重惩罚
                if distance > total * 0.3:
                    logger.debug(f"  跳过窗口外候选: 索引{material_idx}, 距离中心{distance}")

        # 如果过滤后候选太少，返回原列表
        if len(filtered) < 3:
            logger.debug(f"  窗口内候选过少({len(filtered)})，返回全部候选")
            return candidates

        return filtered

    def get_b_roll_materials(self) -> List[Dict]:
        """
        获取所有 B_ROLL 素材
        
        Returns:
            B_ROLL 素材列表
        """
        materials = []
        
        for video_id, mat in self.materials_cache.items():
            if mat["track_type"] == "B_ROLL":
                materials.append({
                    "video_id": video_id,
                    "video_path": mat["path"],
                    "video_name": mat["name"],
                    "track_type": "B_ROLL",
                    "text": mat.get("asr_text", ""),
                    "start": 0,
                    "end": mat.get("duration", 5.0),
                    "valid_offset": mat.get("valid_start_offset", 0.0)
                })
        
        logger.info(f"Total B_ROLL materials: {len(materials)}")
        return materials
    
    def _compute_similarity_batch(self, 
                                   target_texts: List[str], 
                                   candidate_texts: List[str]) -> np.ndarray:
        """
        批量计算相似度矩阵
        
        Args:
            target_texts: 目标文案列表
            candidate_texts: 候选片段文本列表
            
        Returns:
            相似度矩阵 (len(target_texts), len(candidate_texts))
        """
        if not target_texts or not candidate_texts:
            return np.array([])
        
        # 编码所有文本
        target_embeddings = self.model.encode(target_texts)
        candidate_embeddings = self.model.encode(candidate_texts)
        
        # 计算余弦相似度矩阵
        similarities = np.dot(target_embeddings, candidate_embeddings.T)
        
        return similarities
    
    def _retrieve_candidates(self, 
                            target_text: str, 
                            candidates: List[Dict], 
                            top_k: int = 10) -> List[Dict]:
        """
        向量检索召回候选片段
        
        Args:
            target_text: 目标文案
            candidates: 候选片段列表
            top_k: 召回数量
            
        Returns:
            召回的候选片段列表
        """
        if not candidates:
            return []
        
        # 编码目标文本
        target_vec = self.model.encode([target_text])
        
        # 编码候选文本
        candidate_texts = [c.get("text", "") for c in candidates]
        candidate_vecs = self.model.encode(candidate_texts)
        
        # 计算相似度
        similarities = np.dot(target_vec, candidate_vecs.T).flatten()
        
        # 获取 top_k 索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 返回 top_k 候选
        return [candidates[i] for i in top_indices if similarities[i] > 0]
    
    def _calculate_score(self,
                         candidate: Dict,
                         base_score: float,
                         target_text: str) -> Tuple[float, str]:
        """
        计算最终评分 - 包含惩罚逻辑

        Args:
            candidate: 候选片段
            base_score: 基础相似度分数
            target_text: 目标文案

        Returns:
            (最终分数, 原因)
        """
        final_score = base_score
        reason = "base"

        video_id = candidate.get("video_id")
        candidate_text = candidate.get("text", "")

        # 1. 数字匹配检查（硬约束）
        if not self._check_number_match(target_text, candidate_text):
            final_score -= self.PENALTY_NUMBER_MISMATCH
            reason = "number_mismatch"
            return final_score, reason  # 数字不匹配，直接返回

        # 2. A. 惩罚逻辑：防止非连续的视频重复出现
        if video_id in self.used_video_ids and video_id != self.last_match.get("video_id"):
            final_score -= self.PENALTY_REPEAT
            reason = f"repeat_penalty"

        # 3. B. 单调性约束：如果使用同一个视频，必须往后走
        if video_id == self.last_match.get("video_id"):
            last_end = self.last_match.get("end_time", 0)
            cand_start = candidate.get("start", 0)

            if cand_start > last_end:
                final_score += self.REWARD_SEQUENCE  # 顺序奖励
                reason = "sequence_reward"
            else:
                # 同视频内时间回跳，施加重度惩罚
                # 检查是否严重回跳（差距超过0.5秒）
                time_gap = last_end - cand_start
                if time_gap > 0.5:
                    final_score -= self.PENALTY_HARD_TIME_REVERSE  # 严重回跳惩罚
                    reason = f"hard_time_reverse_penalty (gap={time_gap:.1f}s)"
                else:
                    final_score -= self.PENALTY_TIME_REVERSE  # 轻微回跳惩罚
                    reason = f"time_reverse_penalty (gap={time_gap:.1f}s)"

        return final_score, reason

    def _apply_latest_recording_tiebreaker(self,
                                           candidates: List[Dict],
                                           base_scores: List[float],
                                           final_scores: List[Tuple[float, str]],
                                           best_final_score: float,
                                           best_idx: int) -> Tuple[int, float]:
        """
        应用"末次优先"平局决胜规则

        当多个候选的最终分数差距小于 TIEBREAK_THRESHOLD 时,
        选择最后录制的版本（索引最大的）

        Args:
            candidates: 候选片段列表
            base_scores: 基础相似度分数列表
            final_scores: 最终评分列表 [(score, reason), ...]
            best_final_score: 最佳最终分数
            best_idx: 最佳候选索引

        Returns:
            (最佳候选索引, 最佳最终分数)
        """
        # 收集所有分数接近最佳分数的候选
        tie_candidates = []
        for i, (score, _) in enumerate(final_scores):
            if abs(score - best_final_score) < self.TIEBREAK_THRESHOLD:
                tie_candidates.append(i)

        if len(tie_candidates) <= 1:
            # 没有平局，直接返回原结果
            return best_idx, best_final_score

        # 有平局！应用"末次优先"规则
        # 找出索引最大（最后录制）的候选
        logger.debug(f"  -> 平局决胜: {len(tie_candidates)}个候选分数接近 ({best_final_score:.3f} ± {self.TIEBREAK_THRESHOLD})")

        # 按索引（记录时间）排序，选择最大的
        tie_candidates_sorted = sorted(tie_candidates, reverse=True)
        best_tie_idx = tie_candidates_sorted[0]

        # 获取该候选的原始信息
        best_cand = candidates[best_tie_idx]
        best_base_score = base_scores[best_tie_idx]
        best_final_score_with_bonus = best_final_score + self.LATEST_RECORDING_BONUS

        logger.debug(f"  -> 末次优先: 选择索引{best_tie_idx}的片段 ({best_cand.get('video_name', 'N/A')})")
        logger.debug(f"     基础分数: {best_base_score:.3f}, 原最终分数: {best_final_score:.3f}, 加成后: {best_final_score_with_bonus:.3f}")

        return best_tie_idx, best_final_score_with_bonus
    
    def reset_state(self):
        """重置匹配状态"""
        self.used_video_ids.clear()
        self.last_match = {
            "video_id": None,
            "end_time": 0.0,
            "video_path": None
        }
    
    def requires_b_roll(self, text: str) -> bool:
        """
        检测文案是否需要 B-Roll
        
        Args:
            text: 目标文案
            
        Returns:
            是否需要 B-Roll
        """
        # B-Roll 关键词
        b_roll_keywords = {
            '空镜头', '空镜', '风景', '景色', '环境', '画面', '场景',
            '外景', '室内', '办公室', '产品', '商品', '展示', '演示',
            '背景', '氛围', '街头', '道路', '建筑', '自然', '城市',
            '乡村', '海边', '山景', '花', '草', '树', '天空', '云',
            '日出', '日落', '夜景', '灯光', '特写', '远景', '全景'
        }
        
        text_lower = text.lower()
        for keyword in b_roll_keywords:
            if keyword in text_lower:
                return True
        
        return False
    
    def _match_b_roll(self, target_text: str) -> Optional[Dict]:
        """
        匹配 B-Roll 素材

        Args:
            target_text: 目标文案

        Returns:
            匹配的 B-Roll 或 None
        """
        b_roll_materials = self.get_b_roll_materials()

        if not b_roll_materials:
            return None

        # 检索 B-Roll
        candidates = self._retrieve_candidates(target_text, b_roll_materials, top_k=5)

        best_match = None
        best_score = 0.0

        # 记录所有候选的评分（用于平局决胜）
        all_candidates_scores = []

        for cand in candidates:
            score, _ = self._calculate_score(cand, cand.get("similarity", 0))
            all_candidates_scores.append(score)

            if score > best_score:
                best_score = score
                best_match = cand

        # 应用"末次优先"平局决胜（如果有多个候选）
        if len(all_candidates_scores) > 1 and best_score >= self.B_ROLL_THRESHOLD:
            # 找出最佳候选的索引
            best_idx = candidates.index(best_match)
            # 检查是否有平局
            tie_candidates = []
            for i, score in enumerate(all_candidates_scores):
                if abs(score - best_score) < self.TIEBREAK_THRESHOLD:
                    tie_candidates.append(i)

            if len(tie_candidates) > 1:
                # 有平局，选择索引最大的（最后录制）
                logger.debug(f"  B_ROLL平局决胜: {len(tie_candidates)}个候选分数接近")
                best_idx = sorted(tie_candidates, reverse=True)[0]
                best_match = candidates[best_idx]
                logger.debug(f"     选择索引{best_idx}的B_ROLL: {best_match.get('video_name', 'N/A')}")

        if best_score >= self.B_ROLL_THRESHOLD:
            return best_match

        return None
    
    def plan(self, script: str) -> List[Dict]:
        """
        规划剪辑方案

        Args:
            script: 文案脚本（多行文本）

        Returns:
            剪辑决策列表（EDL）
        """
        # 重置状态
        self.reset_state()

        # 分句
        sentences = self._split_sentences(script)

        if not sentences:
            logger.warning("No sentences in script")
            return []

        logger.info(f"Planning for {len(sentences)} sentences")

        # 获取 A_ROLL 候选片段
        a_roll_candidates = self.get_a_roll_segments()

        if not a_roll_candidates:
            logger.warning("No A_ROLL candidates available")
            return []

        # 为候选添加material_index
        for idx, cand in enumerate(a_roll_candidates):
            cand["material_index"] = idx

        # 初始化素材列表（用于动态窗口）
        self.materials_list = a_roll_candidates

        # 对每个句子进行匹配
        final_edl = []

        for i, sentence in enumerate(sentences):
            logger.info(f"[{i+1}/{len(sentences)}] Processing: {sentence[:30]}...")

            # 0. 更新动态窗口
            self._update_dynamic_window(i, len(sentences))

            # 1. 首先检查是否需要 B-Roll
            if self.requires_b_roll(sentence):
                b_match = self._match_b_roll(sentence)
                if b_match:
                    # 应用 valid_offset
                    start_with_offset = b_match["start"] + b_match.get("valid_offset", 0)
                    end_with_offset = b_match["end"] + b_match.get("valid_offset", 0)

                    final_edl.append({
                        "video_path": b_match["video_path"],
                        "video_id": b_match["video_id"],
                        "start": start_with_offset,
                        "end": end_with_offset,
                        "text": sentence,
                        "matched_text": b_match.get("text", ""),
                        "similarity": b_match.get("similarity", 0),
                        "track_type": "B_ROLL",
                        "is_b_roll": True
                    })

                    # 更新状态
                    self.used_video_ids.add(b_match["video_id"])
                    self.last_match = {
                        "video_id": b_match["video_id"],
                        "end_time": end_with_offset,
                        "video_path": b_match["video_path"],
                        "material_index": -1
                    }

                    logger.info(f"  -> B_ROLL match: {b_match['video_name']}")
                    continue

            # 2. 从 A_ROLL 检索候选
            candidates = self._retrieve_candidates(sentence, a_roll_candidates, top_k=10)

            if not candidates:
                logger.warning(f"  -> No candidates found for: {sentence[:30]}...")
                final_edl.append({
                    "video_path": None,
                    "video_id": None,
                    "start": 0,
                    "end": 0,
                    "text": sentence,
                    "matched_text": "",
                    "similarity": 0,
                    "track_type": "UNKNOWN",
                    "missing": True
                })
                continue

            # 3. 应用动态窗口过滤
            filtered_candidates = self._filter_by_window(candidates)
            if len(filtered_candidates) < len(candidates):
                logger.debug(f"  动态窗口过滤: {len(candidates)} -> {len(filtered_candidates)} 候选")

            # 如果过滤后候选为空，使用原候选
            if not filtered_candidates:
                filtered_candidates = candidates

            # 4. 计算评分并选择最佳候选
            best_candidate = None
            best_final_score = float("-inf")
            best_reason = ""

            # 记录所有候选的评分信息（用于平局决胜）
            all_candidates_scores = []
            all_base_scores = []

            for cand in filtered_candidates:
                # 获取基础相似度
                base_score = self.model.compute_similarity(sentence, cand.get("text", ""))

                # 计算最终评分（传入target_text用于数字匹配）
                final_score, reason = self._calculate_score(cand, base_score, sentence)

                all_base_scores.append(base_score)
                all_candidates_scores.append((final_score, reason))

                if final_score > best_final_score:
                    best_final_score = final_score
                    best_candidate = cand
                    best_reason = reason

            # 5. 应用"末次优先"平局决胜规则
            if len(all_candidates_scores) > 1:
                # 找出最佳候选的索引
                best_idx = filtered_candidates.index(best_candidate)
                # 应用平局决胜
                best_idx, best_final_score = self._apply_latest_recording_tiebreaker(
                    filtered_candidates,
                    all_base_scores,
                    all_candidates_scores,
                    best_final_score,
                    best_idx
                )
                # 更新最佳候选
                best_candidate = filtered_candidates[best_idx]

            # 6. 检查是否满足阈值
            if best_candidate and best_final_score >= self.A_ROLL_THRESHOLD:
                # 应用 valid_offset
                start_with_offset = best_candidate["start"] + best_candidate.get("valid_offset", 0)
                end_with_offset = best_candidate["end"] + best_candidate.get("valid_offset", 0)

                final_edl.append({
                    "video_path": best_candidate["video_path"],
                    "video_id": best_candidate["video_id"],
                    "start": start_with_offset,
                    "end": end_with_offset,
                    "text": sentence,
                    "matched_text": best_candidate.get("text", ""),
                    "similarity": best_final_score,
                    "track_type": "A_ROLL",
                    "is_b_roll": False,
                    "reason": best_reason
                })

                # 更新状态
                self.used_video_ids.add(best_candidate["video_id"])
                self.last_match = {
                    "video_id": best_candidate["video_id"],
                    "end_time": end_with_offset,
                    "video_path": best_candidate["video_path"],
                    "material_index": best_candidate.get("material_index", -1)
                }
                
                logger.info(f"  -> A_ROLL match: {best_candidate['video_name']}, score={best_final_score:.3f} ({best_reason})")
                
            else:
                # 6. A_ROLL 匹配失败，尝试 B_ROLL 后备
                logger.info(f"  -> A_ROLL score too low ({best_final_score:.3f}), trying B_ROLL...")
                b_match = self._match_b_roll(sentence)
                
                if b_match:
                    start_with_offset = b_match["start"] + b_match.get("valid_offset", 0)
                    end_with_offset = b_match["end"] + b_match.get("valid_offset", 0)
                    
                    final_edl.append({
                        "video_path": b_match["video_path"],
                        "video_id": b_match["video_id"],
                        "start": start_with_offset,
                        "end": end_with_offset,
                        "text": sentence,
                        "matched_text": b_match.get("text", ""),
                        "similarity": b_match.get("similarity", 0),
                        "track_type": "B_ROLL",
                        "is_b_roll": True,
                        "fallback": True
                    })
                    
                    self.used_video_ids.add(b_match["video_id"])
                    self.last_match = {
                        "video_id": b_match["video_id"],
                        "end_time": end_with_offset,
                        "video_path": b_match["video_path"]
                    }
                    
                    logger.info(f"  -> B_ROLL fallback: {b_match['video_name']}")
                else:
                    # 完全没有匹配
                    final_edl.append({
                        "video_path": None,
                        "video_id": None,
                        "start": 0,
                        "end": 0,
                        "text": sentence,
                        "matched_text": "",
                        "similarity": 0,
                        "track_type": "UNKNOWN",
                        "missing": True
                    })
                    
                    logger.warning(f"  -> No match found")
        
        # 统计结果
        a_roll_count = sum(1 for e in final_edl if e.get("track_type") == "A_ROLL")
        b_roll_count = sum(1 for e in final_edl if e.get("track_type") == "B_ROLL")
        missing_count = sum(1 for e in final_edl if e.get("missing", False))
        
        logger.info(f"=== Planning Result ===")
        logger.info(f"Total: {len(final_edl)}, A_ROLL: {a_roll_count}, B_ROLL: {b_roll_count}, Missing: {missing_count}")
        
        return final_edl
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        分句
        
        Args:
            text: 待分句文本
            
        Returns:
            句子列表
        """
        sentences = re.split(r'[。！？；\n]', text)
        result = []
        for s in sentences:
            s = s.strip()
            if s:
                result.append(s)
        return result


# ============ 测试代码 ============
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试用例
    SCRIPT = """不用确定我们京东315活动3C、家电政府补贴至高15%
我再问一下
不用问,美的四大权益都可以享受
权益一
全屋智能家电套购
送至高1499元豪礼"""
    
    # 初始化
    planner = SequencePlanner(db_path="storage/materials.db")
    
    # 执行规划
    edl = planner.plan(SCRIPT)
    
    # 打印结果
    print("\n=== EDL Result ===")
    for i, clip in enumerate(edl):
        status = "OK" if not clip.get("missing") else "XX"
        print(f"{status} [{i+1}] {clip.get('text', '')[:30]}...")
        if not clip.get("missing"):
            print(f"    -> {clip.get('track_type')}: {clip.get('video_name', 'N/A')}, "
                  f"start={clip.get('start', 0):.2f}, end={clip.get('end', 0):.2f}")
