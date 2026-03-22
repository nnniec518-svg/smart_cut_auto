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
            
            model_path = MODELS_DIR / "sentence_transformers" / "models--all-MiniLM-L6-v2"
            if model_path.exists():
                self.model = SentenceTransformer(str(model_path))
            else:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
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
    PENALTY_REPEAT = 0.8      # 非连续重复惩罚
    PENALTY_TIME_REVERSE = 1.0  # 时间倒流惩罚
    REWARD_SEQUENCE = 0.2     # 顺序保持奖励
    
    # 阈值
    A_ROLL_THRESHOLD = 0.3   # A_ROLL 最低相似度阈值
    B_ROLL_THRESHOLD = 0.1   # B_ROLL 最低相似度阈值
    
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
            "video_path": None
        }
        
        # 加载素材数据
        self.materials_cache: Dict[str, Dict] = {}
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
                         base_score: float) -> Tuple[float, str]:
        """
        计算最终评分 - 包含惩罚逻辑
        
        Args:
            candidate: 候选片段
            base_score: 基础相似度分数
            
        Returns:
            (最终分数, 原因)
        """
        final_score = base_score
        reason = "base"
        
        video_id = candidate.get("video_id")
        
        # A. 惩罚逻辑：防止非连续的视频重复出现
        if video_id in self.used_video_ids and video_id != self.last_match.get("video_id"):
            final_score -= self.PENALTY_REPEAT
            reason = f"repeat_penalty"
        
        # B. 单调性约束：如果使用同一个视频，必须往后走
        if video_id == self.last_match.get("video_id"):
            last_end = self.last_match.get("end_time", 0)
            cand_start = candidate.get("start", 0)
            
            if cand_start > last_end:
                final_score += self.REWARD_SEQUENCE  # 顺序奖励
                reason = "sequence_reward"
            else:
                final_score -= self.PENALTY_TIME_REVERSE  # 致命惩罚
                reason = "time_reverse_penalty"
        
        return final_score, reason
    
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
        
        for cand in candidates:
            score, _ = self._calculate_score(cand, cand.get("similarity", 0))
            
            if score > best_score:
                best_score = score
                best_match = cand
        
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
        
        # 对每个句子进行匹配
        final_edl = []
        
        for i, sentence in enumerate(sentences):
            logger.info(f"[{i+1}/{len(sentences)}] Processing: {sentence[:30]}...")
            
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
                        "video_path": b_match["video_path"]
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
            
            # 3. 计算评分并选择最佳候选
            best_candidate = None
            best_final_score = float("-inf")
            best_reason = ""
            
            for cand in candidates:
                # 获取基础相似度
                base_score = self.model.compute_similarity(sentence, cand.get("text", ""))
                
                # 计算最终评分
                final_score, reason = self._calculate_score(cand, base_score)
                
                if final_score > best_final_score:
                    best_final_score = final_score
                    best_candidate = cand
                    best_reason = reason
            
            # 4. 检查是否满足阈值
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
                    "video_path": best_candidate["video_path"]
                }
                
                logger.info(f"  -> A_ROLL match: {best_candidate['video_name']}, score={best_final_score:.3f} ({best_reason})")
                
            else:
                # 5. A_ROLL 匹配失败，尝试 B_ROLL 后备
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
