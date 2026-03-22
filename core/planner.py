"""
带状态的序列决策引擎 - SequencePlanner
实现带状态的全局匹配算法，输出 FFmpeg 剪辑列表（EDL）

核心功能：
1. 向量检索：使用 BGE-M3 原生 Tokenizer（禁止分词）
2. 候选过滤：主轨仅从 A_ROLL 素材中召回
3. 状态机维护：维护 used_video_ids 集合和 last_video_pos
4. 双重得分制：
   - BaseScore: 使用 BGE-M3 计算语义相似度
   - PenaltyScore:
     * 若 video_id 已使用且非连续，得分 -0.8
     * 若同视频内时间戳回退，得分 -1.0
5. A/B Roll 隔离：主轨匹配仅查询 track_type='A_ROLL'
6. 日志完备性：每一行文案匹配了哪个素材、得分多少、为何过滤，必须清晰打印
"""
import os
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import json

from db.models import Database, Asset, Segment

logger = logging.getLogger("smart_cut")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class EmbeddingModel:
    """向量嵌入模型封装 - BGE-M3 原生接口（禁止分词）"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化嵌入模型
        
        Args:
            model_name: 模型名称，默认为多语言 MiniLM（兼容 torch 2.4.1）
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载 BGE-M3 模型"""
        try:
            from sentence_transformers import SentenceTransformer

            # 获取调整后的批量大小
            from core.hardware import set_batch_size_for_igpu
            self.batch_size = set_batch_size_for_igpu(32)

            logger.info(f"Loading BGE-M3 model via SentenceTransformer: {self.model_name}")
            logger.info(f"Batch size: {self.batch_size}")

            # ====== CPU 设备加载 ======
            # DirectML 与 sentence-transformers inference_mode 存在兼容性冲突，暂时使用 CPU
            self.model = SentenceTransformer(
                self.model_name,
                device="cpu"
            )
            logger.info("BGE-M3 model loaded on CPU (DirectML compatibility issue)")

            # 使用模型自带的 tokenizer
            self.tokenizer = self.model.tokenizer

        except Exception as e:
            logger.error(f"Failed to load BGE-M3: {e}")
            raise RuntimeError(f"模型加载失败: {e}")
    
    def encode(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        将文本编码为向量（使用 BGE-M3 原生接口）

        Args:
            texts: 文本列表
            batch_size: 批处理大小，默认使用初始化时的 batch_size

        Returns:
            向量数组 (n, dim)
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")

        if batch_size is None:
            batch_size = self.batch_size

        try:
            # 使用 SentenceTransformer 原生接口 (兼容 DirectML)
            # 不使用 .half() 避免与 DirectML inference mode 冲突
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"DirectML encode failed: {e}")
            return self._fallback_encode(texts, batch_size)
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """手动 L2 归一化"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        return embeddings / norms
    
    def _fallback_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """降级编码方法"""
        if self.model is None:
            raise RuntimeError("No embedding model available")
        
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
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
        vec_a, vec_b = embeddings[0], embeddings[1]
        return float(np.dot(vec_a, vec_b))


class SequencePlanner:
    """全局匹配规划器 - 带状态机"""
    
    # 惩罚参数
    PENALTY_REPEAT = 0.8        # 非连续重复惩罚
    PENALTY_TIME_REVERSE = 1.0  # 时间倒流惩罚
    REWARD_SEQUENCE = 0.2       # 顺序保持奖励
    
    # 阈值
    A_ROLL_THRESHOLD = 0.2     # A_ROLL 最低相似度阈值（放宽以召回更多素材）
    B_ROLL_THRESHOLD = 0.0    # B_ROLL 最低相似度阈值（强制兜底，0表示不限制）
    
    # B-Roll 关键词
    B_ROLL_KEYWORDS = {
        '空镜头', '空镜', '风景', '景色', '环境', '画面', '场景',
        '外景', '室内', '办公室', '产品', '商品', '展示', '演示',
        '背景', '氛围', '街头', '道路', '建筑', '自然', '城市',
        '乡村', '海边', '山景', '花', '草', '树', '天空', '云',
        '日出', '日落', '夜景', '灯光', '特写', '远景', '全景'
    }
    
    def __init__(self, db: Database, embedding_model: Optional[EmbeddingModel] = None):
        """
        初始化 SequencePlanner
        
        Args:
            db: Database 实例
            embedding_model: 嵌入模型实例
        """
        self.db = db
        self.model = embedding_model or EmbeddingModel()
        
        # ============ 状态机维护 ============
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
        try:
            assets = self.db.get_session().query(Asset).all()
            
            for asset in assets:
                # 解析 ASR 结果
                segments_data = []
                asr_text = ""
                transcript = asset.transcript_json
                if transcript:
                    try:
                        data = json.loads(transcript)
                        asr_text = data.get("text", "")
                        segments_data = data.get("segments", [])
                    except:
                        pass
                
                self.materials_cache[asset.file_path] = {
                    "video_id": asset.id,
                    "path": asset.file_path,
                    "name": asset.file_name,
                    "track_type": asset.track_type,
                    "valid_start_offset": asset.valid_start_offset,
                    "duration": asset.duration,
                    "asr_text": asr_text,
                    "segments": segments_data
                }
            
            logger.info(f"Loaded {len(self.materials_cache)} materials from database")
            
        except Exception as e:
            logger.error(f"Failed to load materials: {e}")
    
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
        for keyword in self.B_ROLL_KEYWORDS:
            if keyword in text:
                return True
        return False
    
    def get_a_roll_segments(self) -> List[Dict]:
        """获取所有 A_ROLL 素材的片段（合并字级时间戳为完整句子）"""
        segments = []

        for file_path, mat in self.materials_cache.items():
            if mat["track_type"] != "A_ROLL":
                continue

            mat_segments = mat.get("segments", [])
            valid_offset = mat.get("valid_start_offset", 0.0)

            if mat_segments:
                # ASR 返回的是逐字时间戳，需要合并为完整句子
                # 按时间排序
                sorted_segments = sorted(mat_segments, key=lambda x: x.get("start", 0))

                # 合并相邻的字为句子
                merged = []
                current_text = ""
                current_start = None
                current_end = None

                for seg in sorted_segments:
                    text = seg.get("text", "")
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)

                    if text and len(text.strip()) > 0:
                        if current_start is None:
                            current_text = text
                            current_start = start
                            current_end = end
                        else:
                            # 合并相邻时间戳（间隔小于0.5秒视为同一句）
                            if start - current_end < 0.5:
                                current_text += text
                                current_end = end
                            else:
                                merged.append({
                                    "text": current_text,
                                    "start": current_start,
                                    "end": current_end
                                })
                                current_text = text
                                current_start = start
                                current_end = end

                # 添加最后一个
                if current_text and current_start is not None:
                    merged.append({
                        "text": current_text,
                        "start": current_start,
                        "end": current_end
                    })

                # 为每个合并后的片段添加 padding 并计算实际时间
                # 重构时间轴逻辑：
                # - start_time = 提示词时间 + 0.2s (valid_offset + padding)
                # - end_time = ASR 最后一个字时间戳 + 0.5s
                # - 如果无时间戳，使用视频总时长
                padding_start = 0.2
                padding_end = 0.5
                min_duration = 1.5

                # 视频总时长
                video_duration = mat.get("duration", 300.0)

                for seg in merged:
                    text = seg.get("text", "")
                    if not text or len(text.strip()) < 2:
                        continue

                    # 获取原始时间戳
                    raw_start = seg.get("start", 0)
                    raw_end = seg.get("end", 0)

                    # 计算实际开始时间
                    # 确保从提示词结束后开始 + padding
                    cue_end = valid_offset + padding_start
                    actual_start = max(cue_end, raw_start)

                    # 计算实际结束时间
                    # 如果 ASR 没有返回有效时间戳，使用视频总时长
                    if raw_end > 0:
                        actual_end = raw_end + padding_end
                    else:
                        # 无时间戳时使用整个视频
                        actual_end = video_duration

                    # 确保不超过视频总时长
                    if actual_end > video_duration:
                        actual_end = video_duration

                    # 确保最小持续时间
                    duration = actual_end - actual_start
                    if duration < min_duration:
                        actual_end = actual_start + min_duration
                        # 如果超过视频时长，则从前面截断
                        if actual_end > video_duration:
                            actual_start = max(0, video_duration - min_duration)
                            actual_end = video_duration

                    # 使用 round 避免浮点误差
                    actual_start = round(actual_start, 3)
                    actual_end = round(actual_end, 3)

                    segments.append({
                        "video_id": mat["video_id"],
                        "video_path": mat["path"],
                        "video_name": mat["name"],
                        "track_type": "A_ROLL",
                        "text": text,
                        "start": actual_start,
                        "end": actual_end,
                        "duration": round(actual_end - actual_start, 3),
                        "valid_offset": valid_offset
                    })

                    logger.debug(f"片段: {mat['name']} - {text[:20]}... "
                               f"start={actual_start:.3f}s end={actual_end:.3f}s duration={actual_end-actual_start:.3f}s")
            else:
                # 无 segments，使用整个素材（从 valid_offset 开始到视频结束）
                text = mat.get("asr_text", "")
                if text:
                    # 从提示词结束后开始
                    start = valid_offset + padding_start
                    # 使用视频总时长作为结束
                    end = mat.get("duration", video_duration)

                    # 确保不超过视频时长
                    if end > video_duration:
                        end = video_duration

                    duration = end - start
                    if duration < min_duration:
                        # 素材太短，调整起始位置
                        start = max(0, end - min_duration)

                    segments.append({
                        "video_id": mat["video_id"],
                        "video_path": mat["path"],
                        "video_name": mat["name"],
                        "track_type": "A_ROLL",
                        "text": text,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "duration": round(end - start, 3),
                        "valid_offset": valid_offset
                    })

        logger.info(f"Total A_ROLL segments: {len(segments)}")
        return segments
    
    def get_b_roll_materials(self) -> List[Dict]:
        """获取所有 B_ROLL 素材"""
        materials = []
        
        for file_path, mat in self.materials_cache.items():
            if mat["track_type"] == "B_ROLL":
                materials.append({
                    "video_id": mat["video_id"],
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

    def get_a_roll_materials(self) -> List[Dict]:
        """获取所有 A_ROLL 素材（用于统计）"""
        materials = []

        for file_path, mat in self.materials_cache.items():
            if mat["track_type"] == "A_ROLL":
                materials.append({
                    "video_id": mat["video_id"],
                    "video_path": mat["path"],
                    "video_name": mat["name"],
                    "track_type": "A_ROLL",
                    "text": mat.get("asr_text", ""),
                    "start": 0,
                    "end": mat.get("duration", 5.0),
                    "valid_offset": mat.get("valid_start_offset", 0.0)
                })

        logger.info(f"Total A_ROLL materials: {len(materials)}")
        return materials

    def _retrieve_candidates(self, target_text: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        向量检索召回候选片段
        
        Args:
            target_text: 目标文案
            candidates: 候选片段列表
            top_k: 召回数量
            
        Returns:
            召回的候选片段列表（包含 similarity 字段）
        """
        if not candidates:
            return []
        
        # 编码
        target_vec = self.model.encode([target_text])
        candidate_texts = [c.get("text", "") for c in candidates]
        candidate_vecs = self.model.encode(candidate_texts)
        
        # 计算相似度
        similarities = np.dot(target_vec, candidate_vecs.T).flatten()
        
        # 获取 top_k 索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 返回带相似度的候选
        result = []
        for i in top_indices:
            if similarities[i] > 0:
                cand = candidates[i].copy()
                cand["similarity"] = float(similarities[i])
                result.append(cand)
        
        return result
    
    def _calculate_score(self, candidate: Dict, base_score: float) -> Tuple[float, str]:
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
            reason = "repeat_penalty"
        
        # B. 单调性约束：如果使用同一个视频，必须往后走
        if video_id == self.last_match.get("video_id"):
            last_end = self.last_match.get("end_time", 0)
            cand_start = candidate.get("start", 0)
            
            if cand_start > last_end:
                final_score += self.REWARD_SEQUENCE
                reason = "sequence_reward"
            else:
                final_score -= self.PENALTY_TIME_REVERSE
                reason = "time_reverse_penalty"
        
        return final_score, reason
    
    def _match_b_roll(self, target_text: str) -> Optional[Dict]:
        """匹配 B-Roll 素材（全量召回：阈值=0时返回最佳匹配）"""
        b_roll_materials = self.get_b_roll_materials()

        if not b_roll_materials:
            return None

        candidates = self._retrieve_candidates(target_text, b_roll_materials, top_k=5)

        best_match = None
        best_score = 0.0

        for cand in candidates:
            score, _ = self._calculate_score(cand, cand.get("similarity", 0))

            if score > best_score:
                best_score = score
                best_match = cand

        # 全量召回：阈值<=0 时总是返回最佳匹配
        if self.B_ROLL_THRESHOLD <= 0:
            return best_match

        if best_score >= self.B_ROLL_THRESHOLD:
            return best_match

        return None
    
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        sentences = re.split(r'[。！？；\n]', text)
        result = []
        for s in sentences:
            s = s.strip()
            if s:
                result.append(s)
        return result
    
    def plan(self, script: str) -> List[Dict]:
        """
        规划剪辑方案
        
        Args:
            script: 文案脚本
            
        Returns:
            剪辑决策列表（EDL）
        """
        # 重置状态
        self.reset_state()

        # 重新加载素材数据（确保获取最新处理的素材）
        self._load_materials()
        
        # 分句
        sentences = self._split_sentences(script)
        
        if not sentences:
            logger.warning("No sentences in script")
            return []
        
        logger.info(f"Planning for {len(sentences)} sentences")

        # 打印素材库统计
        a_roll_materials = self.get_a_roll_materials()
        b_roll_materials = self.get_b_roll_materials()
        logger.info(f"正在扫描素材库，当前可用 A-Roll: {len(a_roll_materials)} 段, B-Roll: {len(b_roll_materials)} 段")

        if len(a_roll_materials) < len(sentences):
            logger.warning(f"素材不足！文案 {len(sentences)} 句，但 A-Roll 只有 {len(a_roll_materials)} 段，将使用 B-Roll 补位")

        # 获取候选片段
        a_roll_candidates = self.get_a_roll_segments()
        
        if not a_roll_candidates:
            logger.warning("No A_ROLL candidates available")
            return []
        
        # 对每个句子进行匹配
        final_edl = []
        
        for i, sentence in enumerate(sentences):
            logger.info(f"[{i+1}/{len(sentences)}] Processing: {sentence[:30]}...")
            
            # 1. 检查是否需要 B-Roll
            if self.requires_b_roll(sentence):
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
                        "is_b_roll": True
                    })
                    
                    self.used_video_ids.add(b_match["video_id"])
                    self.last_match = {
                        "video_id": b_match["video_id"],
                        "end_time": end_with_offset,
                        "video_path": b_match["video_path"]
                    }
                    
                    logger.info(f"  -> B_ROLL match: {b_match['video_name']}, score={b_match.get('similarity', 0):.3f}")
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
                base_score = self.model.compute_similarity(sentence, cand.get("text", ""))
                final_score, reason = self._calculate_score(cand, base_score)
                
                logger.info(f"  -> Candidate: {cand['video_name']}, base={base_score:.3f}, "
                          f"final={final_score:.3f}, reason={reason}")
                
                if final_score > best_final_score:
                    best_final_score = final_score
                    best_candidate = cand
                    best_reason = reason
            
            # 4. 检查阈值
            if best_candidate and best_final_score >= self.A_ROLL_THRESHOLD:
                start_with_offset = round(best_candidate["start"] + best_candidate.get("valid_offset", 0), 3)
                end_with_offset = round(best_candidate["end"] + best_candidate.get("valid_offset", 0), 3)
                clip_duration = round(end_with_offset - start_with_offset, 3)

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

                self.used_video_ids.add(best_candidate["video_id"])
                self.last_match = {
                    "video_id": best_candidate["video_id"],
                    "end_time": end_with_offset,
                    "video_path": best_candidate["video_path"]
                }

                # 详细日志输出
                logger.info(f"文案：{sentence[:30]}... -> 匹配素材：{best_candidate['video_name']}，"
                          f"得分：{best_final_score:.3f}，切片时长：{clip_duration:.3f}秒")
                
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
                    # 兜底：强制随机选择一个 B_ROLL 补位
                    b_roll_materials = self.get_b_roll_materials()
                    if b_roll_materials:
                        import random
                        fallback = random.choice(b_roll_materials)
                        start_with_offset = fallback["start"] + fallback.get("valid_offset", 0)
                        end_with_offset = fallback["end"] + fallback.get("valid_offset", 0)

                        final_edl.append({
                            "video_path": fallback["video_path"],
                            "video_id": fallback["video_id"],
                            "start": start_with_offset,
                            "end": end_with_offset,
                            "text": sentence,
                            "matched_text": fallback.get("text", ""),
                            "similarity": 0,
                            "track_type": "B_ROLL",
                            "is_b_roll": True,
                            "fallback": True,
                            "forced_fallback": True  # 强制兜底标记
                        })

                        self.used_video_ids.add(fallback["video_id"])
                        logger.warning(f"  -> 强制使用 B_ROLL 补位: {fallback['video_name']}")
                    else:
                        # 完全没有素材，报错
                        logger.error(f"  -> 错误：没有任何素材可用！")
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
        
        # 统计结果
        a_roll_count = sum(1 for e in final_edl if e.get("track_type") == "A_ROLL")
        b_roll_count = sum(1 for e in final_edl if e.get("track_type") == "B_ROLL")
        missing_count = sum(1 for e in final_edl if e.get("missing", False))
        
        logger.info(f"=== Planning Result ===")
        logger.info(f"Total: {len(final_edl)}, A_ROLL: {a_roll_count}, B_ROLL: {b_roll_count}, Missing: {missing_count}")

        # 可视化编排清单
        logger.info("=" * 60)
        logger.info("📋 编排清单:")
        logger.info("=" * 60)
        for i, clip in enumerate(final_edl, 1):
            if clip.get("missing"):
                logger.info(f"文案 {i}: [缺失素材]")
            elif clip.get("track_type") == "A_ROLL":
                duration = clip["end"] - clip["start"]
                match_type = "精确匹配" if not clip.get("fallback") else "B-Roll降级"
                logger.info(f"文案 {i}: [{clip.get('video_name', 'A_ROLL')}] [得分 {clip.get('similarity', 0):.2f}] [时长 {duration:.2f}s] {match_type}")
            else:
                duration = clip["end"] - clip["start"]
                fallback_type = "强制兜底" if clip.get("forced_fallback") else "B_Roll"
                logger.info(f"文案 {i}: [自动补位 B-Roll] [时长 {duration:.2f}s] {fallback_type}")
        logger.info("=" * 60)
        
        return final_edl


# ============ 测试代码 ============
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试用例
    SCRIPT = """不用确定我们京东315活动3C、家电政府补贴至高15%
    我再问一下
    不用问,美的四大权益都可以享受
    权益一
    全屋智能家电套购
    送至高1499元豪礼"""
    
    # 初始化
    db = Database("storage/materials.db")
    planner = SequencePlanner(db)
    
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
