"""
文案匹配模块
实现语义匹配、文案分句、跨素材决策等功能
"""
import os
import re
import jieba
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

logger = logging.getLogger("smart_cut")

# 过滤无用的语气词/开场白
FILTER_WORDS = {
    '走', '嗯', '啊', '呀', '哦', '哈', '嘿', '喂', '呃', 
    '那个', '这个', '然后', '就是', '这个那个', 
    '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    '一二三', '一二三四五', '321', '三二一', '三二一', 
    '胜', '可以', '好的', 'OK', 'ok', '呀', '咯', '嘛', '呐',
    '来', '去', '这', '那', '我', '你', '他', '她', '它',
    '的', '了', '着', '过', '吗', '呢', '吧', '啊',
    '想', '说', '要', '会', '能', '就', '还', '也', '都'
}

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class Matcher:
    """文案匹配器"""
    
    def __init__(self, similarity_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化匹配器
        
        Args:
            similarity_model_name: Sentence-Transformer模型名称
        """
        self.model_name = similarity_model_name
        self.model = None
        self._ensure_models_dir()
        self._load_model()
        logger.info(f"Matcher initialized with model: {similarity_model_name}")
    
    def _ensure_models_dir(self) -> None:
        """确保模型目录存在"""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # 设置SentenceTransformers缓存路径
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR / "sentence_transformers")
    
    def _load_model(self) -> None:
        """加载Sentence-Transformer模型"""
        try:
            # 检查本地模型目录
            local_model_path = MODELS_DIR / "sentence_transformers" / f"models--{self.model_name.replace('/', '--')}"
            
            # 使用缓存目录加载（ SentenceTransformers 会自动使用本地缓存）
            cache_folder = str(MODELS_DIR / "sentence_transformers")
            
            if local_model_path.exists():
                # 使用本地模型（通过 cache_folder）
                logger.info(f"Using local cached model: {self.model_name}")
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=cache_folder
                )
            else:
                # 尝试从缓存或网络加载
                logger.info(f"Loading model: {self.model_name}")
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=cache_folder
                )
            logger.info(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            # 降级到简单的词匹配
            self.model = None
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本：去除ASR结果中的空格"""
        # 去除空格
        text = text.replace(' ', '')
        # 去除多余标点
        text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
        return text.lower()

    def text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的余弦相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数 (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # 预处理文本
        text1 = self._preprocess_text(text1)
        text2 = self._preprocess_text(text2)
        
        if self.model is not None:
            try:
                # 编码两个文本
                embeddings = self.model.encode([text1, text2])
                
                # 计算余弦相似度
                cos_sim = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                
                return float(cos_sim)
            except Exception as e:
                logger.warning(f"语义相似度计算失败: {e}")
        
        # 降级：使用简单的词重叠
        return self._word_overlap_similarity(text1, text2)
    
    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """基于字符重叠的相似度 - 增强版"""
        # 预处理：去除空格和非中文字符
        t1 = re.sub(r'[^\u4e00-\u9fff]', '', text1.replace(' ', ''))
        t2 = re.sub(r'[^\u4e00-\u9fff]', '', text2.replace(' ', ''))
        
        if not t1 or not t2:
            return 0.0
        
        # 方法1: 字符级Jaccard
        set1 = set(t1)
        set2 = set(t2)
        intersection = set1 & set2
        union = set1 | set2
        jaccard = len(intersection) / len(union) if union else 0
        
        # 方法2: 最长公共子串
        lcs_len = self._longest_common_substring_len(t1, t2)
        lcs_ratio = lcs_len / max(len(t1), len(t2)) if t1 and t2 else 0
        
        # 方法3: 子串包含检测（关键！）
        # 如果t1的大部分字符都在t2中，给与高分
        contains_score = 0.0
        if len(t1) >= 2:
            # 统计t1中有多少字符在t2中
            contained_chars = sum(1 for c in t1 if c in t2)
            contains_score = contained_chars / len(t1)
        
        # 综合相似度
        # Jaccard 20% + LCS 40% + 包含检测 40%
        return jaccard * 0.2 + lcs_ratio * 0.4 + contains_score * 0.4
    
    def _longest_common_substring_len(self, s1: str, s2: str) -> int:
        """计算最长公共子串长度"""
        if not s1 or not s2:
            return 0
        
        m, n = len(s1), len(s2)
        # 空间优化：只保留上一行
        prev = [0] * (n + 1)
        max_len = 0
        
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    curr[j] = prev[j-1] + 1
                    max_len = max(max_len, curr[j])
            prev = curr
        
        return max_len
    
    def _preprocess_asr_text(self, text: str) -> str:
        """预处理ASR识别结果，去除重复词"""
        # 去除空格
        text = text.replace(' ', '')
        
        # 去除连续重复的词（如"权益四胜权益四胜" -> "权益四"）
        words = list(jieba.cut(text))
        seen = []
        for w in words:
            if not seen or w != seen[-1]:
                seen.append(w)
        
        # 过滤常见无关词
        filtered = [w for w in seen if w not in FILTER_WORDS and len(w) > 1]
        return ''.join(filtered)
    
    def _tokenize(self, text: str) -> List[str]:
        """使用jieba分词"""
        # 去除空格
        text = text.replace(' ', '')
        # 使用jieba分词
        words = list(jieba.cut(text))
        # 过滤无用的词和单字符
        words = [w for w in words if w and len(w) > 1 and w not in FILTER_WORDS]
        return words
    
    def process_single_material(self, 
                                 audio_path: str, 
                                 target_text: str,
                                 asr_model: Any,
                                 audio_processor: Any,
                                 silence_threshold: float = 1.5,
                                 similarity_threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        """
        处理单个素材
        
        Args:
            audio_path: 音频/视频文件路径
            target_text: 目标文案全文
            asr_model: ASR模型实例
            audio_processor: 音频处理器实例
            silence_threshold: 静音分割阈值
            similarity_threshold: 最低相似度阈值
            
        Returns:
            最佳片段信息或None
        """
        try:
            # 1. 提取音频
            if audio_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                audio_path = audio_processor.extract_audio_from_video(audio_path)
            
            # 2. 加载音频
            audio, sr = audio_processor.load_audio(audio_path)
            
            # 2.5 切除开头的非语音部分（提示音等）- 禁用以避免时间偏移问题
            # remove_leading = True  # 禁用，避免时间戳不匹配
            # if remove_leading:
            #     audio, removed_duration = audio_processor.remove_leading_sound(audio, sr, max_duration=2.0)
            #     if removed_duration > 0:
            #         logger.info(f"Removed leading sound: {removed_duration:.2f}s")
            
            # 3. 获取语音段
            speech_segments = audio_processor.get_speech_segments(audio, sr)
            
            # 打印调试信息
            if not speech_segments:
                # 计算音频能量帮助调试
                energy = np.abs(audio)
                mean_energy = np.mean(energy)
                max_energy = np.max(energy)
                logger.warning(f"No speech segments detected in {audio_path}, mean_energy={mean_energy:.6f}, max_energy={max_energy:.6f}")
                return None
            
            logger.info(f"Detected {len(speech_segments)} speech segments in {audio_path}")
            
            # 4. 对整个音频进行ASR识别，获取更准确的时间戳
            logger.info(f"Running ASR on full audio for {audio_path}")
            asr_result = asr_model.transcribe_audio(audio, sr, word_timestamps=True)
            
            if not asr_result or not asr_result.get("text"):
                logger.warning(f"ASR failed for {audio_path}")
                return None
            
            asr_text = asr_result.get("text", "")
            asr_segments = asr_result.get("segments", [])
            
            logger.info(f"ASR result: {asr_text[:50]}... ({len(asr_segments)} segments)")
            
            # 计算相似度
            similarity = self.text_similarity(asr_text, target_text)
            logger.info(f"Full audio similarity: {similarity:.3f}")
            
            if similarity < similarity_threshold:
                logger.warning(f"Similarity {similarity:.3f} below threshold {similarity_threshold}")
                return None
            
            # 使用语音段信息裁剪时间范围
            # 裁剪掉开头的静音段
            trimmed_start = 0
            if speech_segments:
                first_speech = speech_segments[0]
                trimmed_start = first_speech[0]  # 第一个语音段的开始时间
                logger.info(f"Trimming leading silence: {trimmed_start:.3f}s")
            
            # 裁剪掉结尾的静音段
            trimmed_end = len(audio) / sr
            if speech_segments:
                last_speech = speech_segments[-1]
                trimmed_end = last_speech[1]  # 最后一个语音段的结束时间
                logger.info(f"Trimming trailing silence: {trimmed_end:.3f}s")
            
            # 使用ASR返回的第一个和最后一个时间戳作为片段边界
            if asr_segments:
                # 获取整个音频的时间范围
                first_seg = asr_segments[0]
                last_seg = asr_segments[-1]
                start_time = max(first_seg.get("start", 0), trimmed_start)
                end_time = min(last_seg.get("end", trimmed_end), trimmed_end)
                
                logger.info(f"Using ASR timestamps: {start_time:.3f}s - {end_time:.3f}s")
            else:
                # 如果没有时间戳，使用整个音频
                start_time = trimmed_start
                end_time = trimmed_end
            
            # ASR时间戳Padding：前后各增加0.2秒，防止ASR截断导致吞字
            padding_sec = 0.2
            video_duration = len(audio) / sr
            start_time = max(0, start_time - padding_sec)
            end_time = min(video_duration, end_time + padding_sec)
            logger.info(f"Applied timestamp padding: {start_time:.3f}s - {end_time:.3f}s")
            
            return {
                "start_time": start_time,
                "end_time": end_time,
                "asr_result": asr_result,
                "similarity": similarity,
                "audio_path": audio_path,
                "speech_segments": speech_segments  # 保存语音段供后续裁剪使用
            }
            
        except Exception as e:
            logger.error(f"处理素材失败: {audio_path}, {e}")
            return None
    
    def decide_best_materials(self,
                               material_results: List[Optional[Dict[str, Any]]],
                               target_text: str,
                               single_threshold: float = 0.85,
                               sentence_threshold: float = 0.6) -> Dict[str, Any]:
        """
        跨素材决策
        
        Args:
            material_results: 每个素材的process_single_material结果列表
            target_text: 目标文案全文
            single_threshold: 单素材选用阈值
            sentence_threshold: 句子匹配阈值
            
        Returns:
            决策结果字典
        """
        # 过滤有效结果
        valid_results = [r for r in material_results if r is not None]
        
        if not valid_results:
            return {"mode": "none", "segments": [], "error": "No valid material results"}
        
        # 1. 检查是否有单素材满足阈值
        best_single = None
        best_single_similarity = 0.0
        
        for result in valid_results:
            if result["similarity"] >= single_threshold:
                if result["similarity"] > best_single_similarity:
                    best_single_similarity = result["similarity"]
                    best_single = result
        
        if best_single is not None:
            logger.info(f"Single material mode selected, similarity: {best_single_similarity:.3f}")
            return {
                "mode": "single",
                "segments": [{
                    "material_index": material_results.index(best_single),
                    "start": best_single["start_time"],
                    "end": best_single["end_time"],
                    "matched_sentence_indices": list(range(len(best_single["asr_result"]["segments"])))
                }],
                "source": best_single
            }
        
        # 2. 多素材拼接模式
        logger.info("Multi-material concatenation mode")
        
        # 文案分句
        target_sentences = self._split_sentences(target_text)
        
        if not target_sentences:
            return {"mode": "none", "segments": [], "error": "Empty target text"}
        
        # 收集所有素材的句子
        all_sentences = []
        for i, result in enumerate(valid_results):
            for seg in result["asr_result"]["segments"]:
                all_sentences.append({
                    "material_index": material_results.index(result),
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "material_result": result
                })
        
        if not all_sentences:
            return {"mode": "none", "segments": [], "error": "No sentences available"}
        
        # 句子匹配
        matched_segments = self._greedy_sentence_matching(
            target_sentences,
            all_sentences,
            sentence_threshold
        )
        
        return {
            "mode": "concat",
            "segments": matched_segments,
            "target_sentences": target_sentences
        }
    
    def _greedy_sentence_matching(self,
                                   target_sentences: List[str],
                                   available_sentences: List[Dict],
                                   threshold: float = 0.6,
                                   order_bonus: float = 0.2,
                                   max_time_gap: float = 5.0,
                                   index_penalty_factor: float = 0.1,
                                   search_window_size: int = 30,
                                   window_backtrack: int = 5,
                                   adaptive_window: bool = True) -> List[Dict]:
        """
        有序句子匹配算法 - 动态滑动窗口 + 顺序约束 + 索引距离惩罚

        Args:
            target_sentences: 目标文案句子列表
            available_sentences: 可用的素材句子列表
            threshold: 相似度阈值
            order_bonus: 顺序匹配奖励分数（相邻句子即使相似度稍低也优先选择）
            max_time_gap: 判定为"相邻"的最大时间间隔（秒）
            index_penalty_factor: 索引距离惩罚系数（离预期位置越远扣分越多）
            search_window_size: 基础滑动窗口大小
            window_backtrack: 窗口回退距离（允许向前搜索的范围）
            adaptive_window: 是否启用自适应窗口大小

        Returns:
            匹配结果列表
        """
        # 记录已使用的素材句子（避免重复使用）
        used_sentence_indices = set()

        matched = []

        # 记录上一个匹配的句子索引，用于顺序约束和索引距离计算
        last_matched_idx = -1

        # 滑动窗口起始索引：强制"推着走"
        window_start_idx = 0

        # 统计信息
        consecutive_failures = 0  # 连续匹配失败次数
        max_consecutive_failures = 3  # 最大允许连续失败次数
        window_expansion_factor = 2  # 窗口扩展倍数(匹配失败时)

        for i, target_sent in enumerate(target_sentences):
            best_match = None
            best_final_score = -float('inf')  # 初始为负无穷，确保能找到任何有效匹配
            best_idx = -1
            best_raw_similarity = 0.0

            # 用于调试：记录所有候选的得分情况
            all_candidates = []

            # ========== 动态窗口计算 ==========
            # 如果启用自适应窗口且连续失败，扩大窗口范围
            current_window_size = search_window_size
            if adaptive_window and consecutive_failures > 0:
                # 扩大窗口: 基础窗口 * (1 + 扩展倍数 * 失败次数)
                current_window_size = int(search_window_size * (1 + window_expansion_factor * min(consecutive_failures, 2)))
                logger.debug(f"  -> 窗口自适应: 基础{search_window_size} -> 扩展{current_window_size} (连续失败{consecutive_failures}次)")

            # 计算窗口范围：允许一定回退，但主要向前推进
            # 回退机制: 如果上一个匹配在当前位置之后，允许向前回溯5个位置
            if last_matched_idx >= 0:
                window_start_idx = max(0, min(last_matched_idx - window_backtrack, window_start_idx))

            # 滑动窗口：只在最近 current_window_size 个句子中搜索
            search_end = min(window_start_idx + current_window_size, len(available_sentences))
            search_indices = list(range(window_start_idx, search_end))

            logger.debug(f"[{i+1}/{len(target_sentences)}] Target: '{target_sent[:15]}...' | Window: [{window_start_idx}-{search_end}] (size={current_window_size}) | LastIdx: {last_matched_idx}")
            
            for local_idx in search_indices:
                idx = local_idx
                sent = available_sentences[idx]
                
                # 跳过已使用的句子
                if idx in used_sentence_indices:
                    continue
                
                # 计算基础相似度
                raw_sim = self.text_similarity(sent["text"], target_sent)
                
                # ========== 计算顺序奖励 ==========
                order_bonus_score = 0.0
                if last_matched_idx >= 0:
                    # 索引距离惩罚：离预期的下一句越远，扣分越多
                    expected_idx = last_matched_idx + 1
                    index_distance = abs(idx - expected_idx)
                    index_penalty = index_penalty_factor * index_distance
                    
                    # 同一素材 + 时间相邻 = 额外奖励
                    time_bonus = 0.0
                    if last_matched_idx < len(available_sentences):
                        last_sent = available_sentences[last_matched_idx]
                        if (sent["material_index"] == last_sent["material_index"] and
                            not sent.get("missing", False) and
                            not last_sent.get("missing", False)):
                            time_gap = sent["start"] - last_sent["end"]
                            if -0.5 <= time_gap <= max_time_gap:
                                time_bonus = order_bonus * (1.0 - min(time_gap / max_time_gap, 1.0))
                    
                    order_bonus_score = time_bonus
                else:
                    index_penalty = 0
                
                # 最终分数 = 基础相似度 + 顺序奖励 - 索引距离惩罚
                final_score = raw_sim + order_bonus_score - index_penalty
                
                # 记录候选信息用于调试
                all_candidates.append({
                    "idx": idx,
                    "sim": raw_sim,
                    "bonus": order_bonus_score,
                    "penalty": index_penalty if last_matched_idx >= 0 else 0,
                    "final": final_score,
                    "text": sent["text"][:20]
                })
                
                # 找最高最终分数
                if final_score > best_final_score:
                    best_final_score = final_score
                    best_raw_similarity = raw_sim
                    best_match = {
                        "material_index": sent["material_index"],
                        "start": sent["start"],
                        "end": sent["end"],
                        "text": sent["text"],
                        "similarity": raw_sim,
                        "final_score": final_score,
                        "order_bonus": order_bonus_score,
                        "index_penalty": index_penalty if last_matched_idx >= 0 else 0
                    }
                    best_idx = idx
            
            # 调试日志：打印前3名候选
            if all_candidates:
                sorted_cands = sorted(all_candidates, key=lambda x: x["final"], reverse=True)[:3]
                debug_str = " | ".join([
                    f"idx{c['idx']}(sim{c['sim']:.2f}->final{c['final']:.2f})" 
                    for c in sorted_cands
                ])
                logger.debug(f"  Candidates: {debug_str}")
            
            # 如果找到匹配且达到阈值（使用原始相似度判断）
            if best_match and best_raw_similarity >= threshold:
                matched.append(best_match)
                used_sentence_indices.add(best_idx)
                last_matched_idx = best_idx

                # 更新滑动窗口起点，实现"推着走"的效果
                window_start_idx = best_idx + 1

                # 重置连续失败计数
                consecutive_failures = 0

                logger.debug(f"  -> WINNER: idx{best_idx} sim={best_raw_similarity:.3f} final={best_final_score:.3f}")
            else:
                # 未找到匹配，记录为缺失
                matched.append({
                    "material_index": -1,
                    "start": 0,
                    "end": 0,
                    "text": target_sent,
                    "similarity": 0,
                    "missing": True
                })
                last_matched_idx = -1

                # 连续失败处理
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    # 连续失败达到阈值，强制推进窗口(避免卡死)
                    logger.warning(f"  -> 连续失败{consecutive_failures}次，强制推进窗口")
                    window_start_idx = min(window_start_idx + search_window_size // 2, len(available_sentences))
                    consecutive_failures = 0  # 重置计数
        
        # 合并连续来自同一素材的片段
        merged = self._merge_consecutive_segments(matched)

        return merged
    
    def _merge_consecutive_segments(self, segments: List[Dict]) -> List[Dict]:
        """合并连续来自同一素材的片段"""
        if not segments:
            return []

        merged = [segments[0]]

        for seg in segments[1:]:
            last = merged[-1]

            # 检查是否可以合并
            if (seg["material_index"] == last["material_index"] and
                seg.get("missing", False) == last.get("missing", False) and
                not seg.get("missing", False)):
                # 合并
                merged[-1] = {
                    "material_index": last["material_index"],
                    "start": last["start"],
                    "end": seg["end"],
                    "text": last["text"] + " " + seg["text"],
                    "similarity": (last["similarity"] + seg["similarity"]) / 2
                }
            else:
                merged.append(seg)

        return merged

    def _calculate_adaptive_window_size(self,
                                       base_size: int,
                                       target_sentences_count: int,
                                       available_sentences_count: int,
                                       consecutive_failures: int = 0) -> int:
        """
        根据上下文动态调整窗口大小

        Args:
            base_size: 基础窗口大小
            target_sentences_count: 目标文案句子总数
            available_sentences_count: 可用素材句子总数
            consecutive_failures: 连续匹配失败次数

        Returns:
            调整后的窗口大小
        """
        # 场景1: 超长脚本(>100句) → 扩大窗口
        if target_sentences_count > 100:
            scaled_size = int(base_size * 1.5)
            logger.debug(f"  -> 超长脚本(>100句): 窗口 {base_size} -> {scaled_size}")
            return min(scaled_size, available_sentences_count)

        # 场景2: 重复录制(同句多次) → 缩小窗口
        if available_sentences_count > target_sentences_count * 3:
            # 素材数量 > 文案数量的3倍,说明有大量重复录制
            scaled_size = int(base_size * 0.7)
            logger.debug(f"  -> 重复录制: 窗口 {base_size} -> {scaled_size}")
            return max(scaled_size, 10)  # 最小10

        # 场景3: 连续失败 → 扩大窗口
        if consecutive_failures > 0:
            expansion_factor = 1 + 0.5 * min(consecutive_failures, 3)  # 最多2.5倍
            scaled_size = int(base_size * expansion_factor)
            logger.debug(f"  -> 连续失败{consecutive_failures}次: 窗口 {base_size} -> {scaled_size}")
            return min(scaled_size, available_sentences_count)

        # 默认返回基础大小
        return base_size

    def _optimize_window_position(self,
                                 window_start: int,
                                 last_matched_idx: int,
                                 window_size: int,
                                 total_sentences: int,
                                 window_backtrack: int = 5) -> int:
        """
        优化窗口起始位置，允许适度回退但强制向前推进

        Args:
            window_start: 当前窗口起始位置
            last_matched_idx: 上一个匹配的索引
            window_size: 窗口大小
            total_sentences: 总句子数
            window_backtrack: 允许回退的距离

        Returns:
            优化后的窗口起始位置
        """
        if last_matched_idx < 0:
            # 首次匹配，从0开始
            return 0

        # 计算预期窗口起始位置
        expected_start = last_matched_idx + 1

        # 允许适度回退(回退范围: window_backtrack)
        backtrack_start = max(0, last_matched_idx - window_backtrack)

        # 选择较大的值: 当前窗口 vs 回退位置 vs 预期位置
        # 确保窗口不会倒退太多，同时允许一定的灵活性
        optimized_start = max(expected_start, min(window_start, backtrack_start))

        # 边界检查
        return min(optimized_start, max(0, total_sentences - window_size))

    def _should_expand_window(self,
                             current_size: int,
                             base_size: int,
                             consecutive_failures: int,
                             max_failures: int = 3) -> bool:
        """
        判断是否应该扩大窗口

        Args:
            current_size: 当前窗口大小
            base_size: 基础窗口大小
            consecutive_failures: 连续失败次数
            max_failures: 最大允许失败次数

        Returns:
            是否扩大窗口
        """
        # 已经足够大，不需要再扩大
        if current_size >= base_size * 3:
            return False

        # 连续失败超过阈值，应该扩大
        if consecutive_failures >= max_failures:
            return True

        return False
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        简单分句
        
        Args:
            text: 待分句文本
            
        Returns:
            句子列表
        """
        # 使用正则分句
        sentences = re.split(r'[。！？；\n]', text)
        
        # 清理
        result = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                result.append(sent)
        
        return result
    
    def batch_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """
        批量计算相似度
        
        Args:
            texts1: 文本列表1
            texts2: 文本列表2
            
        Returns:
            相似度矩阵
        """
        if self.model is None:
            # 降级
            return np.array([[self.text_similarity(t1, t2) for t2 in texts2] for t1 in texts1])
        
        embeddings1 = self.model.encode(texts1)
        embeddings2 = self.model.encode(texts2)
        
        # 计算余弦相似度矩阵
        similarities = np.inner(embeddings1, embeddings2) / (
            np.outer(np.linalg.norm(embeddings1, axis=1), np.linalg.norm(embeddings2, axis=1))
        )
        
        return similarities
