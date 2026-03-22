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

# B-Roll 关键词 - 匹配这些关键词时优先从 B_ROLL 素材召回
B_ROLL_KEYWORDS = {
    '空镜头', '空镜', '风景', '景色', '环境', '画面', '场景',
    '外景', '室内', '办公室', '产品', '商品', '展示', '演示',
    '背景', '氛围', '街头', '道路', '建筑', '自然', '城市',
    '乡村', '海边', '山景', '花', '草', '树', '天空', '云',
    '日出', '日落', '夜景', '灯光', '特写', '远景', '全景'
}

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class Matcher:
    """文案匹配器"""
    
    def __init__(self, similarity_model_name: str = "all-MiniLM-L6-v2"):
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
            
            if local_model_path.exists():
                # 使用本地模型
                logger.info(f"Loading local model from: {local_model_path}")
                self.model = SentenceTransformer(str(local_model_path))
            else:
                # 使用缓存目录
                cache_folder = str(MODELS_DIR / "sentence_transformers")
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=cache_folder
                )
                logger.info(f"Model '{self.model_name}' loaded from cache")
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
    
    def requires_b_roll(self, text: str) -> bool:
        """
        检测文案是否需要 B-Roll (空镜头)
        
        Args:
            text: 目标文案句子
            
        Returns:
            是否需要 B-Roll
        """
        # 预处理
        text = text.replace(' ', '').lower()
        
        # 检查是否包含 B-Roll 关键词
        for keyword in B_ROLL_KEYWORDS:
            if keyword in text:
                logger.info(f"检测到 B-Roll 关键词: {keyword}")
                return True
        
        return False
    
    def match_b_roll(self, 
                     b_roll_assets: List[Dict],
                     target_text: str,
                     similarity_threshold: float = 0.1) -> Optional[Dict]:
        """
        为 B-Roll 需求匹配素材
        
        Args:
            b_roll_assets: B-Roll 素材列表 (包含 path, name, asr_text 等)
            target_text: 目标文案
            similarity_threshold: 相似度阈值
            
        Returns:
            匹配的 B-Roll 素材信息或 None
        """
        if not b_roll_assets:
            return None
        
        # 预处理目标文案
        target_clean = target_text.replace(' ', '').lower()
        
        best_match = None
        best_similarity = 0.0
        
        for asset in b_roll_assets:
            # B-Roll 的 asr_text 可能为空或很短
            asset_text = asset.get('asr_text', '') or ''
            asset_text_clean = asset_text.replace(' ', '').lower()
            
            if not asset_text_clean:
                # 如果没有文本，给予一个基础分数（随机选择）
                similarity = 0.3
            else:
                # 计算相似度
                similarity = self.text_similarity(asset_text_clean, target_clean)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    "material_path": asset.get('path'),
                    "material_name": asset.get('name'),
                    "track_type": "B_ROLL",
                    "start": 0.0,  # B-Roll 通常从头开始
                    "end": asset.get('duration', 5.0),
                    "similarity": similarity,
                    "is_b_roll": True
                }
        
        if best_match and best_similarity >= similarity_threshold:
            logger.info(f"B-Roll 匹配成功: {best_match['material_name']}, 相似度: {best_similarity:.3f}")
            return best_match
        
        return None
    
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
                                   threshold: float = 0.6) -> List[Dict]:
        """
        无序句子匹配算法 - 从所有素材中找最匹配的，不考虑顺序
        
        Args:
            target_sentences: 目标文案句子列表
            available_sentences: 可用的素材句子列表
            threshold: 相似度阈值
            
        Returns:
            匹配结果列表
        """
        # 记录已使用的素材句子（避免重复使用）
        used_sentence_indices = set()
        
        matched = []
        
        for target_sent in target_sentences:
            best_match = None
            best_similarity = 0.0
            best_idx = -1
            
            # 遍历所有素材句子，找出最佳匹配
            for idx, sent in enumerate(available_sentences):
                # 跳过已使用的句子
                if idx in used_sentence_indices:
                    continue
                
                # 计算相似度
                sim = self.text_similarity(sent["text"], target_sent)
                
                # 找最高相似度
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = {
                        "material_index": sent["material_index"],
                        "start": sent["start"],
                        "end": sent["end"],
                        "text": sent["text"],
                        "similarity": sim
                    }
                    best_idx = idx
            
            # 如果找到匹配且达到阈值
            if best_match and best_similarity >= threshold:
                matched.append(best_match)
                used_sentence_indices.add(best_idx)
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
    
    def match_with_track_type(self,
                              target_text: str,
                              a_roll_materials: List[Dict],
                              b_roll_materials: List[Dict],
                              sentence_threshold: float = 0.6) -> List[Dict]:
        """
        带 track_type 感知的匹配 - 优先从对应类型的素材中召回
        
        Args:
            target_text: 目标文案全文
            a_roll_materials: A_ROLL 素材列表 (包含 path, asr_text, segments, duration 等)
            b_roll_materials: B_ROLL 素材列表
            sentence_threshold: 句子匹配阈值
            
        Returns:
            匹配结果列表，每个元素包含 material_index, start, end, track_type 等
        """
        # 分句
        target_sentences = self._split_sentences(target_text)
        
        if not target_sentences:
            return []
        
        matched = []
        
        # 准备 A_ROLL 句子
        a_roll_sentences = []
        for i, mat in enumerate(a_roll_materials):
            segments = mat.get('segments', [])
            for seg in segments:
                text = seg.get('text', '')
                if text:
                    a_roll_sentences.append({
                        "material_index": i,
                        "material_type": "A_ROLL",
                        "text": text,
                        "start": seg.get('start', 0),
                        "end": seg.get('end', 0),
                        "material": mat
                    })
        
        # 准备 B_ROLL 句子 (B-Roll 通常没有文本，使用素材本身信息)
        b_roll_sentences = []
        for i, mat in enumerate(b_roll_materials):
            b_roll_sentences.append({
                "material_index": i,
                "material_type": "B_ROLL",
                "text": mat.get('asr_text', '') or '',
                "start": 0.0,
                "end": mat.get('duration', 5.0),
                "material": mat
            })
        
        # 对每个目标句子进行匹配
        for target_sent in target_sentences:
            # 1. 首先检查是否需要 B-Roll
            if self.requires_b_roll(target_sent):
                # 优先从 B_ROLL 匹配
                best_match = self._find_best_match(target_sent, b_roll_sentences, sentence_threshold)
                
                if best_match:
                    matched.append({
                        "material_index": best_match["material_index"],
                        "material_type": "B_ROLL",
                        "start": best_match["start"],
                        "end": best_match["end"],
                        "text": best_match["text"],
                        "similarity": best_match["similarity"],
                        "is_b_roll": True,
                        "target_text": target_sent
                    })
                    continue
            
            # 2. 从 A_ROLL 匹配
            best_match = self._find_best_match(target_sent, a_roll_sentences, sentence_threshold)
            
            if best_match:
                matched.append({
                    "material_index": best_match["material_index"],
                    "material_type": "A_ROLL",
                    "start": best_match["start"],
                    "end": best_match["end"],
                    "text": best_match["text"],
                    "similarity": best_match["similarity"],
                    "is_b_roll": False,
                    "target_text": target_sent
                })
            else:
                # A_ROLL 没找到，尝试 B_ROLL
                best_match = self._find_best_match(target_sent, b_roll_sentences, sentence_threshold)
                
                if best_match:
                    matched.append({
                        "material_index": best_match["material_index"],
                        "material_type": "B_ROLL",
                        "start": best_match["start"],
                        "end": best_match["end"],
                        "text": best_match["text"],
                        "similarity": best_match["similarity"],
                        "is_b_roll": True,
                        "target_text": target_sent
                    })
                else:
                    # 没找到匹配
                    matched.append({
                        "material_index": -1,
                        "material_type": "UNKNOWN",
                        "start": 0,
                        "end": 0,
                        "text": target_sent,
                        "similarity": 0,
                        "missing": True,
                        "target_text": target_sent
                    })
        
        return matched
    
    def _find_best_match(self, 
                         target_sent: str, 
                         available_sentences: List[Dict],
                         threshold: float) -> Optional[Dict]:
        """
        从可用句子列表中找到最佳匹配
        
        Args:
            target_sent: 目标句子
            available_sentences: 可用句子列表
            threshold: 相似度阈值
            
        Returns:
            最佳匹配结果或 None
        """
        best_match = None
        best_similarity = 0.0
        
        for sent in available_sentences:
            sim = self.text_similarity(sent["text"], target_sent)
            
            if sim > best_similarity:
                best_similarity = sim
                best_match = {
                    "material_index": sent["material_index"],
                    "start": sent["start"],
                    "end": sent["end"],
                    "text": sent["text"],
                    "similarity": sim
                }
        
        if best_similarity >= threshold:
            return best_match
        
        return None
    
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
