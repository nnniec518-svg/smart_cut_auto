"""
语音识别模块
使用FunASR进行语音识别，支持多线程并行处理
"""
import os
import time
import torch
import numpy as np
import logging
import threading
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

logger = logging.getLogger("smart_cut")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class ASR:
    """语音识别器 - 使用FunASR + DirectML"""
    
    def __init__(self, model_size: str = "paraformer-zh", device: Optional[str] = None):
        """
        初始化ASR
        
        Args:
            model_size: FunASR模型名称
            device: 运行设备，None则自动选择 DirectML
        """
        self.model_size = model_size
        
        # FunASR 强制使用 CPU 运行，避免 DirectML 兼容性问题
        # BGE-M3 依然使用 DirectML 加速
        if device is None:
            self.device = "cpu"
        else:
            self.device = device
            
        self.model = None
        self._model_lock = threading.Lock()  # 线程锁，保护模型调用
        self._ensure_models_dir()
        self._load_model()
        logger.info(f"ASR initialized with FunASR model: {model_size}, device: {self.device}")
    
    def _ensure_models_dir(self) -> None:
        """确保模型目录存在"""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self) -> None:
        """加载FunASR模型"""
        try:
            from funasr import AutoModel
            
            # 优先使用本地缓存的模型
            local_model_path = Path(r"C:\Users\nnniec\.cache\modelscope\hub\Models\iic\speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
            
            if local_model_path.exists():
                logger.info(f"Loading local FunASR model from: {local_model_path}")
                self.model = AutoModel(
                    model=str(local_model_path),
                    device=self.device,
                    disable_pbar=True
                )
            else:
                # 使用Paraformer中文模型，支持VAD和标点
                model_name = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
                
                logger.info(f"Loading FunASR model: {model_name}")
                
                self.model = AutoModel(
                    model=model_name,
                    model_revision="v2.0.4",
                    device=self.device,
                    disable_pbar=True,
                    hub="ms",
                    disable_update=True,
                    cache_dir=str(MODELS_DIR / "funasr")
                )
            
            logger.info(f"FunASR model loaded successfully")
            
        except Exception as e:
            logger.error(f"加载FunASR模型失败: {e}")
            raise
    
    def transcribe(self, audio_path: Union[str, Path], 
                   word_timestamps: bool = True,
                   language: str = "zh",
                   verbose: Optional[bool] = None) -> Dict[str, Any]:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            word_timestamps: 是否返回单词级时间戳
            language: 语言，None则自动检测
            verbose: 是否输出详细信息
            
        Returns:
            转录结果字典
        """
        try:
            # 使用锁保护多线程环境下的模型调用
            with self._model_lock:
                # FunASR直接处理文件路径
                result = self.model.generate(
                    input=str(audio_path),
                    batch_size_s=300,
                    hotword=""
                )
            
            # 调试：打印原始结果
            logger.info(f"ASR raw result for {audio_path}: {result[:1] if isinstance(result, list) else result}")
            
            # 格式化结果
            formatted = self._format_result(result)
            
            logger.debug(f"Transcribed {audio_path}: {len(formatted.get('segments', []))} segments")
            return formatted
            
        except Exception as e:
            logger.error(f"转录失败: {audio_path}, {e}")
            raise
    
    def transcribe_audio(self, audio: np.ndarray, 
                        sample_rate: int = 16000,
                        word_timestamps: bool = True,
                        language: str = "zh") -> Dict[str, Any]:
        """
        转录音频数组
        
        Args:
            audio: 音频数组
            sample_rate: 采样率
            word_timestamps: 是否返回单词级时间戳
            language: 语言
            
        Returns:
            转录结果字典
        """
        try:
            # 检查音频长度
            min_samples = int(0.1 * sample_rate)  # 至少0.1秒
            if len(audio) < min_samples:
                logger.debug(f"Audio too short: {len(audio)} samples, need at least {min_samples}")
                return {"text": "", "segments": [], "language": language}
            
            # FunASR需要音频文件，创建临时文件
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            # 确保音频是float32且在[-1,1]范围内
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            sf.write(temp_path, audio, sample_rate)
            
            try:
                result = self.transcribe(temp_path, word_timestamps, language)
            finally:
                # 清理临时文件
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass
            
            return result
            
        except Exception as e:
            logger.error(f"转录音频数组失败: {e}")
            raise
    
    def transcribe_segment(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        转录音频片段（便捷方法）
        
        Args:
            audio: 音频数组
            sr: 采样率
            
        Returns:
            转录结果
        """
        return self.transcribe_audio(audio, sr)
    
    def _format_result(self, result: Any) -> Dict[str, Any]:
        """
        格式化FunASR输出结果
        
        Returns:
            格式化后的结果字典
        """
        # FunASR返回格式可能是列表嵌套
        # result: [[{"text": "...", "timestamp": ...}], ...]
        
        try:
            # 提取实际结果
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    result = result[0]
                
                if len(result) > 0 and isinstance(result[0], dict):
                    item = result[0]
                else:
                    return {"text": "", "segments": [], "language": "zh"}
            else:
                return {"text": "", "segments": [], "language": "zh"}
            
            # FunASR返回格式解析
            text = item.get("text", "")
            
            # 解析时间戳
            # FunASR返回的timestamp格式: (start, end) 元组列表，单位是毫秒
            timestamp = item.get("timestamp", [])
            
            segments = []
            
            if timestamp and len(timestamp) > 0:
                # 将时间戳对转换为segments
                # FunASR返回的时间戳单位是毫秒，需要转换为秒
                # timestamp格式: [(start1, end1), (start2, end2), ...]
                # 每个元组对应一个词/字，所有元组对应完整的句子
                
                # 按时间顺序合并相邻的时间戳，形成连续的片段
                # 设定合并阈值：小于0.5秒的间隔认为是同一句话
                merge_threshold = 0.5  # 秒
                
                if len(timestamp) > 0:
                    # 第一个segment
                    current_start = timestamp[0][0] / 1000.0
                    current_end = timestamp[0][1] / 1000.0
                    
                    for i in range(1, len(timestamp)):
                        ts_start = timestamp[i][0] / 1000.0
                        ts_end = timestamp[i][1] / 1000.0
                        
                        # 如果与当前片段间隔很小，合并
                        if ts_start - current_end < merge_threshold:
                            current_end = ts_end
                        else:
                            # 保存当前片段
                            segments.append({
                                "text": text.strip(),
                                "start": current_start,
                                "end": current_end,
                            })
                            # 开始新片段
                            current_start = ts_start
                            current_end = ts_end
                    
                    # 保存最后一个片段
                    segments.append({
                        "text": text.strip(),
                        "start": current_start,
                        "end": current_end,
                    })
            
            if not segments:
                # 没有时间戳，返回整个文本作为一个segment
                segments = [{
                    "text": text.strip(),
                    "start": 0,
                    "end": 0,
                }]
            
            return {
                "text": text.strip(),
                "segments": segments,
                "language": "zh"
            }
            
        except Exception as e:
            logger.warning(f"格式化FunASR结果失败: {e}")
            return {"text": "", "segments": [], "language": "zh"}
    
    def get_sentences(self, asr_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从ASR结果中提取句子
        
        Returns:
            句子列表，每个包含text, start, end
        """
        return asr_result.get("segments", [])
    
    def split_long_audio(self, audio_path: Union[str, Path], 
                         max_duration: int = 600,
                         overlap: float = 0.5) -> List[Dict[str, Any]]:
        """
        对长音频分段处理
        
        Args:
            audio_path: 音频路径
            max_duration: 每段最大时长（秒）
            overlap: 相邻段重叠时长（秒）
            
        Returns:
            合并后的转录结果
        """
        from core.audio_processor import AudioProcessor
        
        processor = AudioProcessor()
        audio, sr = processor.load_audio(audio_path)
        duration = len(audio) / sr
        
        if duration <= max_duration:
            return self.transcribe(audio_path)
        
        # 分段处理
        segments_result = {
            "text": "",
            "segments": [],
            "language": "zh"
        }
        
        import soundfile as sf
        import tempfile
        
        step = max_duration - overlap
        num_segments = int(np.ceil((duration - max_duration) / step)) + 1
        
        for i in range(num_segments):
            start_time = i * step
            end_time = min(start_time + max_duration, duration)
            
            # 提取片段
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # 保存为临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            try:
                sf.write(temp_path, segment_audio, sr)
                result = self.transcribe(temp_path, word_timestamps=True)
                
                # 调整时间戳
                for seg in result["segments"]:
                    seg["start"] += start_time
                    seg["end"] += start_time
                    segments_result["segments"].append(seg)
                
                segments_result["text"] += result["text"] + " "
            finally:
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass
        
        segments_result["text"] = segments_result["text"].strip()
        
        return segments_result
    
    @staticmethod
    def supported_models() -> List[str]:
        """返回支持的模型列表"""
        return ["paraformer-zh", "paraformer-en", "sensevoice"]
    
    @property
    def model_name(self) -> str:
        return self.model_size
