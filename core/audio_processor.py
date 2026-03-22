"""
音频处理模块
提供VAD分割、降噪、提示音切除、静音切除等功能
"""
import os
import torch
import numpy as np
import librosa
import logging
import threading
from typing import Tuple, List, Optional, Union
from pathlib import Path

try:
    import noisereduce as nr
except ImportError:
    nr = None

logger = logging.getLogger("smart_cut")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, vad_model: Optional[str] = None):
        """
        初始化音频处理器
        
        Args:
            vad_model: VAD模型路径，None则从torch.hub加载Silero VAD
        """
        self.vad_model = None
        self.vad_utils = None  # 存储 VAD 工具函数
        self._ensure_models_dir()
        self._load_vad_model(vad_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 线程锁，保护多线程环境下的VAD调用
        self._vad_lock = threading.Lock()
        logger.info(f"AudioProcessor initialized, device: {self.device}")
    
    def _ensure_models_dir(self) -> None:
        """确保模型目录存在"""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # 设置torch hub缓存路径
        os.environ["TORCH_HOME"] = str(MODELS_DIR / "torch")
        os.environ["TORCH_HUB_DIR"] = str(MODELS_DIR / "torch_hub")
    
    def _get_vad_model_path(self) -> Path:
        """获取VAD模型本地路径"""
        return MODELS_DIR / "silero_vad.pt"
    
    def _load_vad_model(self, model_path: Optional[str] = None, force_reload: bool = False) -> None:
        """加载VAD模型"""
        try:
            if model_path and Path(model_path).exists():
                self.vad_model = torch.jit.load(model_path)
                logger.info(f"Loaded VAD model from local path: {model_path}")
                return

            # 检查本地模型是否存在
            local_model = self._get_vad_model_path()

            # 设置torch hub缓存
            torch.hub.set_dir(str(MODELS_DIR / "torch_hub"))

            # 强制重新下载（解决缓存损坏问题）
            logger.info(f"Loading Silero VAD from torch.hub (force_reload={force_reload})...")

            # 从torch.hub加载Silero VAD
            self.vad_model, self.vad_utils = torch.hub.load(
                "snakers4/silero-vad",
                model="silero_vad",
                force_reload=force_reload
            )

            # 打印 vad_utils 的属性以便调试
            logger.info(f"VAD utils loaded: {dir(self.vad_utils)}")

            # 检查并复制到本地目录以便管理
            hub_dir = Path(torch.hub.get_dir())
            if hub_dir.exists():
                for f in hub_dir.glob("**/silero_vad*"):
                    if f.suffix in ['.pt', '.jit']:
                        import shutil
                        shutil.copy(f, local_model)
                        break

            logger.info("Silero VAD model loaded successfully")

        except Exception as e:
            logger.error(f"加载VAD模型失败: {e}")
            logger.warning("将降级使用能量检测VAD，但建议重新安装 silero-vad")
            # 降级使用简单的能量检测
            self.vad_model = None
            self.vad_utils = None
    
    def load_audio(self, file_path: Union[str, Path], target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            target_sr: 目标采样率
            
        Returns:
            (audio_array, sample_rate)
        """
        try:
            audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
            logger.debug(f"Loaded audio: {file_path}, duration: {len(audio)/sr:.2f}s")
            return audio, sr
        except Exception as e:
            logger.error(f"加载音频失败: {file_path}, {e}")
            raise
    
    def load_audio_raw(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        加载原始音频（不重采样）
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            (audio_array, sample_rate)
        """
        try:
            # 使用soundfile或audioread
            import soundfile as sf
            audio, sr = sf.read(str(file_path), dtype='float32')
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # 转单声道
            return audio, sr
        except Exception as e:
            logger.error(f"加载原始音频失败: {file_path}, {e}")
            raise
    
    def get_speech_segments(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """
        使用VAD获取语音段
        
        Args:
            audio: 音频数组
            sr: 采样率
            
        Returns:
            语音段列表 [(start_sec, end_sec), ...]
        """
        if self.vad_model is None:
            # 使用简单的能量检测
            return self._energy_vad(audio, sr)
        
        try:
            # 尝试使用 Silero VAD 的 get_speech_timestamps
            audio = audio.astype(np.float32)
            
            # 转换为tensor
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # 获取语音段 - 使用锁保护多线程环境下的VAD调用
            self.vad_model.eval()
            speech_timestamps = []
            
            with torch.no_grad():
                # 方法1: 尝试直接导入 silero_vad 的 get_speech_timestamps
                try:
                    from silero_vad import get_speech_timestamps
                    with self._vad_lock:
                        speech_timestamps = get_speech_timestamps(
                            audio_tensor,
                            self.vad_model,
                            sampling_rate=sr
                        )
                except ImportError:
                    # silero_vad 包未安装，尝试使用模型内置方法
                    try:
                        # Silero VAD 模型内置方法
                        with self._vad_lock:
                            speech_timestamps = self.vad_model(audio_tensor)
                            # 转换为列表格式
                            if hasattr(speech_timestamps, 'cpu'):
                                speech_timestamps = speech_timestamps.cpu().numpy()
                    except Exception as e2:
                        logger.warning(f"VAD 方法1失败: {e2}")
                        speech_timestamps = []

                # 方法2: 如果方法1失败，尝试使用 model.forward
                if not speech_timestamps:
                    try:
                        with self._vad_lock:
                            # 尝试使用模型直接推理
                            speech_timestamps = self.vad_model(audio_tensor)
                            if hasattr(speech_timestamps, 'cpu'):
                                speech_timestamps = speech_timestamps.cpu().numpy()
                    except Exception as e3:
                        logger.warning(f"VAD 方法2失败: {e3}")
            
            if speech_timestamps:
                # 处理不同格式的输出
                segments = []
                try:
                    # 格式1: 字典列表 [{"start": ..., "end": ...}, ...]
                    if isinstance(speech_timestamps, (list, tuple)) and len(speech_timestamps) > 0:
                        if isinstance(speech_timestamps[0], dict):
                            segments = [(s["start"] / sr, s["end"] / sr) for s in speech_timestamps]
                        # 格式2: numpy 数组或 tensor
                        elif hasattr(speech_timestamps, '__iter__'):
                            # 尝试直接使用
                            for s in speech_timestamps:
                                if hasattr(s, 'start') and hasattr(s, 'end'):
                                    segments.append((s.start / sr, s.end / sr))
                except Exception as e:
                    logger.warning(f"VAD 结果解析失败: {e}")

                if segments:
                    logger.debug(f"VAD detected {len(segments)} speech segments")
                    return segments
            
            # 所有方法都失败，使用能量检测
            logger.warning("Silero VAD 不可用，降级使用能量检测VAD")
            raise Exception("Silero VAD 不可用")
        
        except Exception as e:
            logger.warning(f"VAD处理失败: {e}，降级使用能量检测")
            return self._energy_vad(audio, sr)
    
    def _energy_vad(self, audio: np.ndarray, sr: int, threshold: float = 0.005) -> List[Tuple[float, float]]:
        """简单的能量检测VAD"""
        # 计算短时能量
        frame_length = int(0.025 * sr)  # 25ms帧
        hop_length = int(0.010 * sr)    # 10ms hop
        
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2) 
            for i in range(0, len(audio) - frame_length, hop_length)
        ])
        
        # 归一化
        if energy.max() > 0:
            energy = energy / energy.max()
        
        # 阈值判断 - 使用更低的阈值
        is_speech = energy > threshold
        
        # 合并连续语音段
        segments = []
        start = None
        
        for i, speech in enumerate(is_speech):
            time = i * hop_length / sr
            if speech and start is None:
                start = time
            elif not speech and start is not None:
                segments.append((start, time))
                start = None
        
        if start is not None:
            segments.append((start, len(audio) / sr))
        
        # 如果没有检测到任何语音段，返回整个音频作为一个片段
        if not segments:
            logger.warning("能量检测未找到语音段，使用整个音频")
            return [(0.0, len(audio) / sr)]
        
        return segments
    
    def merge_segments(self, segments: List[Tuple[float, float]], silence_threshold: float = 1.5) -> List[Tuple[float, float]]:
        """
        根据静音间隔合并语音段
        
        Args:
            segments: 原始语音段列表
            silence_threshold: 静音阈值（秒）
            
        Returns:
            合并后的片段列表
        """
        if not segments:
            return []
        
        if len(segments) == 1:
            return segments
        
        merged = [segments[0]]
        
        for start, end in segments[1:]:
            last_start, last_end = merged[-1]
            
            if start - last_end < silence_threshold:
                # 合并
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        
        logger.debug(f"Merged {len(segments)} segments into {len(merged)} segments")
        return merged
    
    def trim_silence(self, audio: np.ndarray, sr: int, 
                     start_offset: float = 0.0, 
                     end_offset: float = 0.0) -> Tuple[np.ndarray, float, float]:
        """
        切除开头和结尾静音
        
        Args:
            audio: 音频数组
            sr: 采样率
            start_offset: 开头额外保留（秒）
            end_offset: 结尾额外保留（秒）
            
        Returns:
            (裁剪后的音频, 实际开始时间, 实际结束时间)
        """
        segments = self.get_speech_segments(audio, sr)
        
        if not segments:
            return audio, 0.0, len(audio) / sr
        
        # 开头
        actual_start = segments[0][0] + start_offset
        start_sample = max(0, int(actual_start * sr))
        
        # 结尾
        actual_end = segments[-1][1] - end_offset
        end_sample = min(len(audio), int(actual_end * sr))
        
        trimmed = audio[start_sample:end_sample]
        
        return trimmed, actual_start, actual_end
    
    def remove_leading_sound(self, audio: np.ndarray, sr: int, max_duration: float = 2.0, 
                              skip_short_speech: float = 0.8) -> Tuple[np.ndarray, float]:
        """
        切除开头的非语音部分（提示音等）
        
        Args:
            audio: 音频数组
            sr: 采样率
            max_duration: 最大切除时长（秒）
            skip_short_speech: 跳过开头短于这个时长的语音（秒），用于跳过"走"等提示词
            
        Returns:
            (裁剪后的音频, 实际切除时长)
        """
        segments = self.get_speech_segments(audio, sr)
        
        if not segments:
            return audio, 0.0
        
        # 第一个语音段的开始时间和时长
        first_speech_start = segments[0][0]
        first_speech_end = segments[0][1]
        first_speech_duration = first_speech_end - first_speech_start
        
        # 如果第一个语音段非常短（可能是提示词如"走"、"一走"），则跳过它
        if first_speech_duration < skip_short_speech and first_speech_end < max_duration:
            # 跳过这个短语音段
            logger.info(f"Skipping short leading speech: {first_speech_duration:.2f}s < {skip_short_speech}s")
            remove_duration = first_speech_end  # 切除到第一个短语音段结束
        else:
            # 计算实际切除时长
            remove_duration = min(first_speech_start, max_duration)
        
        if remove_duration > 0.1:  # 只有超过0.1秒才切除
            start_sample = int(remove_duration * sr)
            trimmed = audio[start_sample:]
            logger.info(f"Removed leading sound: {remove_duration:.2f}s")
            return trimmed, remove_duration
        
        return audio, 0.0
    
    def reduce_noise(self, audio: np.ndarray, sr: int, prop_decrease: float = 0.3) -> np.ndarray:
        """
        降噪处理
        
        Args:
            audio: 音频数组
            sr: 采样率
            prop_decrease: 降噪强度 (0-1)
            
        Returns:
            降噪后的音频
        """
        if nr is None:
            logger.warning("noisereduce未安装，跳过降噪")
            return audio
        
        try:
            # 降噪需要合适的分帧参数
            # noisereduce使用固定的2048帧
            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                prop_decrease=prop_decrease,
                stationary=False,
                n_fft=2048,
                hop_length=512
            )
            logger.debug(f"Noise reduction applied, strength: {prop_decrease}")
            return reduced
        except Exception as e:
            logger.error(f"降噪失败: {e}")
            return audio
    
    def normalize_volume(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        音量归一化
        
        Args:
            audio: 音频数组
            target_db: 目标分贝值
            
        Returns:
            归一化后的音频
        """
        # 计算当前RMS
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-10:
            return audio
        
        # 转换为dB
        current_db = 20 * np.log10(rms)
        
        # 计算需要调整的dB值
        adjustment_db = target_db - current_db
        
        # 转换为线性增益
        gain = 10 ** (adjustment_db / 20)
        
        normalized = audio * gain
        
        # 防止clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 0.95:
            normalized = normalized * (0.95 / max_val)
        
        return normalized
    
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        从视频提取音频
        
        Args:
            video_path: 视频路径
            output_path: 输出音频路径，None则使用临时文件
            
        Returns:
            提取的音频文件路径
        """
        from core.utils import get_temp_path
        
        if output_path is None:
            output_path = get_temp_path(f"audio_{Path(video_path).stem}.wav")
        
        # 首先尝试使用ffmpeg提取音频
        import subprocess
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info(f"Extracted audio (ffmpeg) to: {output_path}")
                return str(output_path)
            else:
                logger.warning(f"ffmpeg提取音频失败: {result.stderr}")
        except Exception as e:
            logger.warning(f"ffmpeg提取音频异常: {e}")
        
        # 备用方案：使用 librosa 直接加载视频
        try:
            import soundfile as sf
            audio, sr = librosa.load(str(video_path), sr=16000, mono=True)
            sf.write(str(output_path), audio, 16000)
            logger.info(f"Extracted audio (librosa fallback) to: {output_path}")
            return str(output_path)
        except Exception as e2:
            logger.error(f"提取音频失败: {e2}")
            raise
    
    def get_audio_duration(self, audio: np.ndarray, sr: int) -> float:
        """获取音频时长（秒）"""
        return len(audio) / sr
    
    def get_speech_duration(self, audio: np.ndarray, sr: int) -> float:
        """获取语音段总时长"""
        segments = self.get_speech_segments(audio, sr)
        return sum(end - start for start, end in segments)
