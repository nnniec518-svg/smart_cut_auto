"""
工具函数模块
提供配置读写、日志设置、文件操作等通用功能
"""
import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
TEMP_DIR = PROJECT_ROOT / "temp"
LOG_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
CONFIG_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def _import_moviepy():
    """兼容新旧版本MoviePy的导入"""
    try:
        # MoviePy v2.0+ 使用新导入方式
        from moviepy import VideoFileClip
        return VideoFileClip
    except ImportError:
        # MoviePy v1.x 使用旧导入方式
        from moviepy.editor import VideoFileClip
        return VideoFileClip

DEFAULT_CONFIG = {
    "similarity_threshold": 0.6,
    "single_threshold": 0.85,
    "silence_threshold": 1.5,
    "noise_reduce_strength": 0.3,
    "remove_leading_sound": True,
    "remove_trailing_silence": True,
    "normalize_audio": True,
    "subtitle_font": "Microsoft YaHei",
    "subtitle_size_ratio": 0.07,
    "subtitle_color": "#FFFFFF",
    "subtitle_background": True,
    "subtitle_stroke_color": "#000000",
    "subtitle_stroke_width": 1.5,
    "subtitle_margin": 10,
    "video_resolution": {"width": 1080, "height": 1920},
    "video_fps": 30,
    "fade_duration": 0.5,
    "whisper_model": "large",
    "vad_model": "silero",
    "embedding_model": "all-MiniLM-L6-v2"
}


def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    config_path = CONFIG_DIR / "settings.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 合并默认配置
            merged = DEFAULT_CONFIG.copy()
            merged.update(config)
            return merged
        except Exception as e:
            logging.warning(f"加载配置文件失败: {e}, 使用默认配置")
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]) -> bool:
    """保存配置文件"""
    config_path = CONFIG_DIR / "settings.json"
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"保存配置文件失败: {e}")
        return False


def setup_logger(name: str = "smart_cut", level: int = logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # 文件日志
        log_file = LOG_DIR / f"{name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        # 控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def ensure_dir(path: Path) -> None:
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_temp_path(filename: str) -> Path:
    """获取临时文件路径"""
    return TEMP_DIR / filename


def clean_temp_files(keep_patterns: Optional[List[str]] = None) -> None:
    """清理临时文件"""
    if keep_patterns is None:
        keep_patterns = []
    
    for file in TEMP_DIR.iterdir():
        if file.is_file():
            # 检查是否匹配保留模式
            if any(pattern in file.name for pattern in keep_patterns):
                continue
            try:
                file.unlink()
            except Exception as e:
                logging.warning(f"删除临时文件失败: {file}, {e}")


def check_ffmpeg() -> bool:
    """检查 FFmpeg 是否可用"""
    import shutil
    return shutil.which("ffmpeg") is not None


def run_ffmpeg(cmd: List[str], timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
    """
    运行FFmpeg命令
    
    Args:
        cmd: FFmpeg命令列表
        timeout: 超时时间（秒）
        check: 是否检查返回码
        
    Returns:
        CompletedProcess对象
    """
    logger = logging.getLogger("smart_cut")
    
    # 检查 FFmpeg 是否可用
    if not check_ffmpeg():
        logger.error("FFmpeg not found in system PATH. Please install FFmpeg.")
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to PATH.")
    
    logger.debug(f"Running FFmpeg: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check
        )
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timeout after {timeout}s")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr}")
        raise


def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    获取视频信息
    
    Returns:
        包含resolution, fps, duration, codec等信息
    """
    # 优先使用 FFprobe
    if check_ffmpeg():
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            data = json.loads(result.stdout)
            
            video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
            audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
            
            return {
                "duration": float(data.get("format", {}).get("duration", 0)),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                "video_codec": video_stream.get("codec_name", ""),
                "audio_codec": audio_stream.get("codec_name", ""),
                "bitrate": int(data.get("format", {}).get("bit_rate", 0))
            }
        except Exception as e:
            logging.warning(f"FFprobe failed: {e}")
    
    # 备用：使用 moviepy
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(str(video_path))
        info = {
            "duration": clip.duration,
            "width": clip.w,
            "height": clip.h,
            "fps": clip.fps,
            "video_codec": "h264",
            "audio_codec": "aac"
        }
        clip.close()
        return info
    except Exception as e:
        logging.error(f"获取视频信息失败: {e}")
        return {}


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    else:
        return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def validate_video_file(path: str) -> bool:
    """验证视频文件是否有效"""
    if not os.path.exists(path):
        return False
    
    # 检查文件扩展名
    valid_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    if Path(path).suffix.lower() not in valid_exts:
        return False
    
    # 尝试获取视频信息
    try:
        info = get_video_info(path)
        return info.get("width", 0) > 0 and info.get("height", 0) > 0
    except Exception:
        return False


def validate_audio_file(path: str) -> bool:
    """验证音频文件是否有效"""
    if not os.path.exists(path):
        return False
    
    valid_exts = ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']
    return Path(path).suffix.lower() in valid_exts
