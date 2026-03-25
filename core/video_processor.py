"""
视频处理模块
使用FFmpeg进行视频裁剪、拼接、转码
"""
import os
import subprocess
import tempfile
import threading
import time
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

logger = logging.getLogger("smart_cut")


def _run_ffmpeg(cmd: List[str], timeout: int = 600) -> Tuple[int, str, str]:
    """
    运行ffmpeg命令
    
    Args:
        cmd: ffmpeg命令列表
        timeout: 超时时间（秒）
        
    Returns:
        (返回码, stdout, stderr)
    """
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        process.kill()
        return -1, "", "Command timeout"
    except Exception as e:
        return -1, "", str(e)


def _parse_fraction(fraction_str: str, default: float = 0.0) -> float:
    """
    安全解析分数字符串（如 "30/1"）为浮点数

    Args:
        fraction_str: 分数字符串
        default: 解析失败时的默认值

    Returns:
        浮点数值
    """
    try:
        if '/' in fraction_str:
            numerator, denominator = fraction_str.split('/')
            return float(numerator) / float(denominator)
        return float(fraction_str)
    except (ValueError, ZeroDivisionError, AttributeError):
        return default


def _get_video_info_ffmpeg(video_path: str) -> Dict[str, Any]:
    """使用ffprobe获取视频信息"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)

        video_stream = None
        audio_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
            elif stream.get('codec_type') == 'audio':
                audio_stream = stream

        info = {
            "duration": float(data.get('format', {}).get('duration', 0)),
            "width": int(video_stream.get('width', 0)) if video_stream else 0,
            "height": int(video_stream.get('height', 0)) if video_stream else 0,
            "fps": _parse_fraction(video_stream.get('r_frame_rate', '0/1')) if video_stream else 0,
            "has_audio": audio_stream is not None,
            "video_codec": video_stream.get('codec_name', '') if video_stream else '',
            "audio_codec": audio_stream.get('codec_name', '') if audio_stream else '',
        }
        return info
    except Exception as e:
        logger.warning(f"获取视频信息失败: {e}")
        return {}


class VideoProcessor:
    """视频处理器 - 使用FFmpeg"""
    
    def __init__(self, target_resolution: Tuple[int, int] = (1080, 1920),
                 target_fps: int = 30):
        """
        初始化视频处理器
        
        Args:
            target_resolution: 目标分辨率 (width, height)
            target_fps: 目标帧率
        """
        self.target_resolution = target_resolution
        self.target_fps = target_fps
        logger.info(f"VideoProcessor initialized: {target_resolution[0]}x{target_resolution[1]} @ {target_fps}fps")
    
    def crop_video(self, input_path: str, output_path: str, 
                   start_time: float, end_time: float,
                   add_fade: bool = False,
                   fade_duration: float = 0.5) -> bool:
        """
        裁剪视频段
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            add_fade: 是否添加淡入淡出
            fade_duration: 淡入淡出时长
            
        Returns:
            是否成功
        """
        try:
            # 获取视频信息验证
            info = _get_video_info_ffmpeg(input_path)
            video_duration = info.get("duration", 0)
            
            if start_time >= video_duration:
                logger.error(f"开始时间 {start_time}s 超过视频时长 {video_duration}s")
                return False
            
            if end_time > video_duration:
                logger.warning(f"结束时间 {end_time}s 超过视频时长 {video_duration}s，已调整为 {video_duration}s")
                end_time = video_duration
            
            duration = end_time - start_time
            if duration <= 0:
                logger.error(f"Invalid duration: {duration}")
                return False
            
            width, height = self.target_resolution
            
            # 构建ffmpeg命令
            filters = []
            
            # 1. 裁剪
            filters.append(f"trim=start={start_time}:end={end_time},setpts=PTS-STARTPTS")
            
            # 2. 缩放和填充
            filters.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease")
            filters.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
            
            # 3. 设置帧率
            filters.append(f"fps={self.target_fps}")
            
            # 4. 淡入淡出
            if add_fade and duration > fade_duration * 2:
                # 淡入
                fade_in_expr = f"if(lt(t,{fade_duration}),0,if(lt(t,{fade_duration}+0.1),1,(t-{fade_duration})/0.1))"
                filters.append(f"fade=t=in:st=0:d={fade_duration}")
                # 淡出
                fade_out_start = duration - fade_duration
                filters.append(f"fade=t=out:st={fade_out_start}:d={fade_duration}")
            
            # 构建命令
            cmd = [
                'ffmpeg', '-y',
                '-i', str(input_path),
                '-vf', ','.join(filters),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                str(output_path)
            ]
            
            logger.info(f"Running ffmpeg crop: {' '.join(cmd)}")
            
            returncode, stdout, stderr = _run_ffmpeg(cmd)
            
            if returncode != 0:
                logger.error(f"裁剪视频失败: {stderr}")
                return False
            
            logger.info(f"Video cropped: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"裁剪视频失败: {e}", exc_info=True)
            return False
    
    def concat_videos(self, video_paths: List[str], output_path: str,
                      target_resolution: Optional[Tuple[int, int]] = None,
                      target_fps: Optional[int] = None,
                      add_fade: bool = True,
                      fade_duration: float = 0.5,
                      progress_callback: Optional[callable] = None) -> bool:
        """
        拼接多个视频
        
        Args:
            video_paths: 视频路径列表
            output_path: 输出路径
            target_resolution: 目标分辨率，None则使用默认
            target_fps: 目标帧率，None则使用默认
            add_fade: 是否添加过渡效果
            fade_duration: 淡入淡出时长
            progress_callback: 进度回调函数
            
        Returns:
            是否成功
        """
        if not video_paths:
            logger.error("No videos to concatenate")
            return False
        
        if len(video_paths) == 1:
            return self.transcode_video(video_paths[0], output_path, target_resolution, target_fps)
        
        try:
            resolution = target_resolution or self.target_resolution
            fps = target_fps or self.target_fps
            
            # 转换为绝对路径
            abs_video_paths = []
            for path in video_paths:
                if not os.path.isabs(path):
                    abs_video_paths.append(os.path.abspath(path))
                else:
                    abs_video_paths.append(path)
            
            # 报告进度
            if progress_callback:
                progress_callback(10)
            
            # 创建临时文件列表
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                list_file = f.name
                for path in abs_video_paths:
                    escaped_path = path.replace('\\', '/')
                    f.write(f"file '{escaped_path}'\n")
            
            if progress_callback:
                progress_callback(30)
            
            # 转换输出路径为绝对路径
            if not os.path.isabs(output_path):
                abs_output_path = os.path.abspath(output_path)
            else:
                abs_output_path = output_path
            
            # 构建ffmpeg命令
            width, height = resolution
            
            # 检查是否需要添加淡入淡出滤镜
            filters = []
            filters.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease")
            filters.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
            filters.append(f"fps={fps}")
            
            if add_fade:
                # 淡入淡出在concat后应用，需要使用复杂滤镜
                # 这里简化为不添加淡入淡出，因为concat模式不支持
                pass
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-vf', ','.join(filters),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                str(abs_output_path)
            ]
            
            logger.info(f"Running ffmpeg concat: {' '.join(cmd)}")
            
            if progress_callback:
                progress_callback(50)
            
            returncode, stdout, stderr = _run_ffmpeg(cmd, timeout=900)
            
            # 清理临时文件
            try:
                os.unlink(list_file)
            except (OSError, FileNotFoundError):
                # 临时文件可能已被删除，忽略错误
                pass
            
            if returncode != 0:
                logger.error(f"ffmpeg 拼接失败: {stderr}")
                return False
            
            if progress_callback:
                progress_callback(100)
            
            logger.info(f"ffmpeg 拼接成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"拼接视频失败: {e}", exc_info=True)
            return False
    
    def transcode_video(self, input_path: str, output_path: str,
                        target_resolution: Optional[Tuple[int, int]] = None,
                        target_fps: Optional[int] = None) -> bool:
        """
        转码视频
        
        Args:
            input_path: 输入路径
            output_path: 输出路径
            target_resolution: 目标分辨率
            target_fps: 目标帧率
            
        Returns:
            是否成功
        """
        try:
            resolution = target_resolution or self.target_resolution
            fps = target_fps or self.target_fps
            width, height = resolution
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(input_path),
                '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                str(output_path)
            ]
            
            returncode, stdout, stderr = _run_ffmpeg(cmd)
            
            if returncode != 0:
                logger.error(f"转码失败: {stderr}")
                return False
            
            logger.info(f"Video transcoded: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"转码失败: {e}")
            return False
    
    def add_fade(self, video_path: str, output_path: str, 
                 fade_duration: float = 0.5,
                 fade_in: bool = True,
                 fade_out: bool = True) -> bool:
        """
        添加淡入淡出
        
        Args:
            video_path: 输入视频
            output_path: 输出视频
            fade_duration: 淡入淡出时长
            fade_in: 是否淡入
            fade_out: 是否淡出
            
        Returns:
            是否成功
        """
        try:
            # 获取视频时长
            info = _get_video_info_ffmpeg(video_path)
            duration = info.get("duration", 0)
            
            filters = []
            
            if fade_in and duration > fade_duration:
                filters.append(f"fade=t=in:st=0:d={fade_duration}")
            
            if fade_out and duration > fade_duration:
                fade_out_start = duration - fade_duration
                filters.append(f"fade=t=out:st={fade_out_start}:d={fade_duration}")
            
            if not filters:
                # 没有需要添加的淡入淡出，直接复制
                cmd = ['ffmpeg', '-y', '-i', str(video_path), '-c', 'copy', str(output_path)]
            else:
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(video_path),
                    '-vf', ','.join(filters),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-c:a', 'aac',
                    str(output_path)
                ]
            
            returncode, stdout, stderr = _run_ffmpeg(cmd)
            
            if returncode != 0:
                logger.error(f"添加淡入淡出失败: {stderr}")
                return False
            
            logger.info(f"Added fade to: {video_path}")
            return True
            
        except Exception as e:
            logger.error(f"添加淡入淡出失败: {e}")
            return False
    
    def normalize_audio(self, video_path: str, output_path: str,
                        target_loudness: float = -20.0) -> bool:
        """
        音量归一化
        
        Args:
            video_path: 输入视频
            output_path: 输出视频
            target_loudness: 目标响度(dB)
            
        Returns:
            是否成功
        """
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-af', f'loudnorm=I={target_loudness}',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '128k',
                str(output_path)
            ]
            
            returncode, stdout, stderr = _run_ffmpeg(cmd)
            
            if returncode != 0:
                logger.error(f"音量归一化失败: {stderr}")
                return False
            
            logger.info(f"Audio normalized: {video_path}")
            return True
            
        except Exception as e:
            logger.error(f"音量归一化失败: {e}")
            return False
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """获取视频信息"""
        return _get_video_info_ffmpeg(video_path)
    
    def get_video_duration(self, video_path: str) -> float:
        """获取视频时长"""
        info = self.get_video_info(video_path)
        return info.get("duration", 0.0)
    
    def get_video_resolution(self, video_path: str) -> Tuple[int, int]:
        """获取视频分辨率"""
        info = self.get_video_info(video_path)
        return (info.get("width", 0), info.get("height", 0))
    
    def resize_video(self, input_path: str, output_path: str,
                     target_resolution: Tuple[int, int],
                     keep_aspect: bool = True) -> bool:
        """
        调整视频分辨率
        
        Args:
            input_path: 输入路径
            output_path: 输出路径
            target_resolution: 目标分辨率
            keep_aspect: 是否保持宽高比
            
        Returns:
            是否成功
        """
        try:
            width, height = target_resolution
            
            if keep_aspect:
                scale_filter = f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'
            else:
                scale_filter = f'scale={width}:{height}'
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(input_path),
                '-vf', scale_filter,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                str(output_path)
            ]
            
            returncode, stdout, stderr = _run_ffmpeg(cmd)
            
            if returncode != 0:
                logger.error(f"调整分辨率失败: {stderr}")
                return False
            
            logger.info(f"Video resized: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"调整分辨率失败: {e}")
            return False
    
    def extract_audio(self, video_path: str, output_path: str,
                      sample_rate: int = 16000, channels: int = 1) -> bool:
        """
        提取音频
        
        Args:
            video_path: 输入视频
            output_path: 输出音频
            sample_rate: 采样率
            channels: 声道数
            
        Returns:
            是否成功
        """
        try:
            # 确定输出格式
            if output_path.endswith('.wav'):
                acodec = 'pcm_s16le'
            else:
                acodec = 'aac'
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vn',
                '-acodec', acodec,
                '-ar', str(sample_rate),
                '-ac', str(channels),
                str(output_path)
            ]
            
            returncode, stdout, stderr = _run_ffmpeg(cmd)
            
            if returncode != 0:
                logger.error(f"提取音频失败: {stderr}")
                return False
            
            logger.info(f"Audio extracted: {video_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"提取音频失败: {e}")
            return False
    
    def add_audio(self, video_path: str, audio_path: str, output_path: str,
                  audio_volume: float = 1.0) -> bool:
        """
        添加音频到视频
        
        Args:
            video_path: 输入视频
            audio_path: 音频文件
            output_path: 输出视频
            audio_volume: 音量增益
            
        Returns:
            是否成功
        """
        try:
            volume_filter = f'volume={audio_volume}' if audio_volume != 1.0 else None
            
            if volume_filter:
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(video_path),
                    '-i', str(audio_path),
                    '-vf', volume_filter,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-shortest',
                    str(output_path)
                ]
            else:
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(video_path),
                    '-i', str(audio_path),
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-shortest',
                    str(output_path)
                ]
            
            returncode, stdout, stderr = _run_ffmpeg(cmd)
            
            if returncode != 0:
                logger.error(f"添加音频失败: {stderr}")
                return False
            
            logger.info(f"Audio added: {video_path} + {audio_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"添加音频失败: {e}")
            return False
    
    def create_preview_thumbnail(self, video_path: str, output_path: str,
                                  timestamp: float = 0.0) -> bool:
        """
        生成预览缩略图
        
        Args:
            video_path: 视频路径
            output_path: 输出图片路径
            timestamp: 截取时间点
            
        Returns:
            是否成功
        """
        try:
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(timestamp),
                '-i', str(video_path),
                '-vframes', '1',
                '-q:v', '2',
                str(output_path)
            ]
            
            returncode, stdout, stderr = _run_ffmpeg(cmd)
            
            if returncode != 0:
                logger.error(f"生成缩略图失败: {stderr}")
                return False
            
            logger.info(f"Thumbnail created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成缩略图失败: {e}")
            return False
    
    def batch_process(self, input_files: List[str], output_dir: str,
                      process_func: str = "transcode") -> List[str]:
        """
        批量处理视频
        
        Args:
            input_files: 输入文件列表
            output_dir: 输出目录
            process_func: 处理函数名
            
        Returns:
            输出文件列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = []
        
        for i, input_file in enumerate(input_files):
            filename = Path(input_file).stem
            output_file = os.path.join(output_dir, f"{filename}_out.mp4")
            
            if process_func == "transcode":
                success = self.transcode_video(input_file, output_file)
            else:
                success = False
            
            if success:
                output_files.append(output_file)
        
        return output_files
