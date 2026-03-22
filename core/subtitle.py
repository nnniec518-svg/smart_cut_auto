"""
字幕生成模块
提供ASS字幕文件生成和字幕嵌入视频功能
"""
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from core.utils import run_ffmpeg, check_ffmpeg

logger = logging.getLogger("smart_cut")


class SubtitleGenerator:
    """字幕生成器"""
    
    def __init__(self):
        """初始化字幕生成器"""
        self.default_style = {
            "font": "Microsoft YaHei",
            "font_size": 42,
            "primary_color": "&H00FFFFFF",  # 白色
            "secondary_color": "&H00000000",
            "outline_color": "&H00000000",  # 黑色
            "back_color": "&H80000000",     # 半透明黑色
            "bold": 0,
            "italic": 0,
            "underline": 0,
            "strikeout": 0,
            "scale_x": 100,
            "scale_y": 100,
            "spacing": 0,
            "angle": 0,
            "border_style": 1,
            "outline": 2,
            "shadow": 0,
            "alignment": 2,                 # 底部居中
            "margin_l": 10,
            "margin_r": 10,
            "margin_v": 10,
            "encoding": 1
        }
    
    def style_from_settings(self, settings: Dict[str, Any], 
                           video_height: int = 1920) -> Dict[str, Any]:
        """
        从配置生成字幕样式
        
        Args:
            settings: 配置字典
            video_height: 视频高度，用于计算字体大小
           
        Returns:
            样式字典
        """
        style = self.default_style.copy()
        
        # 字体
        style["font"] = settings.get("subtitle_font", "Microsoft YaHei")
        
        # 字体大小（按视频高度比例）
        size_ratio = settings.get("subtitle_size_ratio", 0.07)
        style["font_size"] = int(video_height * size_ratio)
        
        # 颜色
        primary_color = settings.get("subtitle_color", "#FFFFFF")
        style["primary_color"] = self._hex_to_ass_color(primary_color)
        
        outline_color = settings.get("subtitle_stroke_color", "#000000")
        style["outline_color"] = self._hex_to_ass_color(outline_color)
        
        # 边框宽度
        style["outline"] = settings.get("subtitle_stroke_width", 1.5)
        
        # 背景框
        if settings.get("subtitle_background", True):
            style["back_color"] = "&H80000000"  # 半透明黑色
            style["border_style"] = 1
        else:
            style["back_color"] = "&H00000000"  # 透明
            style["border_style"] = 3  # 3是Outline + drop shadow
        
        # 边距
        style["margin_v"] = settings.get("subtitle_margin", 10)
        
        return style
    
    def _hex_to_ass_color(self, hex_color: str) -> str:
        """将十六进制颜色转换为ASS颜色"""
        # 格式: #RRGGBB -> &HRRGGBB (ABGR)
        hex_color = hex_color.lstrip('#')
        
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            # ASS颜色是ABGR
            return f"&H00{b:02X}{g:02X}{r:02X}"
        
        return "&H00FFFFFF"
    
    def _format_time_ass(self, seconds: float) -> str:
        """格式化时间为ASS时间格式 (H:MM:SS.cc)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
    
    def generate_ass(self, asr_result: Dict[str, Any], 
                     output_path: str,
                     style_params: Optional[Dict[str, Any]] = None,
                     video_height: int = 1920) -> bool:
        """
        生成ASS字幕文件
        
        Args:
            asr_result: ASR结果字典
            output_path: 输出ASS文件路径
            style_params: 样式参数
            video_height: 视频高度
            
        Returns:
            是否成功
        """
        try:
            # 获取样式
            if style_params is None:
                style_params = {}
            
            style = self.style_from_settings(style_params, video_height)
            
            # 生成ASS文件头
            ass_content = self._generate_ass_header(style)
            
            # 添加字幕事件
            segments = asr_result.get("segments", [])
            
            for i, seg in enumerate(segments):
                text = seg.get("text", "").replace("\n", " ").strip()
                if not text:
                    continue
                
                start = self._format_time_ass(seg.get("start", 0))
                end = self._format_time_ass(seg.get("end", 0))
                
                # 处理文本中的特殊字符
                text = self._escape_ass_text(text)
                
                # ASS事件行
                event_line = f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
                ass_content += event_line + "\n"
            
            # 写入文件
            with open(output_path, "w", encoding="utf-8-sig") as f:
                f.write(ass_content)
            
            logger.info(f"ASS subtitle generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成ASS字幕失败: {e}")
            return False
    
    def _generate_ass_header(self, style: Dict[str, Any]) -> str:
        """生成ASS文件头"""
        header = """[Script Info]
Title: Smart Cut Auto Subtitle
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1920
PlayResY: 1080
Timer: 100.0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},{primary_color},{secondary_color},{outline_color},{back_color},{bold},{italic},{underline},{strikeout},{scale_x},{scale_y},{spacing},{angle},{border_style},{outline},{shadow},{alignment},{margin_l},{margin_r},{margin_v},{encoding}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(**style)
        
        return header
    
    def _escape_ass_text(self, text: str) -> str:
        """转义ASS文本中的特殊字符"""
        # 替换
        text = text.replace("\\", "\\\\")
        text = text.replace("{", "\\{")
        text = text.replace("}", "\\}")
        text = text.replace(r"\N", r"\\N")
        
        return text
    
    def generate_srt(self, asr_result: Dict[str, Any], 
                     output_path: str) -> bool:
        """
        生成SRT字幕文件
        
        Args:
            asr_result: ASR结果字典
            output_path: 输出SRT文件路径
            
        Returns:
            是否成功
        """
        try:
            segments = asr_result.get("segments", [])
            
            srt_content = ""
            
            for i, seg in enumerate(segments):
                text = seg.get("text", "").replace("\n", " ").strip()
                if not text:
                    continue
                
                # 序号
                srt_content += f"{i + 1}\n"
                
                # 时间码
                start = self._format_time_srt(seg.get("start", 0))
                end = self._format_time_srt(seg.get("end", 0))
                srt_content += f"{start} --> {end}\n"
                
                # 文本
                srt_content += f"{text}\n\n"
            
            # 写入文件
            with open(output_path, "w", encoding="utf-8-sig") as f:
                f.write(srt_content)
            
            logger.info(f"SRT subtitle generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成SRT字幕失败: {e}")
            return False
    
    def _format_time_srt(self, seconds: float) -> str:
        """格式化时间为SRT格式 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def burn_subtitle(self, video_path: str, ass_path: str, 
                     output_path: str) -> bool:
        """
        使用FFmpeg将字幕嵌入视频
        
        Args:
            video_path: 输入视频路径
            ass_path: ASS字幕文件路径
            output_path: 输出视频路径
            
        Returns:
            是否成功
        """
        try:
            # 检查 FFmpeg 是否可用
            if not check_ffmpeg():
                logger.warning("FFmpeg not available, skipping subtitle burning. Please install FFmpeg.")
                # 复制视频文件作为输出
                import shutil
                shutil.copy(str(video_path), str(output_path))
                return True
            
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"ass={ass_path}",
                "-c:a", "copy",
                str(output_path)
            ]
            
            run_ffmpeg(cmd, timeout=600)
            logger.info(f"Subtitle burned: {video_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"嵌入字幕失败: {e}")
            return False
    
    def burn_srt_subtitle(self, video_path: str, srt_path: str,
                         output_path: str,
                         font_name: str = "Microsoft YaHei",
                         font_size: int = 42) -> bool:
        """
        嵌入SRT字幕（使用drawtext滤镜）
        
        Args:
            video_path: 输入视频
            srt_path: SRT字幕文件
            output_path: 输出视频
            font_name: 字体名称
            font_size: 字体大小
            
        Returns:
            是否成功
        """
        try:
            # 检查 FFmpeg 是否可用
            if not check_ffmpeg():
                logger.warning("FFmpeg not available, skipping subtitle burning.")
                import shutil
                shutil.copy(str(video_path), str(output_path))
                return True
            
            # SRT需要转换为ASS或使用subtitles滤镜
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"subtitles={srt_path}:force_style='FontName={font_name},FontSize={font_size}'",
                "-c:a", "copy",
                str(output_path)
            ]
            
            run_ffmpeg(cmd, timeout=600)
            logger.info(f"SRT subtitle burned: {video_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"嵌入SRT字幕失败: {e}")
            return False
    
    def extract_subtitle(self, video_path: str, output_path: str) -> bool:
        """
        从视频提取字幕
        
        Args:
            video_path: 视频路径
            output_path: 输出SRT路径
            
        Returns:
            是否成功
        """
        try:
            if not check_ffmpeg():
                logger.warning("FFmpeg not available, cannot extract subtitle.")
                return False
            
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-map", "s:s",
                str(output_path)
            ]
            
            # 不检查返回码，因为不是所有视频都有内嵌字幕
            run_ffmpeg(cmd, check=False)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Subtitle extracted: {output_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"提取字幕失败: {e}")
            return False
    
    def preview_subtitle(self, video_path: str, asr_result: Dict[str, Any],
                        output_path: str,
                        style_params: Optional[Dict[str, Any]] = None,
                        max_duration: float = 60.0) -> bool:
        """
        生成字幕预览视频
        
        Args:
            video_path: 视频路径
            asr_result: ASR结果
            output_path: 输出路径
            style_params: 样式参数
            max_duration: 最大预览时长
            
        Returns:
            是否成功
        """
        try:
            # 限制时长
            if asr_result.get("segments"):
                last_end = max(seg.get("end", 0) for seg in asr_result["segments"])
                if last_end > max_duration:
                    # 截断ASR结果
                    asr_result = {
                        "segments": [s for s in asr_result["segments"] if s.get("end", 0) <= max_duration]
                    }
            
            # 获取视频信息
            from core.video_processor import VideoProcessor
            vp = VideoProcessor()
            info = vp.get_video_info(video_path)
            video_height = info.get("height", 1920)
            
            # 生成临时ASS
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".ass", delete=False, mode="w") as f:
                temp_ass = f.name
            
            try:
                success = self.generate_ass(asr_result, temp_ass, style_params, video_height)
                if not success:
                    return False
                
                # 嵌入字幕
                return self.burn_subtitle(video_path, temp_ass, output_path)
            finally:
                try:
                    Path(temp_ass).unlink()
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"生成预览失败: {e}")
            return False
    
    def get_subtitle_text(self, asr_result: Dict[str, Any]) -> str:
        """获取字幕文本"""
        segments = asr_result.get("segments", [])
        texts = [seg.get("text", "").strip() for seg in segments if seg.get("text")]
        return " ".join(texts)
    
    def split_long_subtitle(self, asr_result: Dict[str, Any],
                            max_duration: float = 7.0) -> List[Dict[str, Any]]:
        """
        将长字幕段拆分为短段
        
        Args:
            asr_result: ASR结果
            max_duration: 最大单段时长
            
        Returns:
            拆分后的结果
        """
        segments = asr_result.get("segments", [])
        new_segments = []
        
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            duration = end - start
            
            if duration <= max_duration:
                new_segments.append(seg)
            else:
                # 拆分
                num_parts = int(duration / max_duration) + 1
                part_duration = duration / num_parts
                
                text = seg.get("text", "")
                
                for i in range(num_parts):
                    new_seg = {
                        "text": text,
                        "start": start + i * part_duration,
                        "end": start + (i + 1) * part_duration
                    }
                    
                    if "words" in seg:
                        # 简单处理：复制所有词
                        new_seg["words"] = seg["words"]
                    
                    new_segments.append(new_seg)
        
        return new_segments
