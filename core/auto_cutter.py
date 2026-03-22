"""
自动化剪辑控制器 - VideoAutoCutter
串联全流程：素材扫描 → 文案解析 → 序列生成 → FFmpeg 合成

使用新的模块化架构：
- db/models.py: SQLAlchemy 数据模型
- core/processor.py: 素材净化与分类 (VideoPurifier)
- core/planner.py: 带状态的序列决策引擎 (SequencePlanner)
"""
import os
import re
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger("smart_cut")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
TEMP_DIR = PROJECT_ROOT / "temp"


class VideoAutoCutter:
    """自动化剪辑控制器"""
    
    # 支持的视频格式
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.MOV', '.MP4'}
    
    # 目标分辨率
    TARGET_WIDTH = 1080
    TARGET_HEIGHT = 1920
    TARGET_FPS = 30
    
    def __init__(self, 
                 raw_folder: str = "storage/materials-all",
                 db_path: str = "storage/materials.db",
                 output_dir: str = "temp"):
        """
        初始化控制器
        
        Args:
            raw_folder: 原始素材文件夹
            db_path: SQLite 数据库路径
            output_dir: 输出目录
        """
        self.raw_folder = Path(raw_folder)
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        
        # 确保目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 共享数据库实例
        self.db = None
        
        # 初始化组件
        self.processor = None
        self.planner = None
        self.video_processor = None
        
        # 统计信息
        self.stats = {
            "total_files": 0,
            "new_files": 0,
            "skipped_files": 0,
            "a_roll_count": 0,
            "b_roll_count": 0
        }
    
    def _init_components(self):
        """初始化组件"""
        # 初始化数据库
        if self.db is None:
            from db.models import Database
            self.db = Database(self.db_path)
        
        # 初始化 VideoPurifier
        if self.processor is None:
            logger.info("初始化 VideoPurifier...")
            from core.processor import VideoPurifier
            self.processor = VideoPurifier(self.db)
        
        # 初始化 SequencePlanner
        if self.planner is None:
            logger.info("初始化 SequencePlanner...")
            from core.planner import SequencePlanner, EmbeddingModel
            embedding_model = EmbeddingModel()
            self.planner = SequencePlanner(self.db, embedding_model)
        
        # 初始化 VideoProcessor
        if self.video_processor is None:
            logger.info("初始化 VideoProcessor...")
            from core.video_processor import VideoProcessor
            self.video_processor = VideoProcessor(
                target_resolution=(self.TARGET_WIDTH, self.TARGET_HEIGHT),
                target_fps=self.TARGET_FPS
            )
    
    def scan_materials(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        扫描素材文件夹，执行清洗和特征提取
        
        Args:
            force_reprocess: 是否强制重新处理
            
        Returns:
            扫描结果统计
        """
        self._init_components()
        
        logger.info("=" * 60)
        logger.info("Step 1: 素材扫描与清洗")
        logger.info("=" * 60)
        
        # 获取所有视频文件
        video_files = []
        for ext in self.VIDEO_EXTENSIONS:
            video_files.extend(self.raw_folder.glob(f"*{ext}"))
        
        self.stats["total_files"] = len(video_files)
        logger.info(f"发现 {len(video_files)} 个视频文件")
        
        # 处理每个文件
        for i, video_path in enumerate(sorted(video_files)):
            logger.info(f"[{i+1}/{len(video_files)}] 处理: {video_path.name}")
            
            # 检查是否需要重新处理（断点续传）
            current_mtime = os.path.getmtime(video_path)
            need_process = force_reprocess or not self.db.check_asset_fresh(str(video_path), current_mtime)
            
            if need_process:
                logger.info(f"  -> 新素材，执行 ASR 和清洗...")
                asset = self.processor.purify(str(video_path), force_reprocess=force_reprocess)
                
                self.stats["new_files"] += 1
                
                if asset.track_type == "A_ROLL":
                    self.stats["a_roll_count"] += 1
                    logger.info(f"  -> A_ROLL, offset={asset.valid_start_offset:.3f}s")
                else:
                    self.stats["b_roll_count"] += 1
                    logger.info(f"  -> B_ROLL")
            else:
                logger.info(f"  -> 使用缓存，跳过")
                self.stats["skipped_files"] += 1
                
                # 统计已有素材类型
                db_asset = self.db.get_asset_by_path(str(video_path))
                if db_asset:
                    if db_asset.track_type == "A_ROLL":
                        self.stats["a_roll_count"] += 1
                    else:
                        self.stats["b_roll_count"] += 1
        
        # 打印统计
        logger.info("")
        logger.info("=== 素材扫描结果 ===")
        logger.info(f"总文件数: {self.stats['total_files']}")
        logger.info(f"新处理: {self.stats['new_files']}")
        logger.info(f"使用缓存: {self.stats['skipped_files']}")
        logger.info(f"A_ROLL: {self.stats['a_roll_count']}")
        logger.info(f"B_ROLL: {self.stats['b_roll_count']}")
        
        return self.stats
    
    def parse_script(self, script_text: str) -> List[str]:
        """
        解析文案脚本
        
        Args:
            script_text: 原始文案
            
        Returns:
            句子列表
        """
        # 按标点切分
        sentences = re.split(r'[。！？；\n]', script_text)
        
        # 清理空句子
        result = []
        for s in sentences:
            s = s.strip()
            if s and len(s) > 1:  # 过滤单字符
                result.append(s)
        
        logger.info(f"文案解析: {len(result)} 个句子")
        return result
    
    def plan(self, script_text: str) -> List[Dict]:
        """
        生成剪辑序列规划
        
        Args:
            script_text: 文案脚本
            
        Returns:
            EDL 剪辑列表
        """
        self._init_components()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Step 2: 文案语义编排")
        logger.info("=" * 60)
        
        # 解析文案
        sentences = self.parse_script(script_text)
        
        # 生成 EDL
        logger.info("执行序列规划...")
        edl = self.planner.plan(script_text)
        
        # 打印可视化表格
        self._print_edl_table(edl)
        
        return edl
    
    def _print_edl_table(self, edl: List[Dict]):
        """打印 EDL 可视化表格"""
        logger.info("")
        logger.info("=== 匹配结果详情 ===")
        logger.info(f"{'序号':<4} {'文案句子':<25} {'素材名':<20} {'类型':<8} {'得分':<6} {'惩罚':<10}")
        logger.info("-" * 90)
        
        for i, clip in enumerate(edl):
            text = clip.get("text", "")[:22] + "..." if len(clip.get("text", "")) > 22 else clip.get("text", "")
            
            if clip.get("missing"):
                logger.info(f"{i+1:<4} {text:<25} {'(无匹配)':<20} {'-':<8} {'-':<6} {'-':<10}")
            else:
                material_name = clip.get("video_name", "N/A")[:18]
                track_type = clip.get("track_type", "-")[:6]
                similarity = f"{clip.get('similarity', 0):.2f}"
                
                reason = clip.get("reason", "")
                penalty = ""
                if reason == "repeat_penalty":
                    penalty = "重复惩罚"
                elif reason == "time_reverse_penalty":
                    penalty = "时光倒流!"
                elif reason == "sequence_reward":
                    penalty = "顺序奖励"
                
                fallback = " [后备]" if clip.get("fallback") else ""
                is_b_roll = " [B-Roll]" if clip.get("is_b_roll") else ""
                track_type = f"{track_type}{is_b_roll}"[:8]
                
                logger.info(f"{i+1:<4} {text:<25} {material_name:<20} {track_type:<8} {similarity:<6} {penalty:<10}")
        
        # 统计
        a_count = sum(1 for e in edl if e.get("track_type") == "A_ROLL")
        b_count = sum(1 for e in edl if e.get("track_type") == "B_ROLL")
        missing_count = sum(1 for e in edl if e.get("missing", False))
        
        logger.info("-" * 90)
        logger.info(f"总计: {len(edl)} | A_ROLL: {a_count} | B_ROLL: {b_count} | 缺失: {missing_count}")
    
    def render(self, edl: List[Dict], output_name: str = "final_output.mp4",
               use_crossfade: bool = True, crossfade_duration: float = 0.2) -> bool:
        """
        FFmpeg 物理合成 - 优化版
        
        特性：
        1. 分段导出为 .ts 格式
        2. 统一分辨率、帧率、采样率
        3. 支持音频淡入淡出（消除剪辑点爆音）
        4. 使用 concat 协议合并
        
        Args:
            edl: 剪辑列表
            output_name: 输出文件名
            use_crossfade: 是否使用音频淡入淡出
            crossfade_duration: 淡入淡出时长（秒）
            
        Returns:
            是否成功
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("Step 3: FFmpeg 物理合成")
        logger.info("=" * 60)
        
        # 过滤有效片段
        valid_clips = [c for c in edl if not c.get("missing") and c.get("video_path")]
        
        if not valid_clips:
            logger.error("没有有效的剪辑片段")
            return False
        
        logger.info(f"合成 {len(valid_clips)} 个片段...")
        logger.info(f"参数: 分辨率={self.TARGET_WIDTH}x{self.TARGET_HEIGHT}, "
                   f"帧率={self.TARGET_FPS}fps, 淡入淡出={use_crossfade}")
        
        # 步骤1: 分段导出为 ts 格式
        temp_clips = []

        for i, clip in enumerate(valid_clips):
            video_path = clip["video_path"]
            # 关键：start 已经包含了 valid_offset，使用 round 避免浮点误差
            start = round(clip["start"], 3)
            end = round(clip["end"], 3)
            duration = round(end - start, 3)

            if duration <= 0:
                logger.warning(f"片段 {i} 时长为 0，跳过")
                continue

            # 输出临时 ts 文件
            temp_file = self.output_dir / f"segment_{i:03d}.ts"
            temp_clips.append(temp_file)

            logger.info(f"  [{i+1}/{len(valid_clips)}] 转码: {Path(video_path).name} "
                      f"({start:.3f}s - {end:.3f}s, 时长: {duration:.3f}s)")

            # 统一参数转码为 ts 格式
            success = self._transcode_to_ts(
                video_path,
                str(temp_file),
                start,
                duration,
                add_fade=(use_crossfade and duration > crossfade_duration * 2),
                fade_duration=crossfade_duration
            )
            
            if not success:
                logger.error(f"转码失败: {video_path}")
                return False
        
        # 步骤2: 使用 concat 协议合并
        output_path = self.output_dir / output_name
        logger.info(f"合并 {len(temp_clips)} 个片段...")
        
        if use_crossfade:
            success = self._concat_with_crossfade(temp_clips, str(output_path), crossfade_duration)
        else:
            success = self._concat_ts_protocol(temp_clips, str(output_path))
        
        # 清理临时文件
        for temp_file in temp_clips:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
        
        if success:
            logger.info(f"合成完成: {output_path}")
            logger.info(f"输出文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return success
    
    def _transcode_to_ts(self, input_path: str, output_path: str,
                         start: float, duration: float,
                         add_fade: bool = False, fade_duration: float = 0.2) -> bool:
        """统一规格转码为 ts 格式"""
        try:
            # 构建视频滤镜：缩放 + 填充 + 帧率
            scale_filter = f"scale={self.TARGET_WIDTH}:{self.TARGET_HEIGHT}:force_original_aspect_ratio=decrease"
            pad_filter = f"pad={self.TARGET_WIDTH}:{self.TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
            fps_filter = f"fps={self.TARGET_FPS}"
            
            filters = [scale_filter, pad_filter, fps_filter]
            
            # 添加淡入淡出
            if add_fade:
                filters.append(f"fade=t=in:st=0:d={fade_duration}")
                fade_out_start = duration - fade_duration
                filters.append(f"fade=t=out:st={fade_out_start}:d={fade_duration}")
            
            # 构建命令 - 优先使用 h264_amf，回退到 libx264
            use_amf = getattr(self, '_use_amf', True)  # 默认尝试 AMF

            if use_amf:
                logger.info(f"  [AMF] 使用 AMD 硬件加速转码")
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start),
                    '-t', str(duration),
                    '-hwaccel', 'auto',  # 启用硬件加速
                    '-i', input_path,
                    '-vf', ','.join(filters),
                    '-r', str(self.TARGET_FPS),
                    '-c:v', 'h264_amf',
                    '-rc', 'cbr',
                    '-quality', 'quality',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-ar', '44100',
                    '-b:a', '128k',
                    '-muxrate', '10M',
                    '-f', 'mpegts',
                    output_path
                ]
            else:
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start),
                    '-t', str(duration),
                    '-i', input_path,
                    '-vf', ','.join(filters),
                    '-r', str(self.TARGET_FPS),
                    '-c:v', 'libx264',
                    '-preset', 'superfast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-ar', '44100',
                    '-b:a', '128k',
                    '-muxrate', '10M',
                    '-f', 'mpegts',
                    output_path
                ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # 如果 AMF 失败，回退到 libx264
            if use_amf and result.returncode != 0:
                logger.warning(f"h264_amf 编码失败，回退到 libx264")
                self._use_amf = False
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start),
                    '-t', str(duration),
                    '-i', input_path,
                    '-vf', ','.join(filters),
                    '-r', str(self.TARGET_FPS),
                    '-c:v', 'libx264',
                    '-preset', 'superfast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-ar', '44100',
                    '-b:a', '128k',
                    '-muxrate', '10M',
                    '-f', 'mpegts',
                    output_path
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
            
            if result.returncode != 0:
                logger.error(f"ffmpeg 转码失败: {result.stderr[:200]}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg 转码超时")
            return False
        except Exception as e:
            logger.error(f"转码异常: {e}")
            return False
    
    def _concat_ts_protocol(self, ts_files: List[Path], output_path: str) -> bool:
        """使用 concat 协议合并 ts 文件"""
        try:
            concat_str = "|".join([f.as_posix() for f in ts_files])
            
            cmd = [
                'ffmpeg', '-y',
                '-i', f"concat:{concat_str}",
                '-c', 'copy',
                '-bsf', 'aac_adtstoasc',
                '-movflags', '+faststart',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                logger.warning(f"concat 协议失败，尝试 concat demuxer")
                return self._concat_demuxer(ts_files, output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"合并异常: {e}")
            return False
    
    def _concat_demuxer(self, ts_files: List[Path], output_path: str) -> bool:
        """使用 concat demuxer 合并 ts 文件"""
        try:
            concat_list = self.output_dir / "concat_list.txt"
            
            with open(concat_list, 'w', encoding='utf-8') as f:
                for ts_file in ts_files:
                    abs_path = ts_file.absolute().as_posix()
                    f.write(f"file '{abs_path}'\n")
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_list),
                '-c', 'copy',
                '-movflags', '+faststart',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            try:
                concat_list.unlink()
            except:
                pass
            
            if result.returncode != 0:
                logger.error(f"concat demuxer 失败: {result.stderr[:200]}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"合并异常: {e}")
            return False
    
    def _concat_with_crossfade(self, ts_files: List[Path], output_path: str,
                               crossfade_duration: float = 0.2) -> bool:
        """使用 filter_complex 实现带音频淡入淡出的合并"""
        if len(ts_files) < 2:
            import shutil
            shutil.copy(ts_files[0], output_path)
            return True
        
        try:
            inputs = []
            for f in ts_files:
                inputs.extend(['-i', str(f)])
            
            filter_parts = []
            for i in range(len(ts_files)):
                filter_parts.append(f"[{i}:v]")
                filter_parts.append(f"[{i}:a]")
            
            filter_parts.append(f"concat=n={len(ts_files)}:v=1:a=1[outv][outa]")

            # 使用 AMF 或 libx264
            use_amf = getattr(self, '_use_amf', True)

            if use_amf:
                logger.info(f"  [AMF] 最终合并使用 AMD 硬件加速")
                cmd = [
                    'ffmpeg', '-y',
                    '-hwaccel', 'auto'  # 启用硬件加速
                ] + inputs + [
                    '-filter_complex', ''.join(filter_parts),
                    '-map', '[outv]',
                    '-map', '[outa]',
                    '-c:v', 'h264_amf',
                    '-rc', 'cbr',
                    '-quality', 'quality',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-movflags', '+faststart',
                    output_path
                ]
            else:
                cmd = [
                    'ffmpeg', '-y'
                ] + inputs + [
                    '-filter_complex', ''.join(filter_parts),
                    '-map', '[outv]',
                    '-map', '[outa]',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-movflags', '+faststart',
                    output_path
                ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            # 如果 AMF 失败，回退到 libx264
            if use_amf and result.returncode != 0:
                logger.warning(f"h264_amf 合并失败，回退到 libx264")
                self._use_amf = False
                cmd = [
                    'ffmpeg', '-y'
                ] + inputs + [
                    '-filter_complex', ''.join(filter_parts),
                    '-map', '[outv]',
                    '-map', '[outa]',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-movflags', '+faststart',
                    output_path
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
            
            if result.returncode != 0:
                logger.error(f"合并失败: {result.stderr[:200]}")
                return self._concat_demuxer(ts_files, output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"合并异常: {e}")
            return False
    
    def run(self, script_text: str, output_name: str = "final_output.mp4",
            force_reprocess: bool = False) -> bool:
        """
        执行完整流程
        
        Args:
            script_text: 文案脚本
            output_name: 输出文件名
            force_reprocess: 是否强制重新处理素材
            
        Returns:
            是否成功
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("短视频自动剪辑系统启动")
        logger.info("=" * 60)
        
        # Step 1: 素材扫描
        self.scan_materials(force_reprocess=force_reprocess)
        
        # Step 2: 文案编排
        edl = self.plan(script_text)
        
        # Step 3: FFmpeg 合成
        success = self.render(edl, output_name)
        
        # 统计耗时
        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"流程完成! 耗时: {elapsed:.1f} 秒")
        logger.info("=" * 60)
        
        return success
    
    def close(self):
        """关闭资源"""
        if self.db:
            self.db.close()


# ============ 测试代码 ============
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    SCRIPT = """不用确定我们京东315活动3C、家电政府补贴至高15%
    我再问一下
    不用问,美的四大权益都可以享受
    权益一
    全屋智能家电套购
    送至高1499元豪礼"""
    
    cutter = VideoAutoCutter(
        raw_folder="storage/materials-all",
        db_path="storage/materials.db",
        output_dir="temp"
    )
    
    success = cutter.run(SCRIPT, "test_output.mp4")
    
    cutter.close()
    
    if success:
        print("\n视频生成成功!")
    else:
        print("\n视频生成失败!")
