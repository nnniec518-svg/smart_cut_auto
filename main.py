"""
智能短视频剪辑系统 - 主控制器
串联全流程：素材扫描 → 文案解析 → 序列生成 → FFmpeg 合成

使用新的模块化架构：
- db/models.py: SQLAlchemy 数据模型
- core/processor.py: 素材净化与分类 (VideoPurifier)
- core/planner.py: 带状态的序列决策引擎 (SequencePlanner)
- core/auto_cutter.py: 高性能渲染器 (FastRenderer)

缓存机制优化：
- logic_version: 控制ASR重新识别
- script_hash: 控制文案变化时强制重新规划（内容或顺序变化）
"""
import os
import sys
import time
import logging
import yaml
import hashlib
from pathlib import Path
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入核心模块
from db.models import Database
from core.processor import VideoPurifier
from core.planner import SequencePlanner, EmbeddingModel
from core.auto_cutter import VideoAutoCutter
from core.hardware import init_device, get_device_info


def load_config(config_path: str = "config.yaml") -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        # 返回默认配置
        return {
            "database": {"path": "storage/materials.db"},
            "paths": {
                "raw_folder": "storage/materials-all",
                "output_dir": "temp"
            },
            "render": {
                "width": 1080,
                "height": 1920,
                "fps": 30,
                "crossfade_duration": 0.2
            }
        }

    with open(config_file, 'r', encoding='utf-8') as f:
        import yaml
        return yaml.safe_load(f)


def setup_logging(config):
    """设置日志"""
    log_dir = PROJECT_ROOT / config.get("paths", {}).get("logs_dir", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "smart_cut.log"
    log_level = config.get("logging", {}).get("level", "INFO")
    log_format = config.get("logging", {}).get("format", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # 配置根日志
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def calculate_script_hash(script: str) -> str:
    """
    计算文案的MD5哈希值

    用于检测文案内容或顺序变化，触发序列重新规划

    Args:
        script: 文案脚本

    Returns:
        MD5哈希值
    """
    # 规范化文本：去除多余空格和换行
    normalized = '\n'.join(line.strip() for line in script.strip().split('\n') if line.strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def check_script_cache_valid(script: str, cache_path: Path) -> bool:
    """
    检查文案缓存是否有效

    对比当前文案与缓存的哈希值，判断是否需要重新规划

    Args:
        script: 当前文案
        cache_path: 缓存文件路径（temp/script_hash.txt）

    Returns:
        True表示缓存有效，False表示需要重新规划
    """
    import logging
    _logger = logging.getLogger("smart_cut")

    current_hash = calculate_script_hash(script)

    # 读取缓存的哈希
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_hash = f.read().strip()

            if cached_hash != current_hash:
                _logger.info(f"文案变化检测: 哈希值不同 ({cached_hash[:8]}... -> {current_hash[:8]}...)")
                _logger.info("将重新执行Matcher和Assembler流程")
                return False

            _logger.info(f"文案未变化，哈希值: {current_hash[:8]}... (可使用缓存)")
            return True
        except Exception as e:
            _logger.warning(f"读取缓存哈希失败: {e}")
            return False

    # 首次运行，无缓存
    return False


def save_script_hash(script: str, cache_path: Path):
    """
    保存文案哈希到缓存

    Args:
        script: 文案脚本
        cache_path: 缓存文件路径
    """
    import logging
    _logger = logging.getLogger("smart_cut")

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        script_hash = calculate_script_hash(script)

        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(script_hash)

        _logger.info(f"文案哈希已保存: {script_hash[:8]}...")
    except Exception as e:
        _logger.warning(f"保存文案哈希失败: {e}")


def force_clean_all_caches():
    """
    强制清理所有缓存（文案哈希 + logic_version + 数据库）

    在文案变化或logic_version变化时调用
    """
    import logging
    import sqlite3
    import gc
    _logger = logging.getLogger("smart_cut")

    _logger.info("=== 强制清理所有缓存 ===")

    project_root = Path(__file__).parent
    temp_dir = project_root / "temp"
    db_path = project_root / "storage" / "materials.db"

    # 1. 删除文案哈希缓存
    script_hash_path = temp_dir / "script_hash.txt"
    if script_hash_path.exists():
        try:
            script_hash_path.unlink()
            _logger.info(f"  删除文案哈希缓存")
        except Exception as e:
            _logger.warning(f"  删除文案哈希缓存失败: {e}")

    # 2. 删除 sequence.json
    sequence_path = temp_dir / "sequence.json"
    if sequence_path.exists():
        try:
            sequence_path.unlink()
            _logger.info(f"  删除序列缓存: sequence.json")
        except Exception as e:
            _logger.warning(f"  删除序列缓存失败: {e}")

    # 3. 删除 logic_version.txt
    logic_version_path = temp_dir / "logic_version.txt"
    if logic_version_path.exists():
        try:
            logic_version_path.unlink()
            _logger.info(f"  删除逻辑版本缓存")
        except Exception as e:
            _logger.warning(f"  删除逻辑版本缓存失败: {e}")

    # 4. 清理数据库（先关闭所有连接）
    if db_path.exists():
        try:
            # 触发垃圾回收，关闭未使用的连接
            gc.collect()

            # 检查数据库是否仍在被占用
            try:
                test_conn = sqlite3.connect(str(db_path))
                test_conn.close()
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    _logger.warning(f"  数据库文件被占用，跳过删除: {db_path.name}")
                    _logger.warning(f"  请关闭所有使用该数据库的程序后重试")
                else:
                    raise

            # 文件可访问，尝试删除
            db_path.unlink()
            _logger.info(f"  删除数据库: {db_path.name}")
        except PermissionError as e:
            _logger.warning(f"  数据库文件被占用，无法删除: {db_path.name}")
            _logger.warning(f"  错误: {e}")
            _logger.warning(f"  建议: 重启程序或关闭其他使用该数据库的程序")
        except Exception as e:
            _logger.warning(f"  删除数据库失败: {e}")

    # 5. 清理其他JSON缓存
    if temp_dir.exists():
        try:
            for f in temp_dir.glob("*.json"):
                if f.name not in ["sequence.json"]:  # 已经单独处理
                    try:
                        f.unlink()
                        _logger.info(f"  删除缓存: {f.name}")
                    except Exception as e:
                        _logger.warning(f"  删除{f.name}失败: {e}")
        except Exception as e:
            _logger.warning(f"  清理JSON缓存失败: {e}")

    _logger.info("=== 缓存清理完成 ===")


class SmartCutController:
    """
    智能剪辑系统主控制器
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化控制器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = load_config(config_path)
        
        # 设置日志
        setup_logging(self.config)
        self.logger = logging.getLogger("smart_cut")
        
        # 初始化数据库
        db_path = self.config.get("database", {}).get("path", "storage/materials.db")
        self.db = Database(db_path)
        
        # 初始化组件
        self.purifier: Optional[VideoPurifier] = None
        self.planner: Optional[SequencePlanner] = None
        self.auto_cutter: Optional[VideoAutoCutter] = None
        
        # 配置参数
        self.raw_folder = self.config.get("paths", {}).get("raw_folder", "storage/materials-all")
        self.output_dir = self.config.get("paths", {}).get("output_dir", "temp")
        
        self.logger.info("SmartCutController 初始化完成")
    
    def _init_components(self):
        """初始化组件"""
        if self.purifier is None:
            self.logger.info("初始化 VideoPurifier...")
            self.purifier = VideoPurifier(self.db)
        
        if self.planner is None:
            self.logger.info("初始化 SequencePlanner...")
            embedding_model = EmbeddingModel()
            self.planner = SequencePlanner(self.db, embedding_model)
        
        if self.auto_cutter is None:
            self.logger.info("初始化 VideoAutoCutter...")
            render_config = self.config.get("render", {})
            self.auto_cutter = VideoAutoCutter(
                raw_folder=self.raw_folder,
                db_path=self.config.get("database", {}).get("path", "storage/materials.db"),
                output_dir=self.output_dir
            )
            # 应用渲染配置
            self.auto_cutter.TARGET_WIDTH = render_config.get("width", 1080)
            self.auto_cutter.TARGET_HEIGHT = render_config.get("height", 1920)
            self.auto_cutter.TARGET_FPS = render_config.get("fps", 30)
    
    def scan_materials(self, force_reprocess: bool = False) -> dict:
        """
        扫描和处理素材
        
        Args:
            force_reprocess: 是否强制重新处理所有素材
            
        Returns:
            扫描结果统计
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 1: 素材扫描与净化")
        self.logger.info("=" * 60)
        
        self._init_components()
        
        # 获取视频文件
        raw_path = PROJECT_ROOT / self.raw_folder
        if not raw_path.exists():
            self.logger.error(f"素材目录不存在: {raw_path}")
            return {"error": "素材目录不存在"}
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MOV', '.MP4', '.MP4'}
        video_files = []
        for ext in video_extensions:
            video_files.extend(raw_path.glob(f"*{ext}"))
        
        self.logger.info(f"发现 {len(video_files)} 个视频文件")
        
        # 处理每个文件
        a_roll_count = 0
        b_roll_count = 0
        skipped = 0
        
        for i, video_path in enumerate(sorted(video_files)):
            self.logger.info(f"[{i+1}/{len(video_files)}] 处理: {video_path.name}")
            
            # 检查是否需要重新处理
            current_mtime = os.path.getmtime(video_path)
            need_process = force_reprocess or not self.db.check_asset_fresh(
                str(video_path), current_mtime
            )
            
            if need_process:
                asset = self.purifier.purify(str(video_path), force_reprocess=force_reprocess)
                
                if asset.track_type == "A_ROLL":
                    a_roll_count += 1
                    self.logger.info(f"  -> A_ROLL, offset={asset.valid_start_offset:.3f}s")
                else:
                    b_roll_count += 1
                    self.logger.info(f"  -> B_ROLL")
            else:
                self.logger.info(f"  -> 使用缓存")
                skipped += 1
                
                # 统计类型
                db_asset = self.db.get_asset_by_path(str(video_path))
                if db_asset:
                    if db_asset.track_type == "A_ROLL":
                        a_roll_count += 1
                    else:
                        b_roll_count += 1
        
        result = {
            "total": len(video_files),
            "processed": len(video_files) - skipped,
            "skipped": skipped,
            "a_roll": a_roll_count,
            "b_roll": b_roll_count
        }
        
        self.logger.info("=== 素材扫描结果 ===")
        self.logger.info(f"总文件数: {result['total']}")
        self.logger.info(f"新处理: {result['processed']}")
        self.logger.info(f"使用缓存: {result['skipped']}")
        self.logger.info(f"A_ROLL: {result['a_roll']}")
        self.logger.info(f"B_ROLL: {result['b_roll']}")
        
        return result
    
    def plan_edl(self, script: str, force_replan: bool = False) -> list:
        """
        生成剪辑决策列表

        Args:
            script: 文案脚本
            force_replan: 是否强制重新规划（忽略缓存）

        Returns:
            EDL 列表
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 2: 文案语义编排")
        self.logger.info("=" * 60)

        self._init_components()

        # 检查文案缓存
        script_hash_path = PROJECT_ROOT / "temp" / "script_hash.txt"

        # 检查logic_version是否变化（触发ASR重新识别）
        from core.assembler import check_cache_validity, force_clean_cache, force_clean_sequence_cache
        logic_version_changed = not check_cache_validity()

        # 检查文案是否变化（触发序列重新规划）
        script_changed = not check_script_cache_valid(script, script_hash_path) or force_replan

        # 缓存清理标记（延迟到流程最后执行）
        self._cleanup_after_render_flag = {
            'logic_version_changed': logic_version_changed,
            'script_changed': script_changed
        }

        # 重新初始化数据库连接和扫描素材（logic_version变化时）
        if logic_version_changed:
            self.logger.info("检测到logic_version变化，将在渲染完成后清理ASR和序列缓存")
            # 关闭数据库连接并清理
            self._close_database_connections()
            force_clean_all_caches()
            # 重新初始化数据库连接
            db_path = self.config.get("database", {}).get("path", "storage/materials.db")
            self.db = Database(db_path)
            # 重新扫描素材
            self.logger.info("重新扫描和识别素材...")
            self.scan_materials(force_reprocess=True)
            # 重新加载planner的缓存
            if self.planner:
                self.planner._load_materials()
                self.logger.debug("已重新加载planner缓存")
        elif script_changed:
            self.logger.info("检测到文案变化（内容或顺序），将在渲染完成后清理序列缓存")
            # 仅删除序列缓存和文案哈希
            force_clean_sequence_cache()
            # 重新加载planner的缓存（确保获取最新素材）
            self.logger.info("保留现有素材数据库，仅重新规划序列")
            if self.planner:
                self.planner._load_materials()
                self.logger.debug("已重新加载planner缓存")

        # 执行规划
        edl = self.planner.plan(script)

        # 保存当前文案的哈希（供下次对比）
        save_script_hash(script, script_hash_path)

        # 打印结果
        self._print_edl(edl)

        return edl

    def _close_database_connections(self):
        """关闭所有数据库连接"""
        try:
            if self.db:
                self.db.close()
                self.logger.debug("已关闭数据库连接")
        except Exception as e:
            self.logger.warning(f"关闭数据库连接失败: {e}")

        # 清空planner的缓存（不需要显式关闭，因为planner使用的是共享的数据库）
        try:
            if self.planner:
                # 清空materials_cache，下次使用时会重新加载
                self.planner.materials_cache.clear()
                self.logger.debug("已清空planner缓存")
        except Exception as e:
            self.logger.warning(f"清空planner缓存失败: {e}")
    
    def _print_edl(self, edl: list):
        """打印 EDL 结果"""
        self.logger.info("")
        self.logger.info("=== 匹配结果详情 ===")
        self.logger.info(f"{'序号':<4} {'文案句子':<25} {'素材名':<20} {'类型':<8} {'得分':<6} {'原因':<15}")
        self.logger.info("-" * 95)
        
        for i, clip in enumerate(edl):
            text = clip.get("text", "")[:22] + "..." if len(clip.get("text", "")) > 22 else clip.get("text", "")
            
            if clip.get("missing"):
                self.logger.info(f"{i+1:<4} {text:<25} {'(无匹配)':<20} {'-':<8} {'-':<6} {'-':<15}")
            else:
                material_name = clip.get("video_name", "N/A")[:18]
                track_type = clip.get("track_type", "-")[:6]
                similarity = f"{clip.get('similarity', 0):.3f}"
                reason = clip.get("reason", "")
                
                if clip.get("is_b_roll"):
                    track_type += "[B]"
                
                if clip.get("fallback"):
                    reason = "[后备]"
                
                self.logger.info(f"{i+1:<4} {text:<25} {material_name:<20} {track_type:<8} {similarity:<6} {reason:<15}")
        
        # 统计
        a_count = sum(1 for e in edl if e.get("track_type") == "A_ROLL")
        b_count = sum(1 for e in edl if e.get("track_type") == "B_ROLL")
        missing = sum(1 for e in edl if e.get("missing", False))
        
        self.logger.info("-" * 95)
        self.logger.info(f"总计: {len(edl)} | A_ROLL: {a_count} | B_ROLL: {b_count} | 缺失: {missing}")
    
    def render(self, edl: list, output_name: str = "final_output.mp4") -> bool:
        """
        渲染最终视频
        
        Args:
            edl: 剪辑决策列表
            output_name: 输出文件名
            
        Returns:
            是否成功
        """
        self.logger.info("=" * 60)
        self.logger.info("Step 3: FFmpeg 物理合成")
        self.logger.info("=" * 60)
        
        self._init_components()

        # 保存 sequence.json 编排清单
        import json
        sequence_path = PROJECT_ROOT / "temp" / "sequence.json"
        with open(sequence_path, 'w', encoding='utf-8') as f:
            json.dump({
                "edl": edl,
                "total_clips": len(edl)
            }, f, ensure_ascii=False, indent=2)
        self.logger.info(f"sequence.json 已保存: {sequence_path}")

        # 执行渲染
        success = self.auto_cutter.render(edl, output_name)

        if success:
            self.logger.info(f"渲染完成: {output_name}")
        else:
            self.logger.error("渲染失败")

        return success
    
    def run(self, script: str, output_name: str = "final_output.mp4",
            force_reprocess: bool = False, force_replan: bool = False) -> bool:
        """
        执行完整流程

        Args:
            script: 文案脚本
            output_name: 输出文件名
            force_reprocess: 是否强制重新处理素材
            force_replan: 是否强制重新规划（忽略文案缓存）

        Returns:
            是否成功
        """
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("智能短视频剪辑系统启动")
        self.logger.info("=" * 60)
        
        # Step 1: 素材扫描
        self.scan_materials(force_reprocess=force_reprocess)
        
        # Step 2: 文案编排（自动检测文案变化）
        edl = self.plan_edl(script, force_replan=force_replan)

        # Step 3: 渲染
        success = self.render(edl, output_name)

        # 渲染完成后自动清理缓存和数据库
        # 暂时注释掉清理功能
        # if success:
        #     self._cleanup_after_render(output_name)

        # 统计耗时
        elapsed = time.time() - start_time
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"流程完成! 耗时: {elapsed:.1f} 秒")
        self.logger.info("=" * 60)

        return success

    def _cleanup_after_render(self, output_name: str) -> None:
        """执行完成后清理缓存和数据库"""
        import shutil

        self.logger.info("开始清理缓存和数据库...")

        # 检查是否有待清理标记
        cleanup_flag = getattr(self, '_cleanup_after_render_flag', {})
        logic_version_changed = cleanup_flag.get('logic_version_changed', False)
        script_changed = cleanup_flag.get('script_changed', False)

        # 1. 清理 temp 目录（保留最终输出文件）
        temp_dir = PROJECT_ROOT / "temp"
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                # 保留最终输出文件
                if item.is_file() and item.name == output_name:
                    continue
                # 保留 sequence.json（可选，用于调试）
                if item.is_file() and item.name == "sequence.json":
                    continue
                try:
                    if item.is_file():
                        item.unlink()
                        self.logger.info(f"  删除临时文件: {item.name}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        self.logger.info(f"  删除临时目录: {item.name}")
                except Exception as e:
                    self.logger.warning(f"  删除失败: {item.name}, {e}")

        # 2. 删除数据库文件（根据标记决定）
        db_path = PROJECT_ROOT / self.config.get("database", {}).get("path", "storage/materials.db")
        if logic_version_changed:
            # logic_version变化时删除数据库
            if db_path.exists():
                try:
                    # 先关闭数据库连接
                    if self.db:
                        self.db.close()
                    db_path.unlink()
                    self.logger.info(f"  删除数据库: {db_path.name}")
                except Exception as e:
                    self.logger.warning(f"  删除数据库失败: {db_path.name}, {e}")
        elif script_changed:
            # 文案变化时保留数据库，只删除序列缓存（已在上一步处理）
            self.logger.info("  保留素材数据库（文案变化仅重新规划序列）")

        # 3. 清理 logs 目录（可选）
        logs_dir = PROJECT_ROOT / self.config.get("paths", {}).get("logs_dir", "logs")
        if logs_dir.exists():
            try:
                for log_file in logs_dir.glob("*.log"):
                    log_file.unlink()
                    self.logger.info(f"  删除日志: {log_file.name}")
            except Exception as e:
                self.logger.warning(f"  删除日志失败: {e}")

        self.logger.info("清理完成!")
    
    def close(self):
        """关闭资源"""
        if self.db:
            self.db.close()
        self.logger.info("系统已关闭")


# ============ CLI 入口 ============
def main():
    """CLI 入口"""
    import argparse
    
    # 初始化硬件设备
    print("=" * 50)
    print("初始化硬件设备...")
    device_info = get_device_info()
    print(f"设备类型: {device_info['device_type']}")
    print(f"设备名称: {device_info['device_name']}")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description="智能短视频剪辑系统")
    parser.add_argument("-s", "--script", type=str, help="文案脚本文件路径")
    parser.add_argument("-o", "--output", type=str, default="final_output.mp4", help="输出文件名")
    parser.add_argument("-f", "--force", action="store_true", help="强制重新处理所有素材")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--replan", action="store_true", help="强制重新规划（忽略文案缓存）")
    
    args = parser.parse_args()
    
    # 如果指定 --cpu，强制使用 CPU
    if args.cpu:
        from core.hardware import init_device
        init_device(force_cpu=True)
    
    # 读取文案
    if args.script:
        script_file = PROJECT_ROOT / args.script
        if script_file.exists():
            with open(script_file, 'r', encoding='utf-8') as f:
                script = f.read()
        else:
            print(f"文案文件不存在: {script_file}")
            return
    else:
        # 默认完整文案（美的京东315活动）
        script = """不用确定我们京东315活动3C、家电政府补贴至高15%，还能跟美的代金券叠加使用
我再问一下
不用问,美的四大权益都可以享受
权益一
全屋智能家电套购
送至高1499元豪礼
权益二
空气机及无风感系列空调一价全包
至高省1399元
权益三
购指定型号享最高
价值3200元局改服务
权益四
上抖音团购领
美的专属优惠券
活动时间：2026年3月1日-3月31日
美的火三月福利，今天带你们薅小天鹅洗烘套装的羊毛，错过等一年！
美的50代150 还可以叠加京东活动真的是太划算了
首选这套本色洗烘套装！蓝氧护色黑科技太懂女生了——白T恤越洗越亮，彩色衣服不串色，洗完烘完直接能穿，不用再担心晒不干有异味。现在领50代150的券，叠加京东315补贴，算下来比平时便宜大几百！
重点来了！京东315还有新房全屋全套5折抢，老房焕新补贴直接省到家。买小天鹅 先用券减100，再享政府补贴，双重优惠叠加，这便宜不占白不占！
我在京东电器旗舰店等你们！想买洗烘套装的赶紧来，记得领50代150的券，叠加补贴真的超划算～"""
    
    # 创建控制器
    controller = SmartCutController(config_path=args.config)
    
    # 执行流程
    try:
        success = controller.run(script, args.output, args.force, args.replan)
        
        if success:
            print(f"\n视频生成成功: {args.output}")
            return 0
        else:
            print(f"\n视频生成失败")
            return 1
    finally:
        controller.close()


if __name__ == "__main__":
    exit(main())
