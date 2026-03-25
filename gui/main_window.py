"""
主窗口模块
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QTabWidget, QStatusBar, QMenuBar, QMenu,
                             QAction, QMessageBox, QFileDialog, QLabel, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QIcon, QPixmap

from gui.panels.project_panel import ProjectPanel
from gui.panels.preview_panel import PreviewPanel
from gui.panels.param_panel import ParamPanel
from gui.timeline.timeline_widget import TimelineWidget
from core.utils import load_config, save_config, setup_logger, ensure_dir
from core.audio_processor import AudioProcessor
from core.asr import ASR
from core.matcher import Matcher
from core.video_processor import VideoProcessor
from core.subtitle import SubtitleGenerator


logger = setup_logger("gui")


class WorkerThread(QThread):
    """后台工作线程"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            self.progress.emit(10, "正在处理...")
            result = self.task_func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Worker thread error: {e}")
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 加载配置
        self.config = load_config()
        
        # 初始化核心模块
        self._init_core_modules()
        
        # 项目数据
        self.project_data = {
            "text": "",
            "materials": [],  # 素材列表
            "segments": [],    # 选中的片段
            "result": None     # 最终结果
        }
        
        # 工作线程
        self.worker = None
        
        # 初始化UI
        self.init_ui()
        
        # 信号连接
        self._connect_signals()
        
        logger.info("MainWindow initialized")
    
    def _init_core_modules(self):
        """初始化核心模块（延迟加载模型）"""
        try:
            # 只初始化音频处理器和视频处理器（不需要下载模型）
            self.audio_processor = AudioProcessor()
            self.video_processor = VideoProcessor(
                target_resolution=(
                    self.config.get("video_resolution", {}).get("width", 1080),
                    self.config.get("video_resolution", {}).get("height", 1920)
                ),
                target_fps=self.config.get("video_fps", 30)
            )
            self.subtitle_generator = SubtitleGenerator()
            
            # 延迟加载ASR和Matcher（需要下载模型）
            self.asr_model = None
            self.matcher = None
            self._models_loaded = False
            
            logger.info("Core modules initialized (lazy loading)")
        except Exception as e:
            logger.error(f"初始化核心模块失败: {e}")
            QMessageBox.critical(self, "错误", f"初始化核心模块失败: {e}")
    
    def _ensure_models_loaded(self):
        """确保模型已加载（延迟加载）"""
        if self._models_loaded:
            return
        
        logger.info("Loading AI models...")
        try:
            # 加载ASR模型
            self.asr_model = ASR(self.config.get("whisper_model", "large"))
            logger.info("ASR model loaded")
            
            # 加载Matcher模型
            self.matcher = Matcher(self.config.get("embedding_model", "all-MiniLM-L6-v2"))
            logger.info("Matcher model loaded")
            
            self._models_loaded = True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("智能剪辑 - 抖音团购短视频生成工具")
        self.setGeometry(100, 100, 1600, 900)
        
        # 创建菜单栏
        self._create_menu_bar()
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧面板：项目面板
        self.project_panel = ProjectPanel()
        self.project_panel.setMinimumWidth(300)
        self.project_panel.setMaximumWidth(400)
        splitter.addWidget(self.project_panel)
        
        # 中间面板：预览区
        center_layout = QVBoxLayout()
        
        # 预览面板
        self.preview_panel = PreviewPanel()
        center_layout.addWidget(self.preview_panel, stretch=3)
        
        # 时间线
        self.timeline_widget = TimelineWidget()
        center_layout.addWidget(self.timeline_widget, stretch=1)
        
        center_widget = QWidget()
        center_widget.setLayout(center_layout)
        splitter.addWidget(center_widget)
        
        # 右侧面板：参数面板
        self.param_panel = ParamPanel()
        self.param_panel.setMinimumWidth(280)
        self.param_panel.setMaximumWidth(350)
        splitter.addWidget(self.param_panel)
        
        # 设置分割比例
        splitter.setSizes([300, 800, 300])
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self._create_status_bar()
    
    def _create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        new_action = QAction("新建项目", self)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("打开文案", self)
        open_action.triggered.connect(self.open_text_file)
        file_menu.addAction(open_action)
        
        open_folder_action = QAction("导入素材文件夹", self)
        open_folder_action.triggered.connect(self.import_materials_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("保存项目", self)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        export_action = QAction("导出视频", self)
        export_action.triggered.connect(self.export_video)
        file_menu.addAction(export_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("编辑")
        
        settings_action = QAction("参数设置", self)
        settings_action.triggered.connect(self.show_settings)
        edit_menu.addAction(settings_action)
        
        # 处理菜单
        process_menu = menubar.addMenu("处理")
        
        analyze_action = QAction("分析素材", self)
        analyze_action.triggered.connect(self.analyze_materials)
        process_menu.addAction(analyze_action)
        
        concat_action = QAction("生成视频", self)
        concat_action.triggered.connect(self.generate_video)
        process_menu.addAction(concat_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def _create_status_bar(self):
        """创建状态栏"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.statusBar.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar.addPermanentWidget(self.progress_bar)
    
    def _connect_signals(self):
        """连接信号"""
        # 文案更新
        self.project_panel.text_changed.connect(self.on_text_changed)
        
        # 素材添加
        self.project_panel.materials_added.connect(self.on_materials_added)
        
        # 素材移除
        self.project_panel.material_removed.connect(self.on_material_removed)
        
        # 参数更新
        self.param_panel.params_changed.connect(self.on_params_changed)
        
        # 时间线信号
        self.timeline_widget.segment_changed.connect(self.on_segment_changed)
        self.timeline_widget.segment_deleted.connect(self.on_segment_deleted)
        
        # 预览信号
        self.preview_panel.preview_requested.connect(self.on_preview_requested)
    
    # ==================== 槽函数 ====================
    
    def new_project(self):
        """新建项目"""
        self.project_data = {
            "text": "",
            "materials": [],
            "segments": [],
            "result": None
        }
        
        # 清空UI
        self.project_panel.clear()
        self.timeline_widget.clear()
        self.preview_panel.clear()
        
        self.status_label.setText("新建项目")
        logger.info("New project created")
    
    def open_text_file(self):
        """打开文案文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开文案", "", "文本文件 (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                self.project_data["text"] = text
                self.project_panel.set_text(text)
                self.status_label.setText(f"已加载文案: {Path(file_path).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载文案失败: {e}")
    
    def import_materials_folder(self):
        """导入素材文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择素材文件夹")
        
        if folder_path:
            # 查找视频文件
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(Path(folder_path).glob(f"*{ext}"))
                video_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
            
            if not video_files:
                QMessageBox.warning(self, "警告", "文件夹中没有找到视频文件")
                return
            
            # 添加素材
            for video_file in video_files:
                self.project_data["materials"].append({
                    "path": str(video_file),
                    "name": video_file.name,
                    "info": None,
                    "result": None
                })
            
            # 更新UI
            self.project_panel.update_materials(self.project_data["materials"])
            self.status_label.setText(f"已导入 {len(video_files)} 个素材")
    
    def save_project(self):
        """保存项目"""
        # 保存配置
        save_config(self.config)
        self.status_label.setText("项目已保存")
    
    def export_video(self):
        """导出视频"""
        if not self.project_data.get("segments"):
            QMessageBox.warning(self, "警告", "没有可导出的视频片段")
            return
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "导出视频", "", "MP4视频 (*.mp4)"
        )
        
        if output_path:
            self._export_video_async(output_path)
    
    def _export_video_async(self, output_path: str):
        """异步导出视频"""
        self.status_label.setText("正在导出视频...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        
        def task():
            # 收集所有片段
            segments = self.project_data.get("segments", [])
            
            # 裁剪每个片段
            cropped_files = []
            
            for i, seg in enumerate(segments):
                material_idx = seg.get("material_index")
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                
                material = self.project_data["materials"][material_idx]
                input_path = material["path"]
                
                output_file = f"temp/cropped_{i}.mp4"
                
                self.video_processor.crop_video(
                    input_path, output_file, start, end
                )
                cropped_files.append(output_file)
            
            self.progress_bar.setValue(50)
            
            # 拼接
            if len(cropped_files) == 1:
                final_video = cropped_files[0]
            else:
                final_video = "temp/concat.mp4"
                self.video_processor.concat_videos(cropped_files, final_video)
            
            self.progress_bar.setValue(70)
            
            # 生成字幕
            asr_result = self.project_data.get("result", {}).get("asr_result")
            
            if asr_result:
                ass_file = "temp/subtitle.ass"
                self.subtitle_generator.generate_ass(
                    asr_result, ass_file, self.config
                )
                
                # 嵌入字幕
                output_path_temp = output_path.replace(".mp4", "_nosub.mp4")
                import shutil
                shutil.copy(final_video, output_path_temp)
                
                self.subtitle_generator.burn_subtitle(
                    output_path_temp, ass_file, output_path
                )
            else:
                import shutil
                shutil.copy(final_video, output_path)
            
            self.progress_bar.setValue(100)
            
            return output_path
        
        self.worker = WorkerThread(task)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_export_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    def show_settings(self):
        """显示设置对话框"""
        self.param_panel.show_settings()
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, "关于",
            "智能剪辑 - 抖音团购短视频生成工具\n\n"
            "自动完成素材筛选、音频处理、视频拼接和字幕生成"
        )
    
    # ==================== 信号处理 ====================
    
    def on_text_changed(self, text: str):
        """文案更改"""
        self.project_data["text"] = text
        logger.debug(f"Text changed: {len(text)} chars")
    
    def on_materials_added(self, materials: List[Dict]):
        """素材添加"""
        self.project_data["materials"].extend(materials)
        logger.debug(f"Materials added: {len(materials)}")
    
    def on_material_removed(self, index: int):
        """素材移除"""
        if 0 <= index < len(self.project_data["materials"]):
            self.project_data["materials"].pop(index)
            logger.debug(f"Material removed: {index}")
    
    def on_params_changed(self, params: Dict):
        """参数更改"""
        self.config.update(params)
        logger.debug(f"Params updated: {list(params.keys())}")
    
    def on_segment_changed(self, segment: Dict):
        """片段更改"""
        # 更新片段数据
        pass
    
    def on_segment_deleted(self, index: int):
        """片段删除"""
        if 0 <= index < len(self.project_data["segments"]):
            self.project_data["segments"].pop(index)
            logger.debug(f"Segment deleted: {index}")
    
    def on_preview_requested(self, timestamp: float):
        """预览请求"""
        pass
    
    # ==================== 处理流程 ====================
    
    def analyze_materials(self):
        """分析素材"""
        if not self.project_data["text"]:
            QMessageBox.warning(self, "警告", "请先输入文案")
            return
        
        if not self.project_data["materials"]:
            QMessageBox.warning(self, "警告", "请先导入素材")
            return
        
        # 确保模型已加载
        try:
            self._ensure_models_loaded()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {e}")
            return
        
        self.status_label.setText("正在分析素材...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        def task():
            target_text = self.project_data["text"]
            results = []
            
            total = len(self.project_data["materials"])
            
            for i, material in enumerate(self.project_data["materials"]):
                self.progress_bar.setValue(int((i / total) * 80))
                
                result = self.matcher.process_single_material(
                    material["path"],
                    target_text,
                    self.asr_model,
                    self.audio_processor,
                    silence_threshold=self.config.get("silence_threshold", 1.5)
                )
                
                results.append(result)
                
                # 更新素材信息
                material["result"] = result
            
            # 决策
            self.progress_bar.setValue(90)
            
            decision = self.matcher.decide_best_materials(
                results,
                target_text,
                single_threshold=self.config.get("single_threshold", 0.85)
            )
            
            self.project_data["segments"] = decision.get("segments", [])
            
            return decision
        
        self.worker = WorkerThread(task)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_analyze_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    def generate_video(self):
        """生成视频"""
        if not self.project_data.get("segments"):
            QMessageBox.warning(self, "警告", "请先分析素材")
            return
        
        self.status_label.setText("正在生成视频...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 调用导出
        output_path, _ = QFileDialog.getSaveFileName(
            self, "保存视频", "", "MP4视频 (*.mp4)"
        )
        
        if output_path:
            self._export_video_async(output_path)
    
    # ==================== 进度回调 ====================
    
    def _on_progress(self, value: int, message: str):
        """进度更新"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def _on_analyze_finished(self, result: Any):
        """分析完成"""
        self.progress_bar.setVisible(False)
        
        mode = result.get("mode", "none")
        
        if mode == "single":
            self.status_label.setText("单素材模式")
        elif mode == "concat":
            self.status_label.setText("多素材拼接模式")
        else:
            self.status_label.setText("未找到匹配内容")
        
        # 更新时间线
        self.timeline_widget.set_segments(self.project_data["segments"])
        
        logger.info(f"Analysis finished: {mode}")
    
    def _on_export_finished(self, result: Any):
        """导出完成"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"导出完成: {result}")
        
        QMessageBox.information(self, "完成", f"视频已导出到:\n{result}")
    
    def _on_error(self, error: str):
        """错误处理"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("处理失败")
        
        QMessageBox.critical(self, "错误", f"处理失败:\n{error}")
        logger.error(f"Processing error: {error}")
    
    def closeEvent(self, event):
        """关闭事件"""
        # 保存配置
        save_config(self.config)
        
        # 清理临时文件
        from core.utils import clean_temp_files
        clean_temp_files()
        
        event.accept()
