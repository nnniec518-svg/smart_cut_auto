"""
参数面板
提供各种参数调整功能
"""
from typing import Dict, Any

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTabWidget, QDoubleSpinBox, QCheckBox,
                             QGroupBox, QFormLayout, QSpinBox, QLineEdit,
                             QPushButton, QComboBox, QScrollArea)
from PyQt5.QtCore import pyqtSignal, Qt


class ParamPanel(QWidget):
    """参数面板"""
    
    params_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.params = {}
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("参数设置")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # 匹配参数
        match_group = self._create_match_group()
        scroll_layout.addWidget(match_group)
        
        # 音频参数
        audio_group = self._create_audio_group()
        scroll_layout.addWidget(audio_group)
        
        # 视频参数
        video_group = self._create_video_group()
        scroll_layout.addWidget(video_group)
        
        # 字幕参数
        subtitle_group = self._create_subtitle_group()
        scroll_layout.addWidget(subtitle_group)
        
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # 应用按钮
        self.apply_btn = QPushButton("应用参数")
        self.apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(self.apply_btn)
    
    def _create_match_group(self) -> QGroupBox:
        """创建匹配参数组"""
        group = QGroupBox("匹配参数")
        layout = QFormLayout()
        
        # 单素材阈值
        self.single_threshold = QDoubleSpinBox()
        self.single_threshold.setRange(0, 1)
        self.single_threshold.setSingleStep(0.05)
        self.single_threshold.setValue(0.85)
        layout.addRow("单素材阈值:", self.single_threshold)
        
        # 句子匹配阈值
        self.sentence_threshold = QDoubleSpinBox()
        self.sentence_threshold.setRange(0, 1)
        self.sentence_threshold.setSingleStep(0.05)
        self.sentence_threshold.setValue(0.6)
        layout.addRow("句子匹配阈值:", self.sentence_threshold)
        
        # 静音阈值
        self.silence_threshold = QDoubleSpinBox()
        self.silence_threshold.setRange(0.1, 10)
        self.silence_threshold.setSingleStep(0.1)
        self.silence_threshold.setValue(1.5)
        layout.addRow("静音阈值(秒):", self.silence_threshold)
        
        group.setLayout(layout)
        return group
    
    def _create_audio_group(self) -> QGroupBox:
        """创建音频参数组"""
        group = QGroupBox("音频参数")
        layout = QFormLayout()
        
        # 降噪
        self.noise_reduce = QCheckBox()
        self.noise_reduce.setChecked(True)
        layout.addRow("启用降噪:", self.noise_reduce)
        
        # 降噪强度
        self.noise_strength = QDoubleSpinBox()
        self.noise_strength.setRange(0, 1)
        self.noise_strength.setSingleStep(0.1)
        self.noise_strength.setValue(0.3)
        layout.addRow("降噪强度:", self.noise_strength)
        
        # 切除开头提示音
        self.remove_leading = QCheckBox()
        self.remove_leading.setChecked(True)
        layout.addRow("切除开头提示音:", self.remove_leading)
        
        # 切除结尾静音
        self.remove_trailing = QCheckBox()
        self.remove_trailing.setChecked(True)
        layout.addRow("切除结尾静音:", self.remove_trailing)
        
        # 音量归一化
        self.normalize_audio = QCheckBox()
        self.normalize_audio.setChecked(True)
        layout.addRow("音量归一化:", self.normalize_audio)
        
        group.setLayout(layout)
        return group
    
    def _create_video_group(self) -> QGroupBox:
        """创建视频参数组"""
        group = QGroupBox("视频参数")
        layout = QFormLayout()
        
        # 分辨率
        res_layout = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(480, 3840)
        self.width_spin.setValue(1080)
        self.width_spin.setSuffix(" px")
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(480, 3840)
        self.height_spin.setValue(1920)
        self.height_spin.setSuffix(" px")
        
        res_layout.addWidget(self.width_spin)
        res_layout.addWidget(QLabel("×"))
        res_layout.addWidget(self.height_spin)
        layout.addRow("分辨率:", res_layout)
        
        # 帧率
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(15, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" fps")
        layout.addRow("帧率:", self.fps_spin)
        
        # 淡入淡出
        self.add_fade = QCheckBox()
        self.add_fade.setChecked(True)
        layout.addRow("添加过渡:", self.add_fade)
        
        # 过渡时长
        self.fade_duration = QDoubleSpinBox()
        self.fade_duration.setRange(0.1, 3)
        self.fade_duration.setSingleStep(0.1)
        self.fade_duration.setValue(0.5)
        layout.addRow("过渡时长(秒):", self.fade_duration)
        
        group.setLayout(layout)
        return group
    
    def _create_subtitle_group(self) -> QGroupBox:
        """创建字幕参数组"""
        group = QGroupBox("字幕参数")
        layout = QFormLayout()
        
        # 启用字幕
        self.enable_subtitle = QCheckBox()
        self.enable_subtitle.setChecked(True)
        layout.addRow("启用字幕:", self.enable_subtitle)
        
        # 字体
        self.font_name = QComboBox()
        self.font_name.addItems(["Microsoft YaHei", "SimHei", "Source Han Sans SC", "Noto Sans SC"])
        layout.addRow("字体:", self.font_name)
        
        # 字体大小比例
        self.font_size_ratio = QDoubleSpinBox()
        self.font_size_ratio.setRange(0.03, 0.15)
        self.font_size_ratio.setSingleStep(0.01)
        self.font_size_ratio.setValue(0.07)
        layout.addRow("字体大小比例:", self.font_size_ratio)
        
        # 颜色
        self.subtitle_color = QLineEdit()
        self.subtitle_color.setText("#FFFFFF")
        layout.addRow("字体颜色:", self.subtitle_color)
        
        # 描边颜色
        self.stroke_color = QLineEdit()
        self.stroke_color.setText("#000000")
        layout.addRow("描边颜色:", self.stroke_color)
        
        # 描边宽度
        self.stroke_width = QDoubleSpinBox()
        self.stroke_width.setRange(0, 5)
        self.stroke_width.setSingleStep(0.5)
        self.stroke_width.setValue(1.5)
        layout.addRow("描边宽度:", self.stroke_width)
        
        # 背景框
        self.subtitle_bg = QCheckBox()
        self.subtitle_bg.setChecked(True)
        layout.addRow("背景框:", self.subtitle_bg)
        
        # 边距
        self.margin = QSpinBox()
        self.margin.setRange(0, 50)
        self.margin.setValue(10)
        layout.addRow("边距:", self.margin)
        
        group.setLayout(layout)
        return group
    
    def _on_apply(self):
        """应用参数"""
        self._collect_params()
        self.params_changed.emit(self.params)
    
    def _collect_params(self):
        """收集参数"""
        self.params = {
            # 匹配参数
            "single_threshold": self.single_threshold.value(),
            "sentence_threshold": self.sentence_threshold.value(),
            "silence_threshold": self.silence_threshold.value(),
            
            # 音频参数
            "noise_reduce": self.noise_reduce.isChecked(),
            "noise_reduce_strength": self.noise_strength.value(),
            "remove_leading_sound": self.remove_leading.isChecked(),
            "remove_trailing_silence": self.remove_trailing.isChecked(),
            "normalize_audio": self.normalize_audio.isChecked(),
            
            # 视频参数
            "video_resolution": {
                "width": self.width_spin.value(),
                "height": self.height_spin.value()
            },
            "video_fps": self.fps_spin.value(),
            "add_fade": self.add_fade.isChecked(),
            "fade_duration": self.fade_duration.value(),
            
            # 字幕参数
            "enable_subtitle": self.enable_subtitle.isChecked(),
            "subtitle_font": self.font_name.currentText(),
            "subtitle_size_ratio": self.font_size_ratio.value(),
            "subtitle_color": self.subtitle_color.text(),
            "subtitle_stroke_color": self.stroke_color.text(),
            "subtitle_stroke_width": self.stroke_width.value(),
            "subtitle_background": self.subtitle_bg.isChecked(),
            "subtitle_margin": self.margin.value()
        }
    
    def set_params(self, params: Dict[str, Any]):
        """设置参数"""
        self.params = params
        
        # 匹配参数
        self.single_threshold.setValue(params.get("single_threshold", 0.85))
        self.sentence_threshold.setValue(params.get("sentence_threshold", 0.6))
        self.silence_threshold.setValue(params.get("silence_threshold", 1.5))
        
        # 音频参数
        self.noise_reduce.setChecked(params.get("noise_reduce", True))
        self.noise_strength.setValue(params.get("noise_reduce_strength", 0.3))
        self.remove_leading.setChecked(params.get("remove_leading_sound", True))
        self.remove_trailing.setChecked(params.get("remove_trailing_silence", True))
        self.normalize_audio.setChecked(params.get("normalize_audio", True))
        
        # 视频参数
        res = params.get("video_resolution", {"width": 1080, "height": 1920})
        self.width_spin.setValue(res.get("width", 1080))
        self.height_spin.setValue(res.get("height", 1920))
        self.fps_spin.setValue(params.get("video_fps", 30))
        self.add_fade.setChecked(params.get("add_fade", True))
        self.fade_duration.setValue(params.get("fade_duration", 0.5))
        
        # 字幕参数
        self.enable_subtitle.setChecked(params.get("enable_subtitle", True))
        self.font_name.setCurrentText(params.get("subtitle_font", "Microsoft YaHei"))
        self.font_size_ratio.setValue(params.get("subtitle_size_ratio", 0.07))
        self.subtitle_color.setText(params.get("subtitle_color", "#FFFFFF"))
        self.stroke_color.setText(params.get("subtitle_stroke_color", "#000000"))
        self.stroke_width.setValue(params.get("subtitle_stroke_width", 1.5))
        self.subtitle_bg.setChecked(params.get("subtitle_background", True))
        self.margin.setValue(params.get("subtitle_margin", 10))
    
    def get_params(self) -> Dict[str, Any]:
        """获取参数"""
        if not self.params:
            self._collect_params()
        return self.params
    
    def show_settings(self):
        """显示设置对话框"""
        # 弹出对话框让用户确认参数
        self._collect_params()
        self.params_changed.emit(self.params)
