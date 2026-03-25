"""
预览面板
提供视频预览功能
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSlider, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget


class PreviewPanel(QWidget):
    """预览面板"""
    
    preview_requested = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 视频显示区
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(640, 360)
        self.video_widget.setStyleSheet("background-color: #000;")
        layout.addWidget(self.video_widget)
        
        # 播放控制区
        control_layout = QHBoxLayout()
        
        # 播放/暂停按钮
        self.play_btn = QPushButton("播放")
        self.play_btn.setMaximumWidth(60)
        self.play_btn.clicked.connect(self._on_play_clicked)
        control_layout.addWidget(self.play_btn)
        
        # 时间标签
        self.time_label = QLabel("00:00 / 00:00")
        control_layout.addWidget(self.time_label)
        
        # 进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        control_layout.addWidget(self.slider)
        
        layout.addLayout(control_layout)
        
        # 媒体播放器
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.video_widget)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.stateChanged.connect(self._on_state_changed)
        
        self.is_slider_pressed = False
    
    def load_video(self, path: str):
        """加载视频"""
        try:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
            self.player.play()
            self.player.pause()
        except Exception as e:
            logger.error(f"Load video error: {e}")
    
    def _on_play_clicked(self):
        """播放/暂停"""
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()
    
    def _on_position_changed(self, position: int):
        """播放位置改变"""
        if not self.is_slider_pressed:
            self.slider.setValue(position)
        
        # 更新时间
        duration = self.player.duration()
        current = self._format_time(position)
        total = self._format_time(duration)
        self.time_label.setText(f"{current} / {total}")
    
    def _on_duration_changed(self, duration: int):
        """时长改变"""
        self.slider.setRange(0, duration)
    
    def _on_state_changed(self, state):
        """播放状态改变"""
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setText("暂停")
        else:
            self.play_btn.setText("播放")
    
    def _on_slider_pressed(self):
        """进度条按下"""
        self.is_slider_pressed = True
    
    def _on_slider_released(self):
        """进度条释放"""
        self.is_slider_pressed = False
        position = self.slider.value()
        self.player.setPosition(position)
    
    def _on_slider_moved(self, value: int):
        """进度条拖动"""
        if self.is_slider_pressed:
            current = self._format_time(value)
            duration = self._format_time(self.player.duration())
            self.time_label.setText(f"{current} / {duration}")
    
    def _format_time(self, ms: int) -> str:
        """格式化时间"""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def clear(self):
        """清空"""
        self.player.stop()
        self.player.setMedia(QMediaContent())
        self.time_label.setText("00:00 / 00:00")
        self.slider.setValue(0)
