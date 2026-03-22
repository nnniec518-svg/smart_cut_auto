"""
预览面板
提供视频预览功能
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSlider, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget


class PreviewPanel(QWidget):
    """预览面板"""
    
    preview_requested = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self._video_capture = None
        self._timer = None
        self._is_playing = False
        self._video_path = None
        self._media_player = None
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 视频显示区 - 使用QLabel替代QVideoWidget
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: #000;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("未加载视频")
        layout.addWidget(self.video_label)
        
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
        
        self.is_slider_pressed = False
        
        # 初始化音频播放器
        try:
            self._media_player = QMediaPlayer()
            self._media_player.setVolume(100)
        except Exception as e:
            print(f"Failed to initialize media player: {e}")
            self._media_player = None
    
    def load_video(self, path: str):
        """加载视频"""
        try:
            # 关闭之前的视频
            self.stop()
            
            # 打开视频文件
            self._video_capture = cv2.VideoCapture(path)
            
            if not self._video_capture.isOpened():
                print(f"Failed to open video: {path}")
                return
            
            self._video_path = path
            
            # 获取视频信息
            fps = self._video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self._duration_ms = int(frame_count / fps * 1000) if fps > 0 else 0
            
            # 设置音频播放器
            if self._media_player:
                from PyQt5.QtCore import QUrl
                self._media_player.setMedia(QUrl.fromLocalFile(path))
            
            # 设置进度条范围
            self.slider.setRange(0, self._duration_ms)
            
            # 显示第一帧
            self._update_frame()
            
            # 更新时间标签
            self.time_label.setText(f"00:00 / {self._format_time(self._duration_ms)}")
            
        except Exception as e:
            print(f"Load video error: {e}")
    
    def _update_frame(self):
        """更新视频帧"""
        if self._video_capture is None or not self._video_capture.isOpened():
            return
        
        ret, frame = self._video_capture.read()
        
        if ret:
            # 获取当前帧位置
            current_pos = int(self._video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            current_ms = int(self._video_capture.get(cv2.CAP_PROP_POS_MSEC))
            
            # 更新进度条
            if not self.is_slider_pressed:
                self.slider.setValue(current_ms)
            
            # 更新时间标签
            self.time_label.setText(f"{self._format_time(current_ms)} / {self._format_time(self._duration_ms)}")
            
            # 转换颜色空间 BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 调整大小以适应显示区
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            
            # 创建QImage并显示
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 缩放到显示区大小
            label_size = self.video_label.size()
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
        else:
            # 视频结束，循环播放
            if self._is_playing:
                self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if self._media_player:
                    self._media_player.setPosition(0)
                    self._media_player.play()
                self._update_frame()
    
    def _on_play_clicked(self):
        """播放/暂停"""
        if self._is_playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        """播放"""
        if self._video_capture is None or not self._video_capture.isOpened():
            return
        
        self._is_playing = True
        self.play_btn.setText("暂停")
        
        # 同步音频播放位置
        if self._media_player:
            current_ms = self.slider.value()
            self._media_player.setPosition(current_ms)
            self._media_player.play()
        
        # 启动定时器
        fps = self._video_capture.get(cv2.CAP_PROP_FPS)
        interval = int(1000 / fps) if fps > 0 else 33
        
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_frame)
        self._timer.start(interval)
    
    def pause(self):
        """暂停"""
        self._is_playing = False
        self.play_btn.setText("播放")
        
        # 暂停音频
        if self._media_player:
            self._media_player.pause()
        
        if self._timer:
            self._timer.stop()
    
    def stop(self):
        """停止"""
        self.pause()
        
        # 停止音频
        if self._media_player:
            self._media_player.stop()
        
        if self._video_capture:
            self._video_capture.release()
            self._video_capture = None
        
        self.video_label.setText("未加载视频")
        self.video_label.setPixmap(QPixmap())
        self.time_label.setText("00:00 / 00:00")
        self.slider.setValue(0)
    
    def seek(self, position_ms: int):
        """跳转到指定位置（毫秒）- 使用帧号跳转优化性能"""
        if self._video_capture and self._video_capture.isOpened():
            # 使用帧号跳转比毫秒跳转更快
            fps = self._video_capture.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                target_frame = int(position_ms / 1000.0 * fps)
                self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            else:
                self._video_capture.set(cv2.CAP_PROP_POS_MSEC, position_ms)
            # 同步音频位置
            if self._media_player:
                self._media_player.setPosition(position_ms)
            self._update_frame()
    
    def _on_slider_pressed(self):
        """进度条按下"""
        self.is_slider_pressed = True
        if self._is_playing:
            self.pause()
    
    def _on_slider_released(self):
        """进度条释放"""
        self.is_slider_pressed = False
        position = self.slider.value()
        self.seek(position)
    
    def _on_slider_moved(self, value: int):
        """进度条拖动"""
        if self._is_playing:
            self.seek(value)
            current = self._format_time(value)
            duration = self._format_time(self._duration_ms)
            self.time_label.setText(f"{current} / {duration}")
    
    def _format_time(self, ms: int) -> str:
        """格式化时间"""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def clear(self):
        """清空"""
        self.stop()
