"""
时间线控件
提供片段显示、拖拽调整、裁剪等功能
"""
from typing import List, Dict, Any, Optional

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QScrollArea, QFrame, QPushButton, QMenu,
                             QGraphicsView, QGraphicsScene, QGraphicsRectItem,
                             QGraphicsTextItem)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QColor, QPen, QBrush, QFont, QPainter


class SegmentItem(QGraphicsRectItem):
    """片段图形项"""
    
    def __init__(self, segment: Dict, index: int, parent=None):
        super().__init__(parent)
        self.segment = segment
        self.index = index
        self.setFlag(QGraphicsRectItem.ItemIsMovable)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable)
        self.setBrush(QColor(70, 130, 180))
        self.setPen(QPen(QColor(30, 60, 100), 2))
    
    def get_segment_info(self) -> Dict:
        """获取片段信息"""
        return self.segment


class TimelineWidget(QWidget):
    """时间线控件"""
    
    segment_changed = pyqtSignal(dict)
    segment_deleted = pyqtSignal(int)
    segment_selected = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.segments = []
        self.pixels_per_second = 50  # 每秒对应像素
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标题栏
        header = QHBoxLayout()
        
        title = QLabel("时间线")
        title.setStyleSheet("font-weight: bold;")
        header.addWidget(title)
        
        header.addStretch()
        
        # 清空按钮
        self.clear_btn = QPushButton("清空")
        self.clear_btn.setMaximumWidth(60)
        self.clear_btn.clicked.connect(self.clear)
        header.addWidget(self.clear_btn)
        
        layout.addLayout(header)
        
        # 图形视图
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setMinimumHeight(120)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        layout.addWidget(self.view)
        
        # 片段列表显示
        self.info_label = QLabel("暂无片段")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
    
    def set_segments(self, segments: List[Dict]):
        """设置片段"""
        self.segments = segments
        self._render_segments()
        self._update_info()
    
    def _render_segments(self):
        """渲染片段"""
        self.scene.clear()
        
        # 绘制时间轴背景
        self.scene.addRect(0, 0, max(2000, len(self.segments) * 300), 100, 
                          QPen(Qt.NoPen), QBrush(QColor(240, 240, 240)))
        
        # 绘制刻度
        for i in range(20):
            x = i * self.pixels_per_second * 10
            self.scene.addLine(x, 0, x, 10, QPen(Qt.black, 1))
            
            # 时间标签
            text = self.scene.addText(f"{i*10}s")
            text.setPos(x + 2, 10)
        
        # 绘制片段
        y_offset = 20
        for i, seg in enumerate(self.segments):
            # 计算片段宽度
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            duration = end - start
            
            width = max(30, duration * self.pixels_per_second)
            
            # 片段矩形
            rect = QRectF(i * 150 + 10, y_offset, width, 60)
            
            item = SegmentItem(seg, i)
            item.setRect(rect)
            self.scene.addItem(item)
            
            # 片段标签
            text = seg.get("text", "")[:20]
            label = QGraphicsTextItem(text, item)
            label.setPos(rect.x() + 5, rect.y() + 5)
            label.setDefaultTextColor(Qt.white)
            
            # 时长标签
            duration_text = f"{duration:.1f}s"
            duration_label = QGraphicsTextItem(duration_text, item)
            duration_label.setPos(rect.x() + 5, rect.y() + 40)
            duration_label.setDefaultTextColor(Qt.white)
        
        # 更新场景大小
        scene_width = max(2000, len(self.segments) * 150 + 100)
        self.scene.setSceneRect(0, 0, scene_width, 120)
    
    def _update_info(self):
        """更新信息"""
        if not self.segments:
            self.info_label.setText("暂无片段")
        else:
            total_duration = sum(
                seg.get("end", 0) - seg.get("start", 0) 
                for seg in self.segments
            )
            self.info_label.setText(
                f"片段数: {len(self.segments)} | 总时长: {total_duration:.1f}秒"
            )
    
    def clear(self):
        """清空"""
        self.segments.clear()
        self.scene.clear()
        self._update_info()
    
    def add_segment(self, segment: Dict):
        """添加片段"""
        self.segments.append(segment)
        self._render_segments()
        self._update_info()
    
    def remove_segment(self, index: int):
        """移除片段"""
        if 0 <= index < len(self.segments):
            self.segments.pop(index)
            self._render_segments()
            self._update_info()
            self.segment_deleted.emit(index)
    
    def update_segment(self, index: int, segment: Dict):
        """更新片段"""
        if 0 <= index < len(self.segments):
            self.segments[index] = segment
            self._render_segments()
            self._update_info()
            self.segment_changed.emit(segment)
    
    def get_segments(self) -> List[Dict]:
        """获取所有片段"""
        return self.segments
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        super().mousePressEvent(event)
        
        # 右键菜单
        if event.button() == Qt.RightButton:
            self._show_context_menu(event.globalPos())
    
    def _show_context_menu(self, pos):
        """显示右键菜单"""
        menu = QMenu(self)
        
        delete_action = menu.addAction("删除")
        split_action = menu.addAction("拆分")
        extend_action = menu.addAction("延长")
        
        action = menu.exec_(pos)
        
        if action == delete_action:
            # 删除选中的片段
            pass
        elif action == split_action:
            # 拆分
            pass
        elif action == extend_action:
            # 延长
            pass
