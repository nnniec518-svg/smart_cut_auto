"""
项目面板
提供文案编辑和素材列表功能
"""
from pathlib import Path
from typing import List, Dict, Any

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTextEdit, QPushButton, QListWidget, QListWidgetItem,
                             QAbstractItemView, QMessageBox, QFileDialog)
from PyQt5.QtCore import pyqtSignal, Qt


class ProjectPanel(QWidget):
    """项目面板"""
    
    # 信号
    text_changed = pyqtSignal(str)
    materials_added = pyqtSignal(list)
    material_removed = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 文案编辑区
        text_label = QLabel("文案内容")
        text_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(text_label)
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("请输入口播文案...")
        self.text_edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.text_edit)
        
        # 按钮区
        btn_layout = QHBoxLayout()
        
        self.load_text_btn = QPushButton("加载文案")
        self.load_text_btn.clicked.connect(self._on_load_text)
        btn_layout.addWidget(self.load_text_btn)
        
        layout.addLayout(btn_layout)
        
        # 分隔
        layout.addSpacing(10)
        
        # 素材列表
        materials_label = QLabel("素材列表")
        materials_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(materials_label)
        
        self.materials_list = QListWidget()
        self.materials_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.materials_list.setAlternatingRowColors(True)
        self.materials_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.materials_list)
        
        # 素材按钮
        materials_btn_layout = QHBoxLayout()
        
        self.import_btn = QPushButton("导入素材")
        self.import_btn.clicked.connect(self._on_import_materials)
        materials_btn_layout.addWidget(self.import_btn)
        
        self.remove_btn = QPushButton("移除选中")
        self.remove_btn.clicked.connect(self._on_remove_selected)
        materials_btn_layout.addWidget(self.remove_btn)
        
        layout.addLayout(materials_btn_layout)
        
        # 统计信息
        self.stats_label = QLabel("素材数: 0")
        layout.addWidget(self.stats_label)
        
        layout.addStretch()
    
    def _on_text_changed(self):
        """文案更改"""
        text = self.text_edit.toPlainText()
        self.text_changed.emit(text)
    
    def _on_load_text(self):
        """加载文案文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开文案", "", "文本文件 (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                self.text_edit.setPlainText(text)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败: {e}")
    
    def _on_import_materials(self):
        """导入素材文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择素材文件夹", ""
        )
        
        if folder_path:
            # 查找视频文件
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
            video_files = []
            
            folder = Path(folder_path)
            for ext in video_extensions:
                video_files.extend(folder.glob(f"*{ext}"))
                video_files.extend(folder.glob(f"*{ext.upper()}"))
            
            if not video_files:
                QMessageBox.warning(self, "提示", "文件夹中没有找到视频文件")
                return
            
            materials = []
            file_paths = []
            for file_path in video_files:
                materials.append({
                    "path": str(file_path),
                    "name": file_path.name,
                    "info": None,
                    "result": None
                })
                file_paths.append(str(file_path))
            
            self.materials_added.emit(materials)
            self._update_materials_list(file_paths)
    
    def _on_item_double_clicked(self, item):
        """双击素材项"""
        # 可以打开预览
        pass
    
    def _on_remove_selected(self):
        """移除选中项"""
        selected = self.materials_list.selectedIndexes()
        if selected:
            # 从后往前删除
            for index in sorted(selected, reverse=True):
                self.materials_list.takeItem(index.row())
                self.material_removed.emit(index.row())
            
            self._update_stats()
    
    def _update_materials_list(self, materials: List[str]):
        """更新素材列表"""
        for material in materials:
            item = QListWidgetItem(Path(material).name)
            item.setData(Qt.UserRole, material)
            self.materials_list.addItem(item)
        
        self._update_stats()
    
    def _update_stats(self):
        """更新统计"""
        count = self.materials_list.count()
        self.stats_label.setText(f"素材数: {count}")
    
    def set_text(self, text: str):
        """设置文案"""
        self.text_edit.setPlainText(text)
    
    def get_text(self) -> str:
        """获取文案"""
        return self.text_edit.toPlainText()
    
    def update_materials(self, materials: List[Dict]):
        """更新素材列表"""
        self.materials_list.clear()
        for material in materials:
            item = QListWidgetItem(material.get("name", ""))
            item.setData(Qt.UserRole, material.get("path", ""))
            self.materials_list.addItem(item)
        
        self._update_stats()
    
    def clear(self):
        """清空"""
        self.text_edit.clear()
        self.materials_list.clear()
        self._update_stats()
