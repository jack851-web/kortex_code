"""
控制面板组件
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QGroupBox, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal


class ControlPanel(QWidget):
    """控制面板 - 任务控制按钮"""
    
    # 信号
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    resume_clicked = pyqtSignal()
    skip_clicked = pyqtSignal()
    next_task_clicked = pyqtSignal()
    retry_clicked = pyqtSignal()
    complete_task_clicked = pyqtSignal()
    episode_changed = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 任务控制组
        task_group = QGroupBox("任务控制")
        task_layout = QVBoxLayout(task_group)
        
        # 开始/停止Episode按钮
        btn_layout = QHBoxLayout()
        self._start_btn = QPushButton("开启 Episode")
        self._start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self._start_btn.clicked.connect(self.start_clicked.emit)

        self._stop_btn = QPushButton("关闭 Episode")
        self._stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        self._stop_btn.setEnabled(False)

        btn_layout.addWidget(self._start_btn)
        btn_layout.addWidget(self._stop_btn)
        task_layout.addLayout(btn_layout)
        
        # 执行下一个任务按钮
        self._next_task_btn = QPushButton("执行下一个任务")
        self._next_task_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self._next_task_btn.clicked.connect(self.next_task_clicked.emit)
        self._next_task_btn.setEnabled(False)
        task_layout.addWidget(self._next_task_btn)
        
        # 重做任务按钮
        self._retry_btn = QPushButton("重做当前任务")
        self._retry_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self._retry_btn.clicked.connect(self.retry_clicked.emit)
        self._retry_btn.setEnabled(False)
        task_layout.addWidget(self._retry_btn)

        # 手动确认任务完成按钮（新模式）
        self._complete_task_btn = QPushButton("抓取任务完毕")
        self._complete_task_btn.setStyleSheet("background-color: #673AB7; color: white; font-weight: bold;")
        self._complete_task_btn.clicked.connect(self.complete_task_clicked.emit)
        self._complete_task_btn.setEnabled(False)
        task_layout.addWidget(self._complete_task_btn)
        
        # 暂停/跳过按钮
        pause_layout = QHBoxLayout()
        self._pause_btn = QPushButton("暂停")
        self._pause_btn.clicked.connect(self._on_pause)
        self._pause_btn.setEnabled(False)
        
        self._skip_btn = QPushButton("跳过当前")
        self._skip_btn.clicked.connect(self.skip_clicked.emit)
        self._skip_btn.setEnabled(False)
        
        pause_layout.addWidget(self._pause_btn)
        pause_layout.addWidget(self._skip_btn)
        task_layout.addLayout(pause_layout)
        
        layout.addWidget(task_group)
        
        # Episode 设置组
        episode_group = QGroupBox("Episode 设置")
        episode_layout = QVBoxLayout(episode_group)
        
        # 开始任务索引
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("开始任务索引:"))
        self._task_index_spin = QSpinBox()
        self._task_index_spin.setMinimum(0)
        self._task_index_spin.setMaximum(9999)
        self._task_index_spin.setValue(0)
        self._task_index_spin.valueChanged.connect(self.episode_changed.emit)
        start_layout.addWidget(self._task_index_spin)
        episode_layout.addLayout(start_layout)
        
        layout.addWidget(episode_group)
        
        # 添加弹性空间
        layout.addStretch()
    
    def _on_pause(self):
        """暂停/继续切换"""
        if self._pause_btn.text() == "暂停":
            self._pause_btn.setText("继续")
            self.pause_clicked.emit()
        else:
            self._pause_btn.setText("暂停")
            self.resume_clicked.emit()
    
    def set_running(self, running: bool):
        """设置运行状态"""
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._pause_btn.setEnabled(running)
        self._skip_btn.setEnabled(running)
        self._next_task_btn.setEnabled(running)
        self._retry_btn.setEnabled(running)
        self._complete_task_btn.setEnabled(running)
        self._task_index_spin.setEnabled(not running)

    def set_mock_mode(self, enabled: bool):
        """设置 mock 模式 UI"""
        self._complete_task_btn.setVisible(enabled)
    
    def get_start_task_index(self) -> int:
        return self._task_index_spin.value()
    
    def set_current_task_index(self, index: int):
        self._task_index_spin.setValue(index)
