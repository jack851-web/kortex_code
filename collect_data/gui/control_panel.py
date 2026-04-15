"""
控制面板组件
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QGroupBox
)
from PyQt5.QtCore import pyqtSignal


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
    
    def __init__(self):
        super().__init__()
        self._is_running = False
        self._all_task_btns = []  # 所有任务操作按钮的列表
        self._btn_original_states = {}  # set_busy 时保存的按钮启用状态
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 任务控制组
        task_group = QGroupBox("任务控制")
        task_layout = QVBoxLayout(task_group)
        
        # 开始/停止数据收集按钮
        btn_layout = QHBoxLayout()
        self._start_btn = QPushButton("开始收集")
        self._start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self._start_btn.clicked.connect(self.start_clicked.emit)

        self._stop_btn = QPushButton("停止收集")
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
        
        # 重做任务按钮（任务执行中可直接点击，丢弃当前数据重新执行）
        self._retry_btn = QPushButton("重做任务")
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

        # 收集所有任务操作按钮
        self._all_task_btns = [
            self._start_btn, self._stop_btn,
            self._next_task_btn, self._retry_btn,
            self._complete_task_btn,
            self._pause_btn, self._skip_btn,
        ]
        
        layout.addWidget(task_group)
        
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
        self._is_running = running
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._pause_btn.setEnabled(running)
        self._skip_btn.setEnabled(running)
        self._next_task_btn.setEnabled(running)
        self._retry_btn.setEnabled(running)
        self._complete_task_btn.setEnabled(running)
    
    def set_mock_mode(self, enabled: bool):
        """设置 mock 模式 UI"""
        self._complete_task_btn.setVisible(enabled)

    def set_busy(self, label: str = "处理中..."):
        """禁用所有按钮并显示处理状态（点击后立即调用，防止重复操作）"""
        # 保存当前每个按钮的启用状态
        self._btn_original_states = {id(btn): btn.isEnabled() for btn in self._all_task_btns}

        # 全部禁用
        for btn in self._all_task_btns:
            btn.setEnabled(False)

        # 在对应按钮上显示状态文字
        if self._complete_task_btn.isEnabled() == False and self._btn_original_states.get(id(self._complete_task_btn)):
            self._complete_task_btn.setText(label)
        if self._stop_btn.isEnabled() == False and self._btn_original_states.get(id(self._stop_btn)):
            self._stop_btn.setText(label)

    def set_idle(self):
        """恢复按钮到 set_busy 之前的状态"""
        for btn in self._all_task_btns:
            btn.setEnabled(self._btn_original_states.get(id(btn), btn.isEnabled()))

        # 恢复按钮文字
        self._complete_task_btn.setText("抓取任务完毕")
        self._stop_btn.setText("停止收集")
        self._start_btn.setText("开始收集")
        self._next_task_btn.setText("执行下一个任务")
        self._retry_btn.setText("重做任务")
        self._skip_btn.setText("跳过当前")
        self._btn_original_states = {}
