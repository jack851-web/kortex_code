"""
日志面板组件
"""
from datetime import datetime
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QMetaObject, Q_ARG
from PyQt5.QtGui import QTextCursor, QFont


class LogPanel(QWidget):
    """日志面板 - 显示系统日志"""
    
    # 定义信号用于跨线程日志更新
    log_signal = pyqtSignal(str, str)
    
    def __init__(self, max_lines: int = 500):
        super().__init__()
        self._max_lines = max_lines
        self._init_ui()
        
        # 连接信号到槽
        self.log_signal.connect(self._append_log)
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 日志文本框
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("Consolas", 9))
        self._log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        
        layout.addWidget(self._log_text)
        
        # 按钮行
        btn_layout = QHBoxLayout()
        
        self._clear_btn = QPushButton("清除日志")
        self._clear_btn.clicked.connect(self.clear_log)
        
        self._save_btn = QPushButton("保存日志")
        self._save_btn.clicked.connect(self._save_log)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self._clear_btn)
        btn_layout.addWidget(self._save_btn)
        
        layout.addLayout(btn_layout)
    
    def log(self, message: str, level: str = "INFO"):
        """添加日志 - 线程安全"""
        self.log_signal.emit(message, level)
    
    def _append_log(self, message: str, level: str = "INFO"):
        """实际添加日志到文本框 - 只在主线程调用"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # 根据级别设置颜色
        color_map = {
            "INFO": "#d4d4d4",
            "WARNING": "#ffa500",
            "ERROR": "#ff4444",
            "SUCCESS": "#44ff44",
        }
        color = color_map.get(level, "#d4d4d4")
        
        html = f'<span style="color: #888;">[{timestamp}]</span> <span style="color: {color};">[{level}]</span> {message}'
        
        self._log_text.append(html)
        
        # 限制行数
        if self._log_text.document().blockCount() > self._max_lines:
            cursor = self._log_text.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 10)
            cursor.removeSelectedText()
        
        # 滚动到底部
        self._log_text.moveCursor(QTextCursor.End)
    
    def info(self, message: str):
        self.log(message, "INFO")
    
    def warning(self, message: str):
        self.log(message, "WARNING")
    
    def error(self, message: str):
        self.log(message, "ERROR")
    
    def success(self, message: str):
        self.log(message, "SUCCESS")
    
    def clear_log(self):
        """清除日志"""
        self._log_text.clear()
    
    def _save_log(self):
        """保存日志"""
        from PyQt5.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存日志", "", "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self._log_text.toPlainText())
            self.success(f"日志已保存到: {filename}")
