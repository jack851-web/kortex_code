"""
相机显示组件
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QGridLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class CameraWidget(QWidget):
    """单个相机显示组件"""
    
    def __init__(self, camera_name: str, width: int = 320, height: int = 240):
        super().__init__()
        self._camera_name = camera_name
        self._width = width
        self._height = height
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标题
        self._title_label = QLabel(self._camera_name)
        self._title_label.setStyleSheet("font-weight: bold; color: #333;")
        layout.addWidget(self._title_label)
        
        # 图像显示
        self._image_label = QLabel()
        self._image_label.setMinimumSize(self._width, self._height)
        self._image_label.setMaximumSize(self._width, self._height)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet("background-color: #000000;")
        self._image_label.setText("等待图像...")
        layout.addWidget(self._image_label)
    
    def update_image(self, image: np.ndarray):
        """更新图像显示"""
        if image is None or image.size == 0:
            return
        
        try:
            # 确保图像是 numpy 数组
            image = np.array(image)
            
            # 确保图像是 RGB 格式
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            
            # 转换 BGR 到 RGB（如果是 OpenCV 格式）
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            h, w, ch = image.shape
            
            # 转换为 QImage
            bytes_per_line = ch * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 缩放并显示
            pixmap = QPixmap.fromImage(q_image).scaled(
                self._width, self._height, 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._image_label.setPixmap(pixmap)
            
        except Exception as e:
            print(f"[CameraWidget] Error updating image: {e}")


class CameraPanel(QWidget):
    """相机面板 - 显示多个相机"""
    
    def __init__(self):
        super().__init__()
        self._camera_widgets = {}
        self._columns = 1
        self._init_ui()
    
    def _init_ui(self):
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(8)
        self._layout.setVerticalSpacing(10)

    def set_columns(self, columns: int):
        """设置相机网格列数"""
        cols = max(1, int(columns))
        if cols == self._columns:
            return
        self._columns = cols
        self._reflow()

    def _reflow(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is not None and item.widget() is not None:
                item.widget().setParent(self)

        for i, widget in enumerate(self._camera_widgets.values()):
            row = i // self._columns
            col = i % self._columns
            self._layout.addWidget(widget, row, col)
    
    def add_camera(self, camera_name: str, width: int = 320, height: int = 240):
        """添加相机显示"""
        if camera_name in self._camera_widgets:
            return
        
        widget = CameraWidget(camera_name, width, height)
        self._camera_widgets[camera_name] = widget
        self._reflow()
    
    def update_camera(self, camera_name: str, image: np.ndarray):
        """更新指定相机的图像"""
        if camera_name in self._camera_widgets:
            self._camera_widgets[camera_name].update_image(image)
    
    def update_all_cameras(self, images: dict):
        """更新所有相机图像"""
        for name, image in images.items():
            self.update_camera(name, image)
    
    def clear_all(self):
        """清除所有相机"""
        while self._layout.count():
            self._layout.takeAt(0)
        for widget in self._camera_widgets.values():
            widget.deleteLater()
        self._camera_widgets.clear()
