"""
主窗口
"""
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QTabWidget, QApplication, QMessageBox,
    QGroupBox, QGridLayout, QLabel, QPushButton
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QKeyEvent

from .camera_widget import CameraPanel
from .control_panel import ControlPanel
from .status_panel import StatusPanel
from .log_panel import LogPanel


class DataCollectionWorker(QThread):
    """数据收集工作线程"""
    
    log_signal = pyqtSignal(str, str)
    status_signal = pyqtSignal(str, str)
    joints_signal = pyqtSignal(object)
    cartesian_signal = pyqtSignal(object)
    gripper_signal = pyqtSignal(float)
    task_signal = pyqtSignal(str, int, int)
    camera_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()
    
    def __init__(self, data_collector):
        super().__init__()
        self._collector = data_collector
        self._running = False
        self._paused = False
    
    def run(self):
        self._running = True
        self.log_signal.emit("数据收集线程启动", "INFO")
        
        while self._running:
            if self._paused:
                self.msleep(100)
                continue
            
            try:
                # 这里执行数据收集循环
                # 具体实现由 DataCollectorSystem 提供
                self.msleep(10)
            except Exception as e:
                self.log_signal.emit(f"收集错误: {e}", "ERROR")
        
        self.log_signal.emit("数据收集线程停止", "INFO")
        self.finished_signal.emit()
    
    def stop(self):
        self._running = False
    
    def pause(self):
        self._paused = True
    
    def resume(self):
        self._paused = False


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self, config_path: str = None, mock_mode: bool = False):
        super().__init__()
        self._config_path = config_path
        self._data_system = None
        self._worker = None
        self._update_timer = None
        self._mock_mode = mock_mode
        
        self._keys_pressed = set()
        self._teleop_step_size = 0.008
        self._rot_step_size = 5.0
        
        self._init_ui()
        self._init_timer()
    
    def _init_ui(self):
        self.setWindowTitle("Kortex 数据收集系统")
        
        if self._mock_mode:
            self._init_mock_ui()
        else:
            self._init_real_ui()
        
        self._connect_signals()
    
    def _init_mock_ui(self):
        """Mock 模式专用 UI"""
        self.setMinimumSize(600, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        info_group = QGroupBox("操作说明")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel("W/S:"), 0, 0)
        info_layout.addWidget(QLabel("后退/前进"), 0, 1)
        info_layout.addWidget(QLabel("A/D:"), 1, 0)
        info_layout.addWidget(QLabel("左/右"), 1, 1)
        info_layout.addWidget(QLabel("R/F:"), 2, 0)
        info_layout.addWidget(QLabel("上升/下降"), 2, 1)
        info_layout.addWidget(QLabel("←/→:"), 3, 0)
        info_layout.addWidget(QLabel("旋转"), 3, 1)
        info_layout.addWidget(QLabel("[ / ]:"), 4, 0)
        info_layout.addWidget(QLabel("夹爪闭合/张开"), 4, 1)
        info_layout.addWidget(QLabel("Z:"), 5, 0)
        info_layout.addWidget(QLabel("回初始位"), 5, 1)
        
        main_layout.addWidget(info_group)
        
        task_group = QGroupBox("任务信息")
        task_layout = QVBoxLayout(task_group)
        
        self._task_label = QLabel("当前任务: -")
        self._task_label.setStyleSheet("font-weight: bold;")
        task_layout.addWidget(self._task_label)
        
        pos_layout = QGridLayout()
        pos_layout.addWidget(QLabel("物体:"), 0, 0)
        self._object_label = QLabel("x=0, y=0, z=0")
        pos_layout.addWidget(self._object_label, 0, 1)
        
        pos_layout.addWidget(QLabel("目标:"), 1, 0)
        self._target_label = QLabel("x=0, y=0, z=0")
        pos_layout.addWidget(self._target_label, 1, 1)
        
        pos_layout.addWidget(QLabel("距离:"), 2, 0)
        self._distance_label = QLabel("0.000m")
        pos_layout.addWidget(self._distance_label, 2, 1)
        task_layout.addLayout(pos_layout)
        
        main_layout.addWidget(task_group)
        
        state_group = QGroupBox("实时状态")
        state_layout = QGridLayout(state_group)
        
        self._pose_label = QLabel("TCP: x=0 y=0 z=0")
        state_layout.addWidget(self._pose_label, 0, 0, 1, 2)
        
        self._gripper_label = QLabel("夹爪: 0.000")
        state_layout.addWidget(self._gripper_label, 1, 0, 1, 2)
        
        main_layout.addWidget(state_group)
        
        self._control_panel = ControlPanel()
        main_layout.addWidget(self._control_panel)
        
        self._status_panel = StatusPanel()
        main_layout.addWidget(self._status_panel)
        
        self._log_panel = LogPanel()
        main_layout.addWidget(self._log_panel, stretch=1)
        
        self._status_label = QLabel("等待任务开始...")
        self._status_label.setStyleSheet("color: #666;")
        main_layout.addWidget(self._status_label)
        
        self.setFocusPolicy(Qt.StrongFocus)
    
    def _init_real_ui(self):
        """实机模式 UI"""
        self.setMinimumSize(1100, 750)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        self._left_panel = QWidget()
        self._left_panel.setMinimumWidth(680)
        self._left_layout = QVBoxLayout(self._left_panel)
        self._left_layout.setContentsMargins(0, 0, 0, 0)
        self._left_layout.setSpacing(6)

        self._real_camera_panel = CameraPanel()
        self._left_layout.addWidget(self._real_camera_panel)

        self._simu_camera_panel = CameraPanel()
        self._left_layout.addWidget(self._simu_camera_panel)

        main_layout.addWidget(self._left_panel, stretch=3)
        
        right_panel = QWidget()
        right_panel.setMinimumWidth(350)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        
        self._control_panel = ControlPanel()
        right_layout.addWidget(self._control_panel)
        
        self._status_panel = StatusPanel()
        right_layout.addWidget(self._status_panel)
        
        self._log_panel = LogPanel()
        right_layout.addWidget(self._log_panel, stretch=1)
        
        main_layout.addWidget(right_panel, stretch=4)
    
    def _init_timer(self):
        """初始化更新定时器"""
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_display)
        if self._mock_mode:
            self._update_timer.timeout.connect(self._process_keyboard_teleop)
        self._update_timer.start(50)  # 20 FPS
    
    def _connect_signals(self):
        """连接控制面板信号"""
        self._control_panel.start_clicked.connect(self._on_start)
        self._control_panel.stop_clicked.connect(self._on_stop)
        self._control_panel.pause_clicked.connect(self._on_pause)
        self._control_panel.resume_clicked.connect(self._on_resume)
        self._control_panel.skip_clicked.connect(self._on_skip)
        self._control_panel.next_task_clicked.connect(self._on_next_task)
        self._control_panel.retry_clicked.connect(self._on_retry)
        self._control_panel.complete_task_clicked.connect(self._on_complete_task)
    
    def set_data_system(self, data_system):
        """设置数据收集系统"""
        self._data_system = data_system

        use_real = bool(getattr(self._data_system, '_use_real', False))
        self._status_panel.set_mode(use_real)
        self._control_panel.set_mock_mode(not use_real)
        if use_real:
            self.setWindowTitle("Kortex 数据收集系统（实机模式）")
            self.setMinimumSize(1400, 900)
            self.resize(1500, 960)
        else:
            self.setWindowTitle("Kortex 数据收集系统（模拟模式）")
            self.setMinimumSize(1100, 750)
    
    def setup_cameras(self, real_cameras: list, simu_cameras: list, real_camera_config: dict = None, simu_camera_config: dict = None):
        """设置相机显示"""
        real_camera_config = real_camera_config or {}
        simu_camera_config = simu_camera_config or {}

        # 清除现有相机
        self._real_camera_panel.clear_all()
        self._simu_camera_panel.clear_all()

        # mock 模式下没有实机相机：隐藏实机面板，避免左侧留空
        has_real_cameras = len(real_cameras) > 0
        self._real_camera_panel.setVisible(has_real_cameras)

        # 添加真实相机
        if has_real_cameras:
            self._real_camera_panel.set_columns(1)
            for cam_name in real_cameras:
                cam_cfg = real_camera_config.get(cam_name, {})
                width = int(cam_cfg.get('width', 320))
                height = int(cam_cfg.get('height', 240))
                self._real_camera_panel.add_camera(f"Real: {cam_name}", width, height)

        # 添加仿真相机（垂直排列）
        self._simu_camera_panel.set_columns(1)
        max_cam_width = 0
        for cam_name in simu_cameras:
            cam_cfg = simu_camera_config.get(cam_name, {})
            width = int(cam_cfg.get('width', 320))
            height = int(cam_cfg.get('height', 240))
            max_cam_width = max(max_cam_width, width)
            self._simu_camera_panel.add_camera(f"Simu: {cam_name}", width, height)

        # 根据相机宽度调整左侧面板
        if max_cam_width > 0:
            left_target_width = max_cam_width + 24
            self._left_panel.setMinimumWidth(max(500, left_target_width))
            self.setMinimumWidth(left_target_width + 420)
    
    def _update_display(self):
        """更新显示"""
        if self._data_system is None:
            return
        
        try:
            # 更新相机图像
            if self._real_camera_panel.isVisible() and hasattr(self._data_system, '_real') and self._data_system._real:
                real_images = self._data_system._real.get_camera_images()
                # 将原始相机名称映射为显示名称
                display_images = {f"Real: {name}": img for name, img in real_images.items()}
                self._real_camera_panel.update_all_cameras(display_images)
            
            if hasattr(self._data_system, '_simu') and self._data_system._simu:
                simu_images = self._data_system._simu.get_camera_images()
                # 将原始相机名称映射为显示名称
                display_images = {f"Simu: {name}": img for name, img in simu_images.items()}
                self._simu_camera_panel.update_all_cameras(display_images)
            
            # 更新实机关节状态
            if hasattr(self._data_system, '_real') and self._data_system._real:
                joints = self._data_system._real.get_joint_state()
                self._status_panel.update_joints(joints)

                # 更新实机笛卡尔坐标
                try:
                    pose = self._data_system._real.get_cartesian_pose()
                    self._status_panel.update_cartesian(pose)
                except:
                    pass

                # 更新实机夹爪状态
                try:
                    gripper = self._data_system._real.get_gripper_state()
                    self._status_panel.update_gripper(gripper)
                except:
                    pass

            # 更新仿真关节状态
            if hasattr(self._data_system, '_simu') and self._data_system._simu:
                simu_joints = self._data_system._simu.get_joint_state()
                self._status_panel.update_simu_joints(simu_joints)

                # 更新仿真笛卡尔坐标
                try:
                    simu_pose = self._data_system._simu.get_cartesian_pose()
                    self._status_panel.update_simu_cartesian(simu_pose)
                except:
                    pass

                # 更新仿真夹爪状态
                try:
                    simu_gripper = self._data_system._simu.get_gripper_state()
                    self._status_panel.update_simu_gripper(simu_gripper)
                except:
                    pass

        except Exception as e:
            pass
        
        if self._mock_mode:
            self._update_mock_state()
    
    def _update_mock_state(self):
        """更新 Mock 模式状态"""
        if self._data_system is None:
            return
        
        state = self._data_system.get_tuning_state()
        if not state:
            return
        
        tcp_pos = state.get('tcp_pos', np.zeros(3))
        tcp_euler = state.get('tcp_euler', np.zeros(3))
        gripper = state.get('gripper', 0.0)
        object_pos = state.get('object_pos', np.zeros(3))
        target_pos = state.get('target_pos', np.zeros(3))
        task_name = state.get('task_name', '-')
        
        self._task_label.setText(f"当前任务: {task_name}")
        self._object_label.setText(f"x={object_pos[0]:.3f}, y={object_pos[1]:.3f}, z={object_pos[2]:.3f}")
        self._target_label.setText(f"x={target_pos[0]:.3f}, y={target_pos[1]:.3f}, z={target_pos[2]:.3f}")
        
        distance = np.linalg.norm(object_pos - target_pos)
        self._distance_label.setText(f"{distance:.3f}m")
        
        self._pose_label.setText(f"TCP: x={tcp_pos[0]:.3f} y={tcp_pos[1]:.3f} z={tcp_pos[2]:.3f}")
        self._gripper_label.setText(f"夹爪: {gripper:.3f}")
    
    def keyPressEvent(self, event: QKeyEvent):
        if not self._mock_mode:
            super().keyPressEvent(event)
            return
        
        self._keys_pressed.add(event.key())
        
        if event.key() == Qt.Key_Z:
            self._move_home()
            event.accept()
        elif event.key() == Qt.Key_BracketLeft:
            self._close_gripper_more()
        elif event.key() == Qt.Key_BracketRight:
            self._open_gripper_more()
        
        super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event: QKeyEvent):
        if not self._mock_mode:
            super().keyReleaseEvent(event)
            return
        
        self._keys_pressed.discard(event.key())
        super().keyReleaseEvent(event)
    
    def _process_keyboard_teleop(self):
        if not self._keys_pressed or not self._mock_mode:
            return
        
        state = self._data_system.get_tuning_state() if self._data_system else None
        if not state:
            return
        
        current_tcp_pos = state.get('tcp_pos', np.zeros(3))
        current_tcp_euler = state.get('tcp_euler', np.zeros(3))
        
        dpos = np.zeros(3)
        drot = np.zeros(3)
        
        if Qt.Key_W in self._keys_pressed:
            dpos[0] -= self._teleop_step_size
        if Qt.Key_S in self._keys_pressed:
            dpos[0] += self._teleop_step_size
        if Qt.Key_A in self._keys_pressed:
            dpos[1] -= self._teleop_step_size
        if Qt.Key_D in self._keys_pressed:
            dpos[1] += self._teleop_step_size
        if Qt.Key_R in self._keys_pressed:
            dpos[2] += self._teleop_step_size
        if Qt.Key_F in self._keys_pressed:
            dpos[2] -= self._teleop_step_size
        
        rot_rad = np.deg2rad(self._rot_step_size)
        if Qt.Key_Left in self._keys_pressed:
            drot[2] -= rot_rad
        if Qt.Key_Right in self._keys_pressed:
            drot[2] += rot_rad
        
        if np.any(dpos != 0) or np.any(drot != 0):
            new_pos = current_tcp_pos + dpos
            new_euler = current_tcp_euler + np.rad2deg(drot)
            
            target_pose = np.array([
                new_pos[0], new_pos[1], new_pos[2],
                new_euler[0], new_euler[1], new_euler[2]
            ], dtype=float)
            
            self._data_system.tuning_move_to_pose(target_pose)
    
    def _move_home(self):
        if self._data_system and self._data_system.move_to_home_pose():
            self._status_label.setText("已回到初始位置")
    
    def _close_gripper_more(self):
        if not self._data_system:
            return
        state = self._data_system.get_tuning_state()
        if not state:
            return
        new_gripper = min(1.0, state.get('gripper', 0.0) + 0.05)
        self._data_system.set_tuning_gripper(new_gripper)
        self._status_label.setText(f"夹爪: {new_gripper:.2f}")
    
    def _open_gripper_more(self):
        if not self._data_system:
            return
        state = self._data_system.get_tuning_state()
        if not state:
            return
        new_gripper = max(0.0, state.get('gripper', 0.0) - 0.05)
        self._data_system.set_tuning_gripper(new_gripper)
        self._status_label.setText(f"夹爪: {new_gripper:.2f}")
    
    def _on_start(self):
        """开始任务"""
        if self._data_system is None:
            self._log_panel.error("数据系统未初始化")
            return
        
        self._log_panel.info("开始数据收集...")
        self._control_panel.set_running(True)
        
        # 启动数据收集
        try:
            self._data_system.start_collection(
                start_task_index=self._control_panel.get_start_task_index()
            )
        except Exception as e:
            self._log_panel.error(f"启动失败: {e}")
            import traceback
            traceback.print_exc()
            self._control_panel.set_running(False)
    
    def _on_stop(self):
        """停止任务"""
        self._log_panel.warning("停止数据收集...")
        self._control_panel.set_running(False)
        self._status_panel.update_status("已停止", "#ff4444")
        
        if self._data_system:
            self._data_system.stop()
    
    def _on_pause(self):
        """暂停任务"""
        self._log_panel.info("暂停数据收集")
        self._status_panel.update_status("已暂停", "#ffa500")
        
        if self._data_system:
            self._data_system.pause()
    
    def _on_resume(self):
        """继续任务"""
        self._log_panel.info("继续数据收集")
        self._status_panel.update_status("运行中", "#44ff44")
        
        if self._data_system:
            self._data_system.resume()
    
    def _on_skip(self):
        """跳过当前任务"""
        self._log_panel.info("跳过当前任务")
        
        if self._data_system:
            self._data_system.skip_current_task()
    
    def _on_next_task(self):
        """执行下一个任务"""
        self._log_panel.info("执行下一个任务")
        
        if self._data_system:
            self._data_system.execute_next_task()
    
    def _on_retry(self):
        """重做当前任务"""
        self._log_panel.warning("重做当前任务")
        
        if self._data_system:
            self._data_system.retry_current_task()

    def _on_complete_task(self):
        """手动确认当前任务完成"""
        self._log_panel.info("手动确认任务完毕")
        if self._data_system:
            self._data_system.finish_current_task()
    
    def log_message(self, message: str, level: str = "INFO"):
        """添加日志消息"""
        self._log_panel.log(message, level)
    
    def log(self, message: str, level: str = "INFO"):
        """添加日志消息 (别名)"""
        self._log_panel.log(message, level)
    
    def update_status(self, status: str, color: str = "#666"):
        """更新状态显示"""
        self._status_panel.update_status(status, color)
    
    def update_task_info(self, task_name: str, current: int, total: int):
        """更新任务信息"""
        self._status_panel.update_task_info(task_name, current, total)
    
    def closeEvent(self, event):
        """关闭事件"""
        if self._data_system:
            self._data_system.stop()
        event.accept()


def run_gui(config_path: str = None, data_system=None):
    """运行 GUI"""
    app = QApplication(sys.argv)
    
    window = MainWindow(config_path=config_path)
    
    if data_system:
        window.set_data_system(data_system)
    
    window.show()
    
    return app.exec_()


if __name__ == "__main__":
    sys.exit(run_gui())
