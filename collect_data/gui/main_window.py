"""
主窗口 - 通过 MessageBroker 订阅数据，解耦 Qt 与 MuJoCo/RealInterface

架构:
  MuJoCo/Real ──publish──> MessageBroker ──subscribe──> MainWindow
  (仿真/硬件线程)           (中间件)              (GUI 线程)

  订阅回调在发布者线程执行，通过 pyqtSignal 安全转到 GUI 线程。
"""
import sys
import time
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

# 消息总线
from scripts.core.message_bus import MessageBroker
from scripts.core.topic_defs import (
    SIMU_IMAGES, SIMU_JOINTS, SIMU_GRIPPER, SIMU_TCP_POSE,
    SIMU_OBJECT_POS, SIMU_STATUS, SIMU_CARTESIAN,
    REAL_IMAGES, REAL_JOINTS, REAL_CARTESIAN, REAL_GRIPPER, REAL_STATUS,
    COLLECT_LOG,
)


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
    """主窗口 - 通过 MessageBroker 订阅数据"""

    # Qt 信号：从发布者线程安全转到 GUI 线程
    _simu_images_signal = pyqtSignal(dict)       # {cam_name: np.ndarray (RGB)}
    _simu_joints_signal = pyqtSignal(object)      # np.ndarray (rad)
    _simu_cartesian_signal = pyqtSignal(object)   # np.ndarray [x,y,z,wx,wy,wz]
    _simu_gripper_signal = pyqtSignal(float)      # float
    _simu_tcp_signal = pyqtSignal(object)         # Tuple[pos, rotmat]
    _simu_object_pos_signal = pyqtSignal(object)  # np.ndarray
    _simu_status_signal = pyqtSignal(str)         # str

    _real_images_signal = pyqtSignal(dict)        # {cam_name: np.ndarray (BGR)}
    _real_joints_signal = pyqtSignal(object)      # np.ndarray (deg)
    _real_cartesian_signal = pyqtSignal(object)   # np.ndarray [x,y,z,wx,wy,wz]
    _real_gripper_signal = pyqtSignal(float)      # float
    _real_status_signal = pyqtSignal(str)         # str

    # 后台任务完成信号
    _bg_task_done_signal = pyqtSignal()

    # 任务信息更新信号（从后台线程安全转到 GUI 线程）
    _task_info_signal = pyqtSignal(str, int, int)  # task_name, current, total

    def __init__(self, config_path: str = None, mock_mode: bool = False):
        super().__init__()
        self._busy = False  # 后台任务执行中标志
        self._bg_task_done_signal.connect(self._on_bg_task_done)
        self._task_info_signal.connect(self._on_update_task_info)
        self._config_path = config_path
        self._data_system = None
        self._worker = None
        self._update_timer = None
        self._mock_mode = mock_mode
        self._broker = MessageBroker.instance()

        # 缓存最新数据（由 broker 回调更新，GUI 线程读取）
        self._latest_simu_images: Dict[str, np.ndarray] = {}
        self._latest_real_images: Dict[str, np.ndarray] = {}
        self._latest_simu_joints: Optional[np.ndarray] = None
        self._latest_simu_gripper: float = 0.0
        self._latest_simu_cartesian: Optional[np.ndarray] = None
        self._latest_simu_tcp: Optional[tuple] = None
        self._latest_simu_object_pos: Optional[np.ndarray] = None
        self._latest_simu_status: str = "idle"
        self._latest_real_joints: Optional[np.ndarray] = None
        self._latest_real_gripper: float = 0.0
        self._latest_real_cartesian: Optional[np.ndarray] = None
        self._latest_real_status: str = "disconnected"

        self._keys_pressed = set()
        self._teleop_step_size = 0.04
        self._rot_step_size = 12.0

        # 任务计时相关
        self._task_start_time: float = 0.0  # 任务开始时间
        self._task_elapsed_paused: float = 0.0  # 暂停时累计的时间
        self._task_timer_running: bool = False  # 计时器是否运行中

        self._init_ui()
        self._init_timer()
        self._init_broker_subscriptions()

    def _init_ui(self):
        self.setWindowTitle("Kortex 数据收集系统")

        if self._mock_mode:
            self._init_mock_ui()
        else:
            self._init_real_ui()

        self._connect_signals()

    def _init_mock_ui(self):
        """Mock 模式专用 UI"""
        self.setMinimumSize(700, 900)
        self.resize(850, 1200)

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

        # Mock 模式不需要相机面板（MuJoCo 已显示相机画面）
        # 创建隐藏的空面板占位，避免 setup_cameras 报错
        self._simu_camera_panel = CameraPanel()
        self._simu_camera_panel.setVisible(False)
        self._real_camera_panel = CameraPanel()
        self._real_camera_panel.setVisible(False)

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
        """实机模式 UI：左边实机相机，右边控制面板"""
        self.setMinimumSize(1100, 750)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # 左边：实机相机画面
        self._left_panel = QWidget()
        self._left_panel.setMinimumWidth(680)
        self._left_layout = QVBoxLayout(self._left_panel)
        self._left_layout.setContentsMargins(0, 0, 0, 0)
        self._left_layout.setSpacing(6)

        self._real_camera_panel = CameraPanel()
        self._left_layout.addWidget(self._real_camera_panel)

        # 仿真相机面板隐藏（实机模式不需要在 Qt 显示）
        self._simu_camera_panel = CameraPanel()
        self._simu_camera_panel.setVisible(False)

        main_layout.addWidget(self._left_panel, stretch=3)

        # 右边：控制面板
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
        """初始化更新定时器

        改造后：定时器仅用于刷新 GUI 显示（从缓存读取），
        不再直接调用 SimuInterface/RealInterface。
        """
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_display_from_cache)
        self._update_timer.timeout.connect(self._update_task_timer)
        if self._mock_mode:
            self._update_timer.timeout.connect(self._process_keyboard_teleop)
        self._update_timer.start(50)  # 20 FPS 刷新 GUI

    def _init_broker_subscriptions(self):
        """订阅 MessageBroker 话题

        订阅回调在发布者线程执行，通过 pyqtSignal 转到 GUI 线程。
        """
        # 仿真话题
        self._broker.subscribe(SIMU_IMAGES, self._on_simu_images)
        self._broker.subscribe(SIMU_JOINTS, self._on_simu_joints)
        self._broker.subscribe(SIMU_GRIPPER, self._on_simu_gripper)
        self._broker.subscribe(SIMU_CARTESIAN, self._on_simu_cartesian)
        self._broker.subscribe(SIMU_TCP_POSE, self._on_simu_tcp)
        self._broker.subscribe(SIMU_OBJECT_POS, self._on_simu_object_pos)
        self._broker.subscribe(SIMU_STATUS, self._on_simu_status)

        # 真实机器人话题
        self._broker.subscribe(REAL_IMAGES, self._on_real_images)
        self._broker.subscribe(REAL_JOINTS, self._on_real_joints)
        self._broker.subscribe(REAL_GRIPPER, self._on_real_gripper)
        self._broker.subscribe(REAL_CARTESIAN, self._on_real_cartesian)
        self._broker.subscribe(REAL_STATUS, self._on_real_status)

        # 连接信号到 GUI 线程的更新方法
        self._simu_images_signal.connect(self._handle_simu_images)
        self._simu_joints_signal.connect(self._handle_simu_joints)
        self._simu_gripper_signal.connect(self._handle_simu_gripper)
        self._simu_cartesian_signal.connect(self._handle_simu_cartesian)
        self._simu_tcp_signal.connect(self._handle_simu_tcp)
        self._simu_object_pos_signal.connect(self._handle_simu_object_pos)
        self._simu_status_signal.connect(self._handle_simu_status)

        self._real_images_signal.connect(self._handle_real_images)
        self._real_joints_signal.connect(self._handle_real_joints)
        self._real_gripper_signal.connect(self._handle_real_gripper)
        self._real_cartesian_signal.connect(self._handle_real_cartesian)
        self._real_status_signal.connect(self._handle_real_status)

    # ================================================================
    # 订阅回调（在发布者线程执行 → emit 信号转到 GUI 线程）
    # ================================================================

    def _on_simu_images(self, images):
        if images is not None:
            # 拷贝图像，避免发布者线程修改 buffer
            copied = {k: np.array(v, copy=True) for k, v in images.items()}
            self._simu_images_signal.emit(copied)

    def _on_simu_joints(self, joints):
        if joints is not None:
            self._simu_joints_signal.emit(np.copy(joints))

    def _on_simu_gripper(self, gripper):
        if gripper is not None:
            self._simu_gripper_signal.emit(float(gripper))

    def _on_simu_cartesian(self, cartesian):
        if cartesian is not None:
            self._simu_cartesian_signal.emit(np.copy(cartesian))

    def _on_simu_tcp(self, tcp):
        if tcp is not None:
            pos, rot = tcp
            self._simu_tcp_signal.emit((np.copy(pos), np.copy(rot)))

    def _on_simu_object_pos(self, obj_pos):
        if obj_pos is not None:
            self._simu_object_pos_signal.emit(np.copy(obj_pos))

    def _on_simu_status(self, status):
        self._simu_status_signal.emit(str(status))

    def _on_real_images(self, images):
        if images is not None:
            copied = {k: np.array(v, copy=True) for k, v in images.items()}
            self._real_images_signal.emit(copied)

    def _on_real_joints(self, joints):
        if joints is not None:
            self._real_joints_signal.emit(np.copy(joints))

    def _on_real_gripper(self, gripper):
        if gripper is not None:
            self._real_gripper_signal.emit(float(gripper))

    def _on_real_cartesian(self, cartesian):
        if cartesian is not None:
            self._real_cartesian_signal.emit(np.copy(cartesian))

    def _on_real_status(self, status):
        self._real_status_signal.emit(str(status))

    # ================================================================
    # 信号处理（在 GUI 线程执行，安全更新缓存和 UI）
    # ================================================================

    def _handle_simu_images(self, images):
        self._latest_simu_images = images
        # 直接更新相机面板（仿真相机返回 RGB，is_bgr=False）
        display_images = {f"Simu: {name}": img for name, img in images.items()}
        if hasattr(self, '_simu_camera_panel'):
            self._simu_camera_panel.update_all_cameras(display_images, is_bgr=False)

    def _handle_simu_joints(self, joints):
        self._latest_simu_joints = joints

    def _handle_simu_gripper(self, gripper):
        self._latest_simu_gripper = gripper

    def _handle_simu_cartesian(self, cartesian):
        self._latest_simu_cartesian = cartesian

    def _handle_simu_tcp(self, tcp):
        self._latest_simu_tcp = tcp

    def _handle_simu_object_pos(self, obj_pos):
        self._latest_simu_object_pos = obj_pos

    def _handle_simu_status(self, status):
        self._latest_simu_status = status

    def _handle_real_images(self, images):
        self._latest_real_images = images
        # 真实相机返回 BGR，is_bgr=True
        display_images = {f"Real: {name}": img for name, img in images.items()}
        if hasattr(self, '_real_camera_panel'):
            self._real_camera_panel.update_all_cameras(display_images, is_bgr=True)

    def _handle_real_joints(self, joints):
        self._latest_real_joints = joints

    def _handle_real_gripper(self, gripper):
        self._latest_real_gripper = gripper

    def _handle_real_cartesian(self, cartesian):
        self._latest_real_cartesian = cartesian

    def _handle_real_status(self, status):
        self._latest_real_status = status

    # ================================================================
    # GUI 定时刷新（从缓存读取，不再直接调用 SimuInterface）
    # ================================================================

    def _update_display_from_cache(self):
        """从缓存更新 GUI 显示（50ms 定时器触发）

        改造后：不再直接调用 _data_system._real/simu.get_xxx()，
        只从 broker 回调更新的缓存中读取数据。
        """
        try:
            # 更新关节/状态面板
            if self._mock_mode:
                self._update_mock_display()
            else:
                self._update_real_display()
        except Exception:
            pass

        if self._mock_mode:
            self._update_mock_state()

    def _update_mock_display(self):
        """Mock 模式：用仿真数据更新显示"""
        if self._latest_simu_joints is not None:
            joints_rad = self._latest_simu_joints
            joints_deg = np.rad2deg(joints_rad)
            self._status_panel.update_joints(joints_deg)
            self._status_panel.update_simu_joints(joints_rad)

        if self._latest_simu_cartesian is not None:
            pose = self._latest_simu_cartesian
            self._status_panel.update_cartesian(pose)
            self._status_panel.update_simu_cartesian(pose)

        self._status_panel.update_gripper(self._latest_simu_gripper)
        self._status_panel.update_simu_gripper(self._latest_simu_gripper)

    def _update_real_display(self):
        """实机模式：分别显示实机和仿真状态"""
        # 真实机器人状态
        if self._latest_real_joints is not None:
            self._status_panel.update_joints(self._latest_real_joints)

        if self._latest_real_cartesian is not None:
            self._status_panel.update_cartesian(self._latest_real_cartesian)

        self._status_panel.update_gripper(self._latest_real_gripper)

        # 仿真状态
        if self._latest_simu_joints is not None:
            self._status_panel.update_simu_joints(self._latest_simu_joints)

        if self._latest_simu_cartesian is not None:
            self._status_panel.update_simu_cartesian(self._latest_simu_cartesian)

        self._status_panel.update_simu_gripper(self._latest_simu_gripper)

    def _update_mock_state(self):
        """更新 Mock 模式状态标签"""
        if self._data_system is None:
            return

        state = self._data_system.get_tuning_state()
        if not state:
            return

        tcp_pos = state.get('tcp_pos', np.zeros(3))
        tcp_euler = state.get('tcp_euler', np.zeros(3))
        gripper = state.get('gripper', 0.0)

        self._pose_label.setText(f"TCP: x={tcp_pos[0]:.3f} y={tcp_pos[1]:.3f} z={tcp_pos[2]:.3f}")
        self._gripper_label.setText(f"夹爪: {gripper:.3f}")

    # ================================================================
    # 任务计时功能
    # ================================================================

    def _start_task_timer(self):
        """开始任务计时"""
        self._task_start_time = time.time()
        self._task_elapsed_paused = 0.0
        self._task_timer_running = True
        self._status_panel.reset_timer()

    def _pause_task_timer(self):
        """暂停任务计时"""
        if self._task_timer_running:
            # 记录当前累计时间
            self._task_elapsed_paused += time.time() - self._task_start_time
            self._task_timer_running = False

    def _resume_task_timer(self):
        """继续任务计时"""
        if not self._task_timer_running:
            # 重置开始时间，继续计时
            self._task_start_time = time.time()
            self._task_timer_running = True

    def _stop_task_timer(self):
        """停止任务计时"""
        self._task_timer_running = False

    def _reset_task_timer(self):
        """重置任务计时"""
        self._task_start_time = 0.0
        self._task_elapsed_paused = 0.0
        self._task_timer_running = False
        self._status_panel.reset_timer()

    def _update_task_timer(self):
        """更新任务计时显示（由定时器调用）"""
        if self._task_timer_running:
            elapsed = self._task_elapsed_paused + (time.time() - self._task_start_time)
            self._status_panel.update_timer(elapsed)

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

        # Mock 模式：相机面板已隐藏，不需要设置
        if self._mock_mode:
            return

        # 实机模式：只设置真实相机面板
        self._real_camera_panel.clear_all()
        has_real_cameras = len(real_cameras) > 0
        self._real_camera_panel.setVisible(has_real_cameras)

        if has_real_cameras:
            self._real_camera_panel.set_columns(1)
            max_cam_width = 0
            for cam_name in real_cameras:
                cam_cfg = real_camera_config.get(cam_name, {})
                width = int(cam_cfg.get('width', 640))
                height = int(cam_cfg.get('height', 480))
                max_cam_width = max(max_cam_width, width)
                self._real_camera_panel.add_camera(f"Real: {cam_name}", width, height)

            # 调整左侧面板宽度
            if hasattr(self, '_left_panel'):
                left_target_width = max_cam_width + 24
                self._left_panel.setMinimumWidth(max(500, left_target_width))
                self.setMinimumWidth(left_target_width + 420)

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
        current_tcp_euler = state.get('tcp_euler_deg', np.zeros(3))

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

        if Qt.Key_Left in self._keys_pressed:
            drot[2] += self._rot_step_size
        if Qt.Key_Right in self._keys_pressed:
            drot[2] -= self._rot_step_size

        if np.any(dpos != 0) or np.any(drot != 0):
            new_pos = current_tcp_pos + dpos
            new_euler = current_tcp_euler + drot

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
        if self._busy:
            return

        self._log_panel.info("开始数据收集...")
        self._control_panel.set_running(True)
        self._reset_task_timer()  # 重置计时器

        try:
            self._data_system.start_collection()
        except Exception as e:
            self._log_panel.error(f"启动失败: {e}")
            import traceback
            traceback.print_exc()
            self._control_panel.set_running(False)

    def _on_stop(self):
        """停止任务"""
        if self._busy:
            self._log_panel.warning("后台任务执行中，请稍候")
            return
        self._log_panel.warning("停止数据收集...")
        self._control_panel.set_busy("停止中...")
        self._status_panel.update_status("停止中", "#ff4444")
        self._stop_task_timer()  # 停止计时器
        if self._data_system:
            self._run_in_background(self._data_system.stop)

    def _on_pause(self):
        """暂停任务"""
        if self._busy:
            self._log_panel.warning("后台任务执行中，请稍候")
            return
        self._log_panel.info("暂停数据收集")
        self._status_panel.update_status("已暂停", "#ffa500")
        self._pause_task_timer()  # 暂停计时器

        if self._data_system:
            self._data_system.pause()

    def _on_resume(self):
        """继续任务"""
        if self._busy:
            self._log_panel.warning("后台任务执行中，请稍候")
            return
        self._log_panel.info("继续数据收集")
        self._status_panel.update_status("运行中", "#44ff44")
        self._resume_task_timer()  # 继续计时器

        if self._data_system:
            self._data_system.resume()

    def _on_skip(self):
        """跳过当前任务"""
        if self._busy:
            self._log_panel.warning("后台任务执行中，请稍候")
            return
        self._log_panel.info("跳过当前任务")
        self._control_panel.set_busy("跳过中...")
        self._stop_task_timer()  # 停止计时器
        if self._data_system:
            self._run_in_background(self._data_system.skip_current_task)

    def _on_next_task(self):
        """执行下一个任务"""
        if self._busy:
            self._log_panel.warning("后台任务执行中，请稍候")
            return
        self._log_panel.info("执行下一个任务")
        self._control_panel.set_busy("启动任务...")
        if self._data_system:
            self._run_in_background(self._data_system.execute_next_task)

    def _on_retry(self):
        """重做当前任务"""
        if self._busy:
            self._log_panel.warning("后台任务执行中，请稍候")
            return
        self._log_panel.warning("重做当前任务")
        self._control_panel.set_busy("重试中...")
        self._reset_task_timer()  # 重置计时器（重试会重新启动）
        if self._data_system:
            self._run_in_background(self._data_system.retry_current_task)

    def _on_complete_task(self):
        """手动确认当前任务完成"""
        if self._busy:
            self._log_panel.warning("后台任务执行中，请稍候")
            return
        self._log_panel.info("手动确认任务完毕")
        self._control_panel.set_busy("完成任务中...")
        self._stop_task_timer()  # 停止计时器
        if self._data_system:
            self._run_in_background(self._data_system.finish_current_task)

    def _run_in_background(self, func, on_done=None):
        """将阻塞操作放到后台线程执行，完成后通过信号通知 GUI"""
        if self._busy:
            return
        self._busy = True

        def worker():
            try:
                func()
            except Exception as e:
                self._log_panel.error(f"后台任务异常: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if on_done:
                    on_done()
                self._bg_task_done_signal.emit()

        import threading
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_bg_task_done(self):
        """后台任务完成后恢复 UI"""
        self._busy = False
        self._control_panel.set_idle()

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
        """更新任务信息（可在任意线程调用，通过信号转到GUI线程）"""
        self._task_info_signal.emit(task_name, current, total)

    def _on_update_task_info(self, task_name: str, current: int, total: int):
        """任务信息信号槽（在 GUI 线程执行）"""
        self._status_panel.update_task_info(task_name, current, total)
        # 当新任务开始时，重置并启动计时器
        if current > 0:
            self._reset_task_timer()
            self._start_task_timer()

    def closeEvent(self, event):
        """关闭事件"""
        # 取消所有 broker 订阅
        self._broker.unsubscribe(SIMU_IMAGES, self._on_simu_images)
        self._broker.unsubscribe(SIMU_JOINTS, self._on_simu_joints)
        self._broker.unsubscribe(SIMU_GRIPPER, self._on_simu_gripper)
        self._broker.unsubscribe(SIMU_CARTESIAN, self._on_simu_cartesian)
        self._broker.unsubscribe(SIMU_TCP_POSE, self._on_simu_tcp)
        self._broker.unsubscribe(SIMU_OBJECT_POS, self._on_simu_object_pos)
        self._broker.unsubscribe(SIMU_STATUS, self._on_simu_status)
        self._broker.unsubscribe(REAL_IMAGES, self._on_real_images)
        self._broker.unsubscribe(REAL_JOINTS, self._on_real_joints)
        self._broker.unsubscribe(REAL_GRIPPER, self._on_real_gripper)
        self._broker.unsubscribe(REAL_CARTESIAN, self._on_real_cartesian)
        self._broker.unsubscribe(REAL_STATUS, self._on_real_status)

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
