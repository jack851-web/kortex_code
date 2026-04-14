"""
Mock 模式任务微调窗口
支持键盘控制机器人移动（WASD移动，RF上下，QE旋转，[ ]控制夹爪）
"""

import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QFrame,
)
from PyQt5.QtGui import QKeyEvent


class MockTaskTunerWindow(QWidget):
    def __init__(self, data_system):
        super().__init__()
        self._data_system = data_system

        self._keys_pressed = set()
        self._teleop_step_size = 0.04  # 增大移动步长，提高键盘控制灵敏度
        self._rot_step_size = 12.0  # 增大旋转步长（度）

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_state)
        self._timer.timeout.connect(self._process_keyboard_teleop)

        self._current_tcp_pos = np.zeros(3)
        self._current_tcp_euler = np.zeros(3)
        self._gripper_state = 0.0

        self._init_ui()
        self._timer.start(50)
        self.setFocusPolicy(Qt.StrongFocus)

    def _init_ui(self):
        self.setWindowTitle("Mock 模式 - 键盘控制")
        self.setMinimumSize(500, 400)

        root = QVBoxLayout(self)

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

        root.addWidget(info_group)

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

        root.addWidget(task_group)

        state_group = QGroupBox("实时状态")
        state_layout = QGridLayout(state_group)

        self._pose_label = QLabel("TCP: x=0 y=0 z=0")
        state_layout.addWidget(self._pose_label, 0, 0, 1, 2)

        self._gripper_label = QLabel("夹爪: 0.000")
        state_layout.addWidget(self._gripper_label, 1, 0, 1, 2)

        root.addWidget(state_group)

        btn_layout = QHBoxLayout()
        self._to_home_btn = QPushButton("回初始位 (Z)")
        self._to_home_btn.clicked.connect(self._move_home)
        btn_layout.addWidget(self._to_home_btn)

        self._finish_btn = QPushButton("任务完毕")
        self._finish_btn.setStyleSheet("background-color: #E91E63; color: white;")
        self._finish_btn.clicked.connect(self._finish_task)
        btn_layout.addWidget(self._finish_btn)

        root.addLayout(btn_layout)

        self._status_label = QLabel("等待任务开始...")
        self._status_label.setStyleSheet("color: #666;")
        root.addWidget(self._status_label)

    def keyPressEvent(self, event: QKeyEvent):
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
        self._keys_pressed.discard(event.key())
        super().keyReleaseEvent(event)

    def _process_keyboard_teleop(self):
        if not self._keys_pressed:
            return

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
            self._move_delta(dpos, drot)

    def _move_delta(self, dpos: np.ndarray, drot: np.ndarray):
        if self._data_system is None:
            return

        new_pos = self._current_tcp_pos + dpos
        new_euler = self._current_tcp_euler + np.rad2deg(drot)

        target_pose = np.array([
            new_pos[0], new_pos[1], new_pos[2],
            new_euler[0], new_euler[1], new_euler[2]
        ], dtype=float)

        self._data_system.tuning_move_to_pose(target_pose)

    def _move_home(self):
        if self._data_system.move_to_home_pose():
            self._status_label.setText("已回到初始位置")

    def _close_gripper_more(self):
        new_gripper = min(1.0, self._gripper_state + 0.05)
        self._data_system.set_tuning_gripper(new_gripper)
        self._gripper_state = new_gripper
        self._status_label.setText(f"夹爪: {new_gripper:.2f}")

    def _open_gripper_more(self):
        new_gripper = max(0.0, self._gripper_state - 0.05)
        self._data_system.set_tuning_gripper(new_gripper)
        self._gripper_state = new_gripper
        self._status_label.setText(f"夹爪: {new_gripper:.2f}")

    def _finish_task(self):
        self._data_system.finish_current_task()
        self._status_label.setText("任务完成")

    def _refresh_state(self):
        state = self._data_system.get_tuning_state()
        if not state:
            return

        tcp_pos = np.asarray(state.get('tcp_pos', np.zeros(3)), dtype=float)
        gripper = float(state.get('gripper', 0.0))

        self._current_tcp_pos = tcp_pos.copy()
        self._gripper_state = gripper

        task_info = self._data_system.get_current_task_runtime_info()
        task_name = task_info.get('task_name', '-')
        obj_pos = np.asarray(task_info.get('object_pos', np.zeros(3)), dtype=float)
        target_pos = np.asarray(task_info.get('target_pos', np.zeros(3)), dtype=float)

        distance = np.linalg.norm(obj_pos - target_pos)

        self._task_label.setText(f"任务: {task_name}")
        self._object_label.setText(f"x={obj_pos[0]:.3f}, y={obj_pos[1]:.3f}, z={obj_pos[2]:.3f}")
        self._target_label.setText(f"x={target_pos[0]:.3f}, y={target_pos[1]:.3f}, z={target_pos[2]:.3f}")
        self._distance_label.setText(f"{distance:.4f}m")
        self._pose_label.setText(f"TCP: x={tcp_pos[0]:.3f}, y={tcp_pos[1]:.3f}, z={tcp_pos[2]:.3f}")
        self._gripper_label.setText(f"夹爪: {gripper:.3f}")
