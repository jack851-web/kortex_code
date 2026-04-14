"""
状态面板组件
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QGridLayout, QProgressBar
)
from PyQt5.QtCore import Qt


class StatusPanel(QWidget):
    """状态面板 - 显示任务状态和机器人状态"""
    
    def __init__(self):
        super().__init__()
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 当前任务
        self._task_label = QLabel("当前任务: -")
        self._task_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self._task_label)

        # 进度
        progress_layout = QHBoxLayout()
        self._progress_label = QLabel("进度: 0/0")
        progress_layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_bar, stretch=1)
        layout.addLayout(progress_layout)

        # 状态
        self._status_label = QLabel("就绪")
        self._status_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #666;")
        layout.addWidget(self._status_label)

        # 机器人状态组
        self._robot_group = QGroupBox("机器人状态")
        robot_layout = QVBoxLayout(self._robot_group)
        
        # 关节位置
        joint_layout = QGridLayout()
        joint_layout.setHorizontalSpacing(20)  # 增加水平间距
        joint_layout.setVerticalSpacing(5)     # 增加垂直间距
        self._joint_labels = []
        for i in range(6):
            label = QLabel(f"T{i}: 0.00")
            label.setMinimumWidth(100)  # 设置最小宽度
            joint_layout.addWidget(label, i // 3, i % 3)
            self._joint_labels.append(label)
        robot_layout.addLayout(joint_layout)
        
        # 笛卡尔坐标
        pos_layout = QGridLayout()
        pos_layout.setHorizontalSpacing(20)  # 增加水平间距
        pos_layout.setVerticalSpacing(5)     # 增加垂直间距
        self._pos_labels = []
        pos_names = ["X", "Y", "Z", "RX", "RY", "RZ"]
        for i, name in enumerate(pos_names):
            label = QLabel(f"{name}: 0.0000")
            label.setMinimumWidth(100)  # 设置最小宽度
            pos_layout.addWidget(label, i // 3, i % 3)
            self._pos_labels.append(label)
        robot_layout.addLayout(pos_layout)

        # 夹爪状态
        self._gripper_label = QLabel("夹爪: 0.00")
        robot_layout.addWidget(self._gripper_label)

        layout.addWidget(self._robot_group)

        # 仿真状态组
        self._simu_group = QGroupBox("仿真状态")
        simu_layout = QVBoxLayout(self._simu_group)

        # 仿真关节位置
        simu_joint_layout = QGridLayout()
        simu_joint_layout.setHorizontalSpacing(20)  # 增加水平间距
        simu_joint_layout.setVerticalSpacing(5)     # 增加垂直间距
        self._simu_joint_labels = []
        for i in range(6):
            label = QLabel(f"T{i}: 0.00")
            label.setMinimumWidth(100)  # 设置最小宽度
            simu_joint_layout.addWidget(label, i // 3, i % 3)
            self._simu_joint_labels.append(label)
        simu_layout.addLayout(simu_joint_layout)

        # 仿真笛卡尔坐标
        simu_pos_layout = QGridLayout()
        simu_pos_layout.setHorizontalSpacing(20)  # 增加水平间距
        simu_pos_layout.setVerticalSpacing(5)     # 增加垂直间距
        self._simu_pos_labels = []
        for i, name in enumerate(pos_names):
            label = QLabel(f"{name}: 0.0000")
            label.setMinimumWidth(100)  # 设置最小宽度
            simu_pos_layout.addWidget(label, i // 3, i % 3)
            self._simu_pos_labels.append(label)
        simu_layout.addLayout(simu_pos_layout)

        # 仿真夹爪状态
        self._simu_gripper_label = QLabel("夹爪: 0.00")
        simu_layout.addWidget(self._simu_gripper_label)

        layout.addWidget(self._simu_group)
        
        # 添加弹性空间
        layout.addStretch()

    def set_mode(self, use_real: bool):
        """根据模式调整排版

        use_real=True: 显示实机+仿真双状态
        use_real=False: 仅显示仿真状态，避免冗余
        """
        if use_real:
            self._robot_group.setVisible(True)
            self._simu_group.setTitle("仿真状态")
        else:
            self._robot_group.setVisible(False)
            self._simu_group.setTitle("机器人状态（仿真）")
    
    def update_task_info(self, task_name: str, current: int, total: int):
        """更新任务信息"""
        self._task_label.setText(f"当前任务: {task_name}")
        self._progress_label.setText(f"进度: {current}/{total}")
        if total > 0:
            self._progress_bar.setValue(int(current / total * 100))
    
    def update_status(self, status: str, color: str = "#666"):
        """更新状态"""
        self._status_label.setText(status)
        self._status_label.setStyleSheet(f"font-weight: bold; color: {color};")
    
    def update_joints(self, joints):
        """更新关节位置（输入为角度）"""
        for i, label in enumerate(self._joint_labels):
            if i < len(joints):
                label.setText(f"T{i}: {joints[i]:.2f}")
    
    def update_cartesian(self, pose):
        """更新笛卡尔坐标"""
        pos_names = ["X", "Y", "Z", "RX", "RY", "RZ"]
        for i, (label, name) in enumerate(zip(self._pos_labels, pos_names)):
            if i < len(pose):
                label.setText(f"{name}: {pose[i]:.4f}")
    
    def update_gripper(self, position: float):
        """更新夹爪状态"""
        self._gripper_label.setText(f"夹爪: {position:.2f}")

    def update_simu_joints(self, joints):
        """更新仿真关节位置（输入为弧度，转换为角度显示）"""
        import numpy as np
        for i, label in enumerate(self._simu_joint_labels):
            if i < len(joints):
                # 弧度转角度
                angle_deg = np.rad2deg(joints[i])
                label.setText(f"T{i}: {angle_deg:.2f}")

    def update_simu_cartesian(self, pose):
        """更新仿真笛卡尔坐标"""
        pos_names = ["X", "Y", "Z", "RX", "RY", "RZ"]
        for i, (label, name) in enumerate(zip(self._simu_pos_labels, pos_names)):
            if i < len(pose):
                label.setText(f"{name}: {pose[i]:.4f}")

    def update_simu_gripper(self, position: float):
        """更新仿真夹爪状态"""
        self._simu_gripper_label.setText(f"夹爪: {position:.2f}")
