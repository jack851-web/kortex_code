"""
物体参数标定窗口
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QDoubleSpinBox,
    QComboBox,
    QLineEdit,
)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve_path(path_str: str, base_dir: Path = None) -> str:
    """解析路径，支持相对路径和绝对路径"""
    if not path_str:
        return path_str
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    base = base_dir or PROJECT_ROOT
    return str((base / path_str).resolve())


def _resolve_config_paths(config: dict, base_dir: Path = None) -> dict:
    """递归解析配置中的路径字段"""
    path_keys = {
        'xml_path', 'model_xml_path', 'scene_base_xml_path',
        'real_data_root', 'simu_data_root', 'mock_simu_data_root',
    }
    
    def _resolve_value(key: str, value):
        if isinstance(value, str):
            is_path = (
                key in path_keys or 
                key.endswith('_path') or 
                key.endswith('_root') or
                key.endswith('_xml') or
                '.xml' in value.lower()
            )
            if is_path:
                return _resolve_path(value, base_dir)
            return value
        elif isinstance(value, dict):
            return {k: _resolve_value(k, v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_resolve_value(key, item) for item in value]
        return value
    
    if config is None:
        return config
    return {k: _resolve_value(k, v) for k, v in config.items()}


class ObjectProfileTunerWindow(QWidget):
    def __init__(self, data_system, config_path: str, default_task_id: Optional[str] = None, default_object_name: Optional[str] = None):
        super().__init__()
        self._data_system = data_system
        self._config_path = config_path
        self._default_task_id = default_task_id
        self._default_object_name = default_object_name

        self._config: Dict[str, Any] = {}
        self._tasks: Dict[str, Any] = {}
        self._object_library: Dict[str, Any] = {}

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_state)

        self._joint_ui_updating = False
        self._joint_apply_timer = QTimer(self)
        self._joint_apply_timer.setSingleShot(True)
        self._joint_apply_timer.timeout.connect(self._apply_joint_controls)

        self._load_config()
        self._init_ui()
        self._timer.start(100)
        QTimer.singleShot(0, self.initialize_tuning)

    def _load_config(self):
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            # 解析配置中的相对路径为绝对路径
            self._config = _resolve_config_paths(self._config, PROJECT_ROOT)
        except Exception:
            self._config = {}
        self._tasks = self._config.get('tasks', {}) if isinstance(self._config, dict) else {}
        simu_cfg = self._config.get('simulation', {}) if isinstance(self._config, dict) else {}
        self._object_library = simu_cfg.get('object_library', {}) if isinstance(simu_cfg, dict) else {}

    def _init_ui(self):
        self.setWindowTitle('物体参数标定器')
        self.setMinimumSize(620, 560)

        root = QVBoxLayout(self)

        task_group = QGroupBox('标定目标')
        task_layout = QGridLayout(task_group)

        task_layout.addWidget(QLabel('Object Name:'), 0, 0)
        self._object_combo = QComboBox()
        self._object_combo.setEditable(True)
        for obj_name in self._object_library.keys():
            self._object_combo.addItem(str(obj_name))
        if self._default_object_name:
            self._object_combo.setCurrentText(self._default_object_name)
        task_layout.addWidget(self._object_combo, 0, 1)

        task_layout.addWidget(QLabel('Object XML Path:'), 1, 0)
        self._object_xml_edit = QLineEdit('')
        task_layout.addWidget(self._object_xml_edit, 1, 1)

        task_layout.addWidget(QLabel('Object Position (x,y,z):'), 2, 0)
        pos_row = QHBoxLayout()
        self._obj_x_spin = QDoubleSpinBox()
        self._obj_y_spin = QDoubleSpinBox()
        self._obj_z_spin = QDoubleSpinBox()
        for spin in (self._obj_x_spin, self._obj_y_spin, self._obj_z_spin):
            spin.setRange(-2.0, 2.0)
            spin.setSingleStep(0.001)
            spin.setDecimals(4)
            pos_row.addWidget(spin)
        task_layout.addLayout(pos_row, 2, 1)

        btn_row = QHBoxLayout()
        self._prepare_btn = QPushButton('加载并IK到目标上方')
        self._prepare_btn.clicked.connect(self.initialize_tuning)
        btn_row.addWidget(self._prepare_btn)

        self._save_btn = QPushButton('保存到 config/object/<object>.yaml')
        self._save_btn.clicked.connect(self._save_profile)
        btn_row.addWidget(self._save_btn)
        task_layout.addLayout(btn_row, 3, 0, 1, 2)

        self._save_result = QLabel('')
        task_layout.addWidget(self._save_result, 4, 0, 1, 2)

        self._fill_defaults_from_config()
        root.addWidget(task_group)

        state_group = QGroupBox('实时状态')
        state_layout = QGridLayout(state_group)

        self._object_label = QLabel('Object: -')
        self._joint_label = QLabel('Joints(deg): [0, 0, 0, 0, 0, 0]')
        self._gripper_label = QLabel('Gripper: 0.000')
        self._pose_label = QLabel('TCP: x=0 y=0 z=0 | rx=0 ry=0 rz=0')
        self._offset_label = QLabel('Offset(夹爪-物体中心): dx=0 dy=0 dz=0 | norm=0')

        state_layout.addWidget(self._object_label, 0, 0)
        state_layout.addWidget(self._joint_label, 1, 0)
        state_layout.addWidget(self._gripper_label, 2, 0)
        state_layout.addWidget(self._pose_label, 3, 0)
        state_layout.addWidget(self._offset_label, 4, 0)

        root.addWidget(state_group)

        joint_group = QGroupBox('关节角度调节（绝对控制）')
        joint_layout = QGridLayout(joint_group)
        self._joint_spins = []
        for i in range(6):
            joint_layout.addWidget(QLabel(f'J{i} (deg):'), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-360.0, 360.0)
            spin.setSingleStep(0.5)
            spin.setDecimals(3)
            self._joint_spins.append(spin)
            spin.valueChanged.connect(self._schedule_auto_apply_joint_controls)
            joint_layout.addWidget(spin, i, 1)

        joint_layout.addWidget(QLabel('Gripper (0~1):'), 6, 0)
        self._gripper_spin = QDoubleSpinBox()
        self._gripper_spin.setRange(0.0, 1.0)
        self._gripper_spin.setSingleStep(0.01)
        self._gripper_spin.setDecimals(3)
        self._gripper_spin.valueChanged.connect(self._schedule_auto_apply_joint_controls)
        joint_layout.addWidget(self._gripper_spin, 6, 1)

        joint_btn_row = QHBoxLayout()
        self._joint_apply_btn = QPushButton('应用关节/夹爪')
        self._joint_apply_btn.clicked.connect(self._apply_joint_controls)
        joint_btn_row.addWidget(self._joint_apply_btn)

        self._joint_read_btn = QPushButton('读取当前姿态')
        self._joint_read_btn.clicked.connect(self._load_joint_controls_from_state)
        joint_btn_row.addWidget(self._joint_read_btn)
        joint_layout.addLayout(joint_btn_row, 7, 0, 1, 2)

        root.addWidget(joint_group)
        root.addStretch()

    def _fill_defaults_from_config(self):
        task_cfg = {}
        if isinstance(self._tasks, dict) and len(self._tasks) > 0:
            if self._default_task_id and self._default_task_id in self._tasks:
                task_cfg = self._tasks.get(self._default_task_id, {})
            else:
                first_task_id = next(iter(self._tasks.keys()))
                task_cfg = self._tasks.get(first_task_id, {})

        cfg_obj_name = self._config.get('object_name', '') if isinstance(self._config, dict) else ''
        if not cfg_obj_name and isinstance(task_cfg, dict):
            cfg_obj_name = task_cfg.get('object_name', '')
        if cfg_obj_name and (not self._default_object_name):
            self._object_combo.setCurrentText(str(cfg_obj_name))

        cfg_obj_pos = self._config.get('object_position', [0.215, -0.614, 0.03]) if isinstance(self._config, dict) else [0.215, -0.614, 0.03]
        if isinstance(task_cfg, dict) and 'object_position' in task_cfg:
            cfg_obj_pos = task_cfg.get('object_position', cfg_obj_pos)
        if isinstance(cfg_obj_pos, (list, tuple)) and len(cfg_obj_pos) >= 3:
            self._obj_x_spin.setValue(float(cfg_obj_pos[0]))
            self._obj_y_spin.setValue(float(cfg_obj_pos[1]))
            self._obj_z_spin.setValue(float(cfg_obj_pos[2]))

        cfg_xml = self._config.get('object_xml_path', '') if isinstance(self._config, dict) else ''
        if not cfg_xml and isinstance(task_cfg, dict):
            cfg_xml = task_cfg.get('object_xml_path', '')
        if not cfg_xml:
            obj_name = self._object_combo.currentText().strip()
            obj_cfg = self._object_library.get(obj_name, {}) if isinstance(self._object_library, dict) else {}
            if isinstance(obj_cfg, dict):
                cfg_xml = obj_cfg.get('model_xml_path', '')
        self._object_xml_edit.setText(str(cfg_xml or ''))

    def _selected_object_name(self) -> Optional[str]:
        text = self._object_combo.currentText().strip()
        return text or None

    def _selected_object_position(self) -> np.ndarray:
        return np.array([
            float(self._obj_x_spin.value()),
            float(self._obj_y_spin.value()),
            float(self._obj_z_spin.value()),
        ], dtype=float)

    def _selected_object_xml_path(self) -> Optional[str]:
        text = self._object_xml_edit.text().strip()
        return text or None

    def initialize_tuning(self):
        object_name = self._selected_object_name()
        object_position = self._selected_object_position()
        object_xml_path = self._selected_object_xml_path()

        ok = self._data_system.prepare_object_tuning(
            task_id=None,
            object_name=object_name,
            object_position=object_position,
            object_model_xml=object_xml_path,
        )
        if ok:
            self._data_system.ensure_simu_viewer()
            self._load_joint_controls_from_state()
            self._save_result.setText('已进入标定模式，可在本窗口调节关节')
        else:
            self._save_result.setText('进入标定模式失败，请看主日志')

    def _refresh_state(self):
        state = self._data_system.get_tuning_state()
        if not state:
            return

        joints_deg = np.asarray(state.get('joints_deg', np.zeros(6)), dtype=float)
        gripper = float(state.get('gripper', 0.0))
        tcp_pos = np.asarray(state.get('tcp_pos', np.zeros(3)), dtype=float)
        tcp_euler = np.asarray(state.get('tcp_euler_deg', np.zeros(3)), dtype=float)
        offset = np.asarray(state.get('offset', np.zeros(3)), dtype=float)

        joint_text = ', '.join([f'{x:.2f}' for x in joints_deg[:6]])
        self._joint_label.setText(f'Joints(deg): [{joint_text}]')
        self._gripper_label.setText(f'Gripper: {gripper:.3f}')

        self._object_label.setText(
            f"Object={state.get('object_name')} | body={state.get('object_body_name')}"
        )
        self._pose_label.setText(
            f"TCP: x={tcp_pos[0]:.4f} y={tcp_pos[1]:.4f} z={tcp_pos[2]:.4f} | "
            f"rx={tcp_euler[0]:.2f} ry={tcp_euler[1]:.2f} rz={tcp_euler[2]:.2f}"
        )
        self._offset_label.setText(
            f"Offset(夹爪-物体中心): dx={offset[0]:.4f} dy={offset[1]:.4f} dz={offset[2]:.4f} | "
            f"norm={float(np.linalg.norm(offset)):.4f}"
        )

    def _load_joint_controls_from_state(self):
        state = self._data_system.get_tuning_state()
        if not state:
            return
        joints_deg = np.asarray(state.get('joints_deg', np.zeros(6)), dtype=float)
        self._joint_ui_updating = True
        try:
            for i, spin in enumerate(self._joint_spins[:6]):
                spin.setValue(float(joints_deg[i]) if i < len(joints_deg) else 0.0)
            self._gripper_spin.setValue(float(state.get('gripper', 0.0)))
        finally:
            self._joint_ui_updating = False

    def _schedule_auto_apply_joint_controls(self, *_):
        if self._joint_ui_updating:
            return
        self._joint_apply_timer.start(30)

    def _apply_joint_controls(self):
        if self._joint_ui_updating:
            return
        joints_deg = np.array([float(spin.value()) for spin in self._joint_spins], dtype=float)
        gripper = float(self._gripper_spin.value())
        ok_j = self._data_system.set_tuning_joints_deg(joints_deg)
        ok_g = self._data_system.set_tuning_gripper(gripper)
        if ok_j and ok_g:
            self._save_result.setText('已自动应用关节/夹爪目标')
        else:
            self._save_result.setText('应用失败，请看日志')

    def _save_profile(self):
        state = self._data_system.get_tuning_state()
        object_name = self._selected_object_name() or state.get('object_name')
        current_gripper = float(state.get('gripper', 0.0))
        out = self._data_system.save_tuning_profile(
            object_name=object_name,
            gripper_open=current_gripper,
            gripper_close=current_gripper,
        )
        if out:
            rel = Path(out).as_posix()
            self._save_result.setText(f'已保存: {rel}')
        else:
            self._save_result.setText('保存失败，请检查日志')
