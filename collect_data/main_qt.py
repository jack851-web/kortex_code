"""
数据收集系统主程序 - Qt GUI 版本
"""
import sys
import argparse
import time
import threading
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from gui import MainWindow, MockTaskTunerWindow
from PyQt5.QtWidgets import QApplication
from scripts.real_interface import RealInterface
from scripts.simu_interface import SimuInterface
from scripts.sync_controller import SyncController
from scripts.grasp_executor import GraspExecutor
from scripts.real_data_collector import RealDataCollector
from scripts.simu_data_collector import SimuDataCollector


class DataCollectionSystem:
    """数据收集系统 - Qt 版本
    
    支持两种模式:
    1. 实机模式 (--real): 同步真实机器人和仿真数据
    2. 模拟模式 (--mock): 使用模拟接口测试
    """
    
    def __init__(self, config_path: str, use_real: bool = True, show_simu_viewer: bool = False):
        self._config_path = config_path
        self._use_real = use_real
        self._show_simu_viewer = bool(show_simu_viewer)
        self._config = None
        
        # 接口
        self._real: Optional[RealInterface] = None
        self._simu: Optional[SimuInterface] = None
        self._sync_controller: Optional[SyncController] = None
        self._grasp_executor: Optional[GraspExecutor] = None
        
        # 数据收集器
        self._real_data_collector: Optional[RealDataCollector] = None
        self._simu_data_collector: Optional[SimuDataCollector] = None
        
        # 状态
        self._running = False
        self._paused = False
        self._current_task_index = 0
        self._current_episode = 1
        self._waiting_for_next_task = False  # 等待用户按键执行下一个任务
        
        # GUI 回调
        self._log_callback = None
        self._status_callback = None
        self._task_callback = None
        
        # 任务线程
        self._task_thread: Optional[threading.Thread] = None
        
        # 坐标变换参数
        self._coord_rotation_z: float = 0.0  # 绕Z轴旋转角度（度）

        # 动态场景/物体配置
        self._simu_base_xml_path: str = ""
        self._object_library: Dict[str, Any] = {}
        self._sim_initial_joints_deg: Optional[np.ndarray] = None

        # 物体参数标定上下文
        self._tuning_task_id: Optional[str] = None
        self._tuning_object_name: Optional[str] = None
        self._tuning_object_body_name: str = "cube"
        self._tuning_object_center: np.ndarray = np.zeros(3, dtype=float)

        # 新模式任务运行态
        self._task_in_progress: bool = False
        self._active_task_info: Dict[str, Any] = {}
        
        # Mock模式吸附机制
        self._mock_adhesion_enabled: bool = False
        self._mock_adhesion_attached: bool = False
        self._mock_adhesion_object_body_name: str = "cube"
        self._mock_adhesion_object_pos: np.ndarray = np.zeros(3, dtype=float)
        self._mock_adhesion_plate_pos: np.ndarray = np.zeros(3, dtype=float)
        self._mock_attach_distance: float = 0.06
        self._mock_detach_distance: float = 0.05
        self._mock_gripper_close_threshold: float = 0.3
    
    def _transform_position(self, position: np.ndarray) -> np.ndarray:
        """将用户坐标系位置转换到MuJoCo坐标系"""
        if self._coord_rotation_z == 0:
            return position.copy()
        
        angle_rad = np.deg2rad(self._coord_rotation_z)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        x, y, z = position[0], position[1], position[2]
        new_x = cos_a * x - sin_a * y
        new_y = sin_a * x + cos_a * y
        
        return np.array([new_x, new_y, z])
    
    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            return True
        except Exception as e:
            self._log(f"加载配置失败: {e}", "ERROR")
            return False
    
    def initialize(self) -> bool:
        """初始化系统"""
        if not self.load_config():
            return False
        
        self._log("初始化数据收集系统...", "INFO")
        
        # 获取配置
        robot_config = self._config.get('robot', {})
        real_camera_config = self._config.get('cameras', {}).get('real', {})
        simu_camera_config = self._config.get('cameras', {}).get('simu', {})
        simu_config = self._config.get('simulation', {})
        grasp_config = self._config.get('grasp', {})
        dataset_config = self._config.get('dataset', {})

        self._simu_base_xml_path = simu_config.get('scene_base_xml_path', simu_config.get('xml_path', ''))
        self._object_library = simu_config.get('object_library', {})

        # 仿真初始关节（可在 config 里设置，支持 rad/deg）
        self._sim_initial_joints_deg = None
        sim_init_joints = simu_config.get('initial_joints', None)
        sim_init_unit = str(simu_config.get('initial_joints_unit', 'rad')).lower()
        if isinstance(sim_init_joints, list) and len(sim_init_joints) >= 6:
            arr = np.array(sim_init_joints[:6], dtype=float)
            if sim_init_unit in ('rad', 'radian', 'radians'):
                self._sim_initial_joints_deg = np.rad2deg(arr)
            else:
                self._sim_initial_joints_deg = arr
            self._log(f"已加载仿真初始关节({sim_init_unit}): {sim_init_joints[:6]}", "INFO")
        
        # 创建接口
        if self._use_real:
            # 实机模式: 同步真实机器人和仿真
            self._log("使用真实机器人接口", "INFO")
            self._real = RealInterface(camera_config=real_camera_config)
            if not self._real.connect(robot_config.get('ip', '192.168.1.10')):
                self._log("连接真实机器人失败", "ERROR")
                return False
            self._log("真实机器人已连接", "SUCCESS")

            # 创建仿真接口（禁用 IK，用于同步）
            self._simu = SimuInterface(simu_config.get('xml_path', ''), use_ik=False)
            if not self._simu.initialize(show_viewer=self._show_simu_viewer):
                self._log("初始化仿真失败", "ERROR")
                return False
            self._log("仿真已初始化", "SUCCESS")
        else:
            # 仿真独立模式: 启用 IK，不依赖实机同步
            self._log("使用仿真独立模式（IK）", "INFO")
            from scripts import MockRealInterface
            self._real = MockRealInterface()
            self._real.connect("mock")

            self._simu = SimuInterface(simu_config.get('xml_path', ''), use_ik=True)
            # Mock 模式下不自动启动 GLFW 查看器，而是在执行任务时按需启动
            if not self._simu.initialize(show_viewer=False):
                self._log("初始化仿真失败", "ERROR")
                return False
            self._log(f"仿真已初始化，IK可用: {self._simu.is_ik_available()}", "SUCCESS")
            # GLFW 查看器将在执行任务时按需启动，结束时关闭

        self._apply_sim_initial_joints()
        
        # 读取坐标变换参数
        self._coord_rotation_z = simu_config.get('coord_rotation_z', 0.0)
        self._log(f"坐标变换: 绕Z轴旋转 {self._coord_rotation_z}°", "INFO")
        
        # 设置相机
        simu_camera_names = list(simu_camera_config.keys()) if simu_camera_config else ['agentview']
        self._simu.set_display_cameras(simu_camera_names)

        if self._use_real:
            try:
                self._simu.start_render_process(simu_camera_names)
                self._log("使用独立渲染进程", "INFO")
            except Exception as e:
                self._log(f"独立渲染进程启动失败，回退主仿真渲染: {e}", "WARNING")
        
        # 创建数据收集器
        video_fps = 30
        if real_camera_config:
            first_cam = list(real_camera_config.values())[0]
            video_fps = first_cam.get('fps', 30)
        
        if self._use_real:
            real_data_root = dataset_config.get('real_data_root', './data/Real/realdata')
            simu_data_root = dataset_config.get('simu_data_root', './data/Real/simudata')
            self._real_data_collector = RealDataCollector(
                self._real,
                data_root=real_data_root,
                fps=dataset_config.get('fps', 20),
                video_fps=video_fps,
            )
            self._simu_data_collector = SimuDataCollector(
                self._simu,
                data_root=simu_data_root,
                fps=dataset_config.get('fps', 20),
                video_fps=video_fps,
                run_in_thread=False,
            )
            self._sync_controller = SyncController(
                self._real,
                self._simu,
                on_sync_callback=self._simu_data_collector.collect_frame
            )
        else:
            simu_data_root = dataset_config.get('mock_simu_data_root', dataset_config.get('simu_data_root', './data/Simu/simu_data'))
            self._real_data_collector = None
            self._simu_data_collector = SimuDataCollector(
                self._simu,
                data_root=simu_data_root,
                fps=dataset_config.get('fps', 20),
                video_fps=video_fps,
                run_in_thread=False,
            )
            self._sync_controller = None
        
        # 创建抓取执行器
        self._grasp_executor = GraspExecutor(
            self._real,
            self._simu,
            home_position=[0, 0, 0, 0, 0, 0],
            pre_grasp_offset=grasp_config.get('pre_grasp_offset', [0, 0, 0.15]),
            lift_height=grasp_config.get('lift_height', 0.20),
            approach_height=grasp_config.get('approach_height', 0.05),
            use_simulation=(not self._use_real),
        )
        
        if 'default_orientation' in grasp_config:
            self._grasp_executor.set_default_orientation(grasp_config['default_orientation'])
        if 'grasp_offsets' in grasp_config:
            self._grasp_executor.set_grasp_offsets(grasp_config['grasp_offsets'])
        if 'gripper_positions' in grasp_config:
            self._grasp_executor.set_gripper_positions_by_object(grasp_config['gripper_positions'])
        if 'object_profiles' in grasp_config:
            self._grasp_executor.set_object_profiles(grasp_config['object_profiles'])
        
        # 设置放置时的抬升高度
        micro_lift = grasp_config.get('micro_lift_height', 0.02)
        release_lift = grasp_config.get('release_lift_height', 0.08)
        self._grasp_executor.set_release_lift_heights(micro_lift, release_lift)

        if (not self._use_real) and self._simu_data_collector is not None:
            self._grasp_executor.set_frame_callback(self._simu_data_collector.collect_frame)
        
        # 加载进度
        progress_collector = self._real_data_collector if self._real_data_collector is not None else self._simu_data_collector
        saved_count = progress_collector.load_progress() if progress_collector is not None else 0
        if saved_count > 0:
            self._log(f"从 Episode {saved_count} 恢复", "INFO")
        
        return True
    
    def set_callbacks(self, log_callback=None, status_callback=None, task_callback=None):
        """设置 GUI 回调"""
        self._log_callback = log_callback
        self._status_callback = status_callback
        self._task_callback = task_callback
    
    def _log(self, message: str, level: str = "INFO"):
        """输出日志"""
        print(f"[{level}] {message}")
        if self._log_callback:
            self._log_callback(message, level)
    
    def _update_status(self, status: str, color: str = "#666"):
        """更新状态"""
        if self._status_callback:
            self._status_callback(status, color)
    
    def _update_task(self, task_name: str, current: int, total: int):
        """更新任务信息"""
        if self._task_callback:
            self._task_callback(task_name, current, total)
    
    def get_real_camera_names(self) -> list:
        """获取真实相机名称列表"""
        if not self._use_real:
            return []
        real_camera_config = self._config.get('cameras', {}).get('real', {})
        return list(real_camera_config.keys()) if real_camera_config else []
    
    def get_simu_camera_names(self) -> list:
        """获取仿真相机名称列表"""
        simu_camera_config = self._config.get('cameras', {}).get('simu', {})
        return list(simu_camera_config.keys()) if simu_camera_config else ['agentview']

    def get_real_camera_config(self) -> dict:
        """获取真实相机配置"""
        if not self._use_real:
            return {}
        return self._config.get('cameras', {}).get('real', {}) or {}

    def get_simu_camera_config(self) -> dict:
        """获取仿真相机配置"""
        return self._config.get('cameras', {}).get('simu', {}) or {}

    def _apply_sim_initial_joints(self):
        if self._simu is None or self._sim_initial_joints_deg is None:
            return
        ok = self._simu.set_joint_target(self._sim_initial_joints_deg)
        if ok:
            self._simu.step(300)
            self._log(f"已应用仿真初始关节(deg): {self._sim_initial_joints_deg.tolist()}", "INFO")

    @staticmethod
    def _rotmat_to_euler_xyz_deg(rot: np.ndarray) -> np.ndarray:
        r = np.asarray(rot, dtype=float).reshape(3, 3)
        sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(r[2, 1], r[2, 2])
            y = np.arctan2(-r[2, 0], sy)
            z = np.arctan2(r[1, 0], r[0, 0])
        else:
            x = np.arctan2(-r[1, 2], r[1, 1])
            y = np.arctan2(-r[2, 0], sy)
            z = 0.0

        return np.rad2deg(np.array([x, y, z], dtype=float))

    @staticmethod
    def _infer_object_body_name_from_xml(object_model_xml: str, fallback: str = "cube") -> str:
        if not object_model_xml:
            return fallback
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(object_model_xml).getroot()
            worldbody = root.find("worldbody")
            if worldbody is None:
                return fallback
            first_body = worldbody.find("body")
            if first_body is None:
                return fallback
            return first_body.attrib.get("name", fallback) or fallback
        except Exception:
            return fallback

    def _resolve_tuning_task(self, task_id: Optional[str], object_name: Optional[str], object_position: Optional[np.ndarray]) -> Tuple[str, str, np.ndarray]:
        tasks = self._config.get('tasks', {}) if isinstance(self._config, dict) else {}
        task_id_resolved = task_id

        if task_id_resolved is None:
            task_id_resolved = next(iter(tasks.keys()), 'task1') if isinstance(tasks, dict) and len(tasks) > 0 else 'task1'

        task_cfg = tasks.get(task_id_resolved, {}) if isinstance(tasks, dict) else {}

        cfg_obj_name = self._config.get('object_name', None) if isinstance(self._config, dict) else None
        obj_name = object_name or cfg_obj_name or task_cfg.get('object_name', 'cube')

        if object_position is None:
            cfg_obj_pos = self._config.get('object_position', None) if isinstance(self._config, dict) else None
            if cfg_obj_pos is not None:
                obj_pos = np.array(cfg_obj_pos[:3], dtype=float)
            else:
                obj_pos = np.array(task_cfg.get('object_position', [0.215, -0.614, 0.03]), dtype=float)
        else:
            obj_pos = np.array(object_position[:3], dtype=float)

        return task_id_resolved, obj_name, obj_pos

    def prepare_object_tuning(
        self,
        task_id: Optional[str] = None,
        object_name: Optional[str] = None,
        object_position: Optional[np.ndarray] = None,
        object_model_xml: Optional[str] = None,
    ) -> bool:
        if self._simu is None or self._grasp_executor is None:
            self._log('系统未初始化，无法进入标定模式', 'ERROR')
            return False

        try:
            task_id_resolved, obj_name, obj_pos = self._resolve_tuning_task(task_id, object_name, object_position)
            self._tuning_task_id = task_id_resolved
            self._tuning_object_name = obj_name

            object_cfg = self._object_library.get(obj_name, {}) if isinstance(self._object_library, dict) else {}
            cfg_xml = self._config.get('object_xml_path', '') if isinstance(self._config, dict) else ''
            object_model_xml = object_model_xml or cfg_xml or object_cfg.get('model_xml_path', '')

            cfg_body_name = object_cfg.get('body_name', '') if isinstance(object_cfg, dict) else ''
            object_body_name = cfg_body_name or self._infer_object_body_name_from_xml(object_model_xml, fallback=obj_name)

            if object_model_xml and self._simu_base_xml_path:
                reloaded = self._simu.reload_scene_with_object(
                    self._simu_base_xml_path,
                    object_model_xml,
                    object_body_name=object_body_name,
                    show_viewer=self._show_simu_viewer,
                )
                if not reloaded:
                    self._log(f'标定模式加载物体失败: {obj_name}, xml={object_model_xml}', 'ERROR')
                    return False
                self._apply_sim_initial_joints()
                self._log(f'标定模式已加载物体: name={obj_name}, body={object_body_name}', 'INFO')

            self._tuning_object_body_name = object_body_name
            self._simu.set_active_object_body_name(object_body_name)
            self._grasp_executor.set_sim_object_body_name(object_body_name)
            self._grasp_executor.set_object_type(obj_name)

            simu_object_pos = self._transform_position(obj_pos)
            self._tuning_object_center = np.asarray(simu_object_pos, dtype=float)
            self._simu.set_object_position(object_body_name, simu_object_pos, reset_z=True)
            self._grasp_executor.set_target(object_position=obj_pos.tolist(), place_position=obj_pos.tolist())

            waypoints = self._grasp_executor.get_waypoints()
            pre_pose = np.array(waypoints[0][0], dtype=float)
            pre_pos = pre_pose[:3]
            pre_ori = self._grasp_executor._euler_xyz_deg_to_rotmat(pre_pose[3:6]) if len(pre_pose) >= 6 else None

            self._log(f'标定模式: 先执行IK到目标上方，task={task_id_resolved}, object={obj_name}', 'INFO')
            ok = self._simu.move_to_cartesian(pre_pos, orientation=pre_ori, duration=2.0, steps=100)
            if not ok:
                self._log('IK到预抓取位失败，可直接手动拖动关节继续标定', 'WARNING')

            self._simu.step(120)
            return True
        except Exception as e:
            self._log(f'prepare_object_tuning 失败: {e}', 'ERROR')
            return False

    def set_tuning_joints_deg(self, joints_deg: np.ndarray) -> bool:
        if self._simu is None:
            return False
        ok = self._simu.set_joint_target(np.asarray(joints_deg[:6], dtype=float))
        if ok:
            self._simu.step(30)
        return ok

    def set_tuning_gripper(self, gripper: float) -> bool:
        if self._simu is None:
            return False
        ok = self._simu.set_gripper(float(np.clip(gripper, 0.0, 1.0)))
        if ok:
            self._simu.step(30)
        return ok

    def tuning_move_to_pose(self, pose: np.ndarray) -> bool:
        """快速移动到指定的TCP姿态（用于键盘遥操作）"""
        if self._simu is None:
            return False
        if self._grasp_executor is None:
            return False
        
        pose = np.asarray(pose[:6], dtype=float)
        
        if hasattr(self._simu, 'move_to_cartesian'):
            target_pos = pose[:3]
            target_ori = None
            if len(pose) >= 6:
                target_ori = self._grasp_executor._euler_xyz_deg_to_rotmat(np.asarray(pose[3:6], dtype=float))
            
            success = self._simu.move_to_cartesian(
                target_pos,
                orientation=target_ori,
                duration=0.3,
                steps=15,
            )
            return success
        
        return False

    def ensure_simu_viewer(self) -> bool:
        """确保查看器已运行，按需启动"""
        if self._simu is None:
            return False
        try:
            # 检查 GLFW 查看器是否已运行（Mock 模式）
            if hasattr(self._simu, '_glfw_viewer_running') and self._simu._glfw_viewer_running:
                return True
            
            # 检查被动查看器是否已运行（实机模式）
            viewer = getattr(self._simu, '_viewer', None)
            if viewer is not None:
                try:
                    if viewer.is_running():
                        return True
                except Exception:
                    pass
            
            # 按需启动查看器
            if self._use_real:
                # 实机模式：启动被动查看器
                if hasattr(self._simu, 'start_viewer'):
                    self._simu.start_viewer()
                    return True
            else:
                # Mock 模式：启动 GLFW 查看器
                if self._show_simu_viewer and hasattr(self._simu, 'start_glfw_viewer'):
                    return self._simu.start_glfw_viewer()
            return False
        except Exception:
            return False

    def get_tuning_state(self) -> Dict[str, Any]:
        if self._simu is None:
            return {}

        try:
            if hasattr(self._simu, 'sync_control_to_current_state'):
                self._simu.sync_control_to_current_state()
            self._simu.step(1)
        except Exception:
            pass

        joints_rad = self._simu.get_joint_state()
        joints_deg = np.rad2deg(joints_rad)
        gripper = float(self._simu.get_gripper_state())

        tcp_pos = np.zeros(3, dtype=float)
        tcp_euler = np.zeros(3, dtype=float)
        if hasattr(self._simu, 'get_tcp_pose'):
            p, r = self._simu.get_tcp_pose()
            if p is not None:
                tcp_pos = np.asarray(p, dtype=float)
            if r is not None:
                tcp_euler = self._rotmat_to_euler_xyz_deg(np.asarray(r, dtype=float))

        obj_pos = np.asarray(self._tuning_object_center, dtype=float)
        if self._tuning_object_body_name:
            body_pos = np.asarray(self._simu.get_object_position(self._tuning_object_body_name), dtype=float)
            if float(np.linalg.norm(body_pos)) > 1e-9:
                obj_pos = body_pos

        offset = tcp_pos - obj_pos

        return {
            'task_id': self._tuning_task_id,
            'object_name': self._tuning_object_name,
            'object_body_name': self._tuning_object_body_name,
            'joints_deg': joints_deg,
            'gripper': gripper,
            'tcp_pos': tcp_pos,
            'tcp_euler_deg': tcp_euler,
            'object_pos': obj_pos,
            'offset': offset,
        }

    def save_tuning_profile(self, object_name: str, gripper_open: float, gripper_close: float) -> Optional[str]:
        try:
            state = self.get_tuning_state()
            if not state:
                return None

            obj_name = object_name or state.get('object_name') or 'unknown'
            out_dir = Path(self._config_path).parent / 'object'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'{obj_name}.yaml'

            payload = {
                'object_name': obj_name,
                'task_id': state.get('task_id'),
                'object_body_name': state.get('object_body_name'),
                'object_center': [float(x) for x in state['object_pos']],
                'tcp_pose': {
                    'position': [float(x) for x in state['tcp_pos']],
                    'orientation_euler_xyz_deg': [float(x) for x in state['tcp_euler_deg']],
                },
                'grasp_offset': [float(x) for x in state['offset']],
                'gripper': {
                    'open': float(np.clip(gripper_open, 0.0, 1.0)),
                    'close': float(np.clip(gripper_close, 0.0, 1.0)),
                    'current': float(state['gripper']),
                },
                'joint_angles_deg': [float(x) for x in state['joints_deg']],
            }

            with open(out_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

            return str(out_path)
        except Exception as e:
            self._log(f'保存标定文件失败: {e}', 'ERROR')
            return None

    def start_collection(self, start_task_index: int = 0, episode: int = 1):
        """开始数据收集 - 一个 episode 包含多个任务"""
        self._current_task_index = start_task_index
        self._current_episode = episode
        self._running = True
        self._paused = False
        self._waiting_for_next_task = False
        
        self._log(f"开始任务索引设置为: {start_task_index}", "INFO")
        
        # 先开始 episode（初始化数据结构）
        self._start_episode()
        
        # 启动数据收集器
        if self._real_data_collector is not None:
            self._real_data_collector.start_collection()
        self._simu_data_collector.start_collection()
        # 仅实机模式启动同步控制器
        if self._sync_controller is not None:
            self._sync_controller.start_sync()
        self._log(f"Episode {self._current_episode} 已启动，准备执行任务", "SUCCESS")
        
        self._update_status("等待执行任务", "#ffa500")
    
    def _start_episode(self):
        """开始一个新的 episode"""
        # 获取所有任务信息
        tasks = self._config.get('tasks', {})
        task_list = list(tasks.items())
        
        # 收集所有任务信息
        all_task_info = []
        for task_id, task_config in task_list:
            all_task_info.append({
                'task_id': task_id,
                'task_name': task_config.get('description', task_id),
                'object_name': task_config.get('object_name', 'cube'),
                'object_position': task_config.get('object_position', [0.215, -0.614, 0.17]),
                'plate_position': task_config.get('plate_position', [0.215, -0.60, 0.17]),
            })
        
        # 开始 episode
        if self._real_data_collector is not None:
            self._real_data_collector.start_episode(
                self._current_episode,
                self.get_real_camera_names(),
                {'tasks': all_task_info}
            )
        self._simu_data_collector.start_episode(
            self._current_episode,
            self.get_simu_camera_names(),
            {'tasks': all_task_info}
        )
    
    def execute_next_task(self):
        """执行下一个任务（由 GUI 按钮触发）"""
        if not self._running:
            self._log("数据收集未启动", "WARNING")
            return

        if self._waiting_for_next_task:
            self._log("任务准备中，请稍候", "WARNING")
            return

        if self._task_in_progress:
            self._log("当前任务尚未点击“抓取任务完毕”", "WARNING")
            return

        tasks = self._config.get('tasks', {})
        task_list = list(tasks.items())

        self._log(f"当前任务索引: {self._current_task_index}, 总任务数: {len(task_list)}", "INFO")

        if self._current_task_index >= len(task_list):
            self._log("所有任务已完成", "SUCCESS")
            self._end_episode()
            return

        self._waiting_for_next_task = True
        task_thread = threading.Thread(target=self._execute_current_task, daemon=True)
        task_thread.start()

    def _execute_current_task(self):
        """准备当前任务（新模式：准备后等待人工完成）"""
        try:
            tasks = self._config.get('tasks', {})
            task_list = list(tasks.items())

            task_id, task_config = task_list[self._current_task_index]
            task_name = task_config.get('description', task_id)

            self._update_task(task_name, self._current_task_index + 1, len(task_list))
            self._log(f"准备任务 {self._current_task_index + 1}/{len(task_list)}: {task_name}", "INFO")
            self._update_status("任务准备中", "#44ff44")

            success = self._execute_task(task_id, task_config)
            self._waiting_for_next_task = False

            if success:
                self._task_in_progress = True
                self._log("任务已就绪，请手动操作并点击“抓取任务完毕”", "INFO")
                self._update_status("等待手动完成", "#ffa500")
            else:
                self._task_in_progress = False
                self._log("任务准备失败", "WARNING")
                self._update_status("准备失败", "#ff4444")

        except Exception as e:
            self._log(f"任务错误: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            self._waiting_for_next_task = False
            self._task_in_progress = False
            self._update_status("错误", "#ff0000")

    def retry_current_task(self):
        """重做当前任务（丢弃当前任务数据，重新准备）"""
        if not self._running:
            self._log("数据收集未启动", "WARNING")
            return

        if self._waiting_for_next_task:
            self._log("任务准备中，请稍后再重做", "WARNING")
            return

        if self._task_in_progress:
            self._log("当前任务进行中，先点击“抓取任务完毕”或停止", "WARNING")
            return

        if self._current_task_index == 0:
            self._log("当前没有任务需要重做", "WARNING")
            return

        self._current_task_index -= 1

        tasks = self._config.get('tasks', {})
        task_list = list(tasks.items())
        task_id, task_config = task_list[self._current_task_index]
        task_name = task_config.get('description', task_id)

        self._log(f"重做任务 {self._current_task_index + 1}/{len(task_list)}: {task_name}", "WARNING")

        if self._real_data_collector is not None:
            self._real_data_collector.discard_current_task()
        self._simu_data_collector.discard_current_task()

        self._waiting_for_next_task = True
        task_thread = threading.Thread(target=self._execute_current_task, daemon=True)
        task_thread.start()

    def finish_current_task(self):
        """手动确认当前任务完成，并进行成功判定"""
        if not self._running:
            self._log("数据收集未启动", "WARNING")
            return

        if not self._task_in_progress:
            self._log("当前没有进行中的任务", "WARNING")
            return

        if self._real_data_collector is not None:
            self._real_data_collector.stop_recording()
        if self._simu_data_collector is not None:
            self._simu_data_collector.stop_recording()

        success = self._evaluate_task_success()
        self._log(f"任务完成: {'成功' if success else '失败'}", "SUCCESS" if success else "WARNING")

        if self._sync_controller is not None and hasattr(self._sync_controller, 'clear_adhesion_targets'):
            self._sync_controller.clear_adhesion_targets()

        # Mock 模式：关闭 GLFW 查看器
        if not self._use_real and self._show_simu_viewer:
            try:
                self._simu.close_glfw_viewer()
                self._log("GLFW 查看器已关闭", "INFO")
            except Exception as e:
                self._log(f"关闭 GLFW 查看器失败: {e}", "WARNING")

        self._task_in_progress = False
        self._active_task_info = {}
        self._current_task_index += 1

        tasks = self._config.get('tasks', {})
        task_total = len(list(tasks.items()))
        if self._current_task_index >= task_total:
            self._log("所有任务已完成，请结束 Episode", "SUCCESS")
            self._update_status("任务完成", "#00aa00")
        else:
            self._update_status("等待执行任务", "#ffa500")
    
    def _end_episode(self):
        """结束当前 episode"""
        if not self._running:
            return
        
        # 结束 episode
        if self._real_data_collector is not None:
            self._real_data_collector.end_episode(self._current_episode, True)
        self._simu_data_collector.end_episode(self._current_episode, True)

        self._log(f"Episode {self._current_episode} 已结束", "SUCCESS")
        if self._real_data_collector is not None:
            self._real_data_collector._save_progress()
        else:
            self._simu_data_collector._save_progress()
        
        self._current_episode += 1
        self._current_task_index = 0
        self._running = False
        self._update_status("已停止", "#ff4444")
    
    def _execute_task(self, task_id: str, task_config: dict) -> bool:
        """准备单个任务（新模式：不自动完成抓放）"""
        try:
            object_pos = np.array(task_config.get('object_position', [0.215, -0.614, 0.17]))
            plate_pos = np.array(task_config.get('plate_position', [0.215, -0.60, 0.17]))

            object_name = task_config.get('object_name', 'cube')
            self._grasp_executor.set_object_type(object_name)
            self._log(f"物体类型: {object_name}", "INFO")

            # 调试：打印 object_library 内容
            self._log(f"DEBUG: _object_library keys: {list(self._object_library.keys()) if self._object_library else 'None'}", "INFO")
            
            object_cfg = self._object_library.get(object_name, {}) if isinstance(self._object_library, dict) else {}
            self._log(f"DEBUG: object_cfg for '{object_name}': {object_cfg}", "INFO")
            
            object_model_xml = object_cfg.get('model_xml_path', '')
            cfg_body_name = object_cfg.get('body_name', '') if isinstance(object_cfg, dict) else ''
            inferred_body_name = self._infer_object_body_name_from_xml(object_model_xml, fallback='cube')
            object_body_name = task_config.get('sim_body_name', cfg_body_name or inferred_body_name)
            
            self._log(f"DEBUG: object_model_xml={object_model_xml}", "INFO")
            self._log(f"DEBUG: object_body_name={object_body_name}", "INFO")

            plate_target_cfg = self._config.get('simulation', {}).get('plate_target', {})
            if not plate_target_cfg:
                plate_target_cfg = self._config.get('plate_target', {})
            plate_model_xml = plate_target_cfg.get('model_xml_path', '')
            plate_body_name = plate_target_cfg.get('body_name', 'body_obj_plate')
            self._log(f"DEBUG: plate_target_cfg={plate_target_cfg}", "INFO")
            self._log(f"DEBUG: plate_model_xml={plate_model_xml}", "INFO")
            self._log(f"DEBUG: plate_body_name={plate_body_name}", "INFO")

            if object_model_xml and self._simu_base_xml_path:
                if plate_model_xml:
                    reloaded = self._simu.reload_scene_with_objects(
                        self._simu_base_xml_path,
                        object_model_xml,
                        object_body_name=object_body_name,
                        plate_model_xml=plate_model_xml,
                        plate_body_name=plate_body_name,
                        show_viewer=False,
                    )
                else:
                    reloaded = self._simu.reload_scene_with_object(
                        self._simu_base_xml_path,
                        object_model_xml,
                        object_body_name=object_body_name,
                        show_viewer=False,
                    )
                if not reloaded:
                    self._log(f"动态加载物体失败: {object_name}", "ERROR")
                    return False
                self._apply_sim_initial_joints()
                self._log(f"已加载任务物体模型: {object_name}", "INFO")

            self._grasp_executor.set_sim_object_body_name(object_body_name)
            self._simu.set_active_object_body_name(object_body_name)

            simu_object_pos = self._transform_position(object_pos)
            simu_plate_pos = self._transform_position(plate_pos)
            self._log(f"坐标变换: {object_pos} -> {simu_object_pos}", "INFO")
            self._log(f"放置目标变换: {plate_pos} -> {simu_plate_pos}", "INFO")

            self._simu.set_object_position(object_body_name, simu_object_pos, reset_z=True)
            self._log(f"物块位置已重置: {simu_object_pos} (body={object_body_name})", "INFO")

            if plate_model_xml and plate_body_name:
                self._simu.set_object_position(plate_body_name, simu_plate_pos, reset_z=True)
                self._log(f"放置目标位置已重置: {simu_plate_pos} (body={plate_body_name})", "INFO")

            for _ in range(3):
                self._simu.get_camera_images(self.get_simu_camera_names())
                time.sleep(0.03)
            self._log("渲染已预热，首任务物体应已加载到相机帧", "INFO")

            self._grasp_executor.set_target(
                object_position=object_pos.tolist(),
                place_position=plate_pos.tolist(),
            )

            self._active_task_info = {
                'task_id': task_id,
                'task_name': task_config.get('description', task_id),
                'object_name': object_name,
                'object_body_name': object_body_name,
                'object_pos': np.asarray(simu_object_pos, dtype=float),
                'target_pos': np.asarray(simu_plate_pos, dtype=float),
                'target_pos_user': np.asarray(plate_pos, dtype=float),
            }

            if self._real_data_collector is not None:
                self._real_data_collector.start_recording()
            self._simu_data_collector.start_recording()

            if self._use_real:
                if self._sync_controller is not None and hasattr(self._sync_controller, 'set_adhesion_targets'):
                    self._sync_controller.set_adhesion_targets(
                        object_body_name=object_body_name,
                        object_pos=simu_object_pos,
                        plate_pos=simu_plate_pos,
                    )
                self._log("实机模式: 已启动关节同步+吸附机制，请手动抓放", "INFO")
            else:
                # mock 模式：启动 GLFW 查看器显示场景
                self._log("Mock模式: 启动仿真窗口", "INFO")
                # 检查模型是否已加载
                self._log(f"DEBUG: _simu._model={self._simu._model is not None}, _simu._data={self._simu._data is not None}", "INFO")
                if self._show_simu_viewer:
                    self._log(f"DEBUG: 调用 start_glfw_viewer, _show_simu_viewer={self._show_simu_viewer}", "INFO")
                    viewer_started = self._simu.start_glfw_viewer(
                        width=1200,
                        height=900,
                        title=f"Mock Mode - {task_config.get('description', task_id)}"
                    )
                    if viewer_started:
                        self._log("GLFW 查看器已启动", "SUCCESS")
                    else:
                        self._log("GLFW 查看器启动失败", "WARNING")
                else:
                    self._log(f"DEBUG: _show_simu_viewer=False, 跳过查看器启动", "INFO")
                # 自动 IK 到预抓取位点
                self.move_current_task_to_pre_grasp()

            return True
        except Exception as e:
            if self._real_data_collector:
                self._real_data_collector.stop_recording()
            if self._simu_data_collector:
                self._simu_data_collector.stop_recording()
            self._log(f"任务执行错误: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

    def _evaluate_task_success(self) -> bool:
        if not self._active_task_info:
            return False
        if not hasattr(self._simu, 'get_object_position'):
            return True

        body_name = self._active_task_info.get('object_body_name', 'cube')
        target_pos = np.asarray(self._active_task_info.get('target_pos', np.zeros(3)), dtype=float)
        object_pos = np.asarray(self._simu.get_object_position(body_name), dtype=float)

        distance_xy = np.linalg.norm(object_pos[:2] - target_pos[:2])
        z_diff = abs(object_pos[2] - target_pos[2])
        xy_threshold = 0.08
        z_threshold = 0.05

        self._log(f"任务判定: body={body_name}, object={object_pos}, target={target_pos}", "INFO")
        self._log(f"任务判定距离: xy={distance_xy:.4f}m, z={z_diff:.4f}m", "INFO")

        return bool(distance_xy < xy_threshold and z_diff < z_threshold)

    def get_current_task_runtime_info(self) -> Dict[str, Any]:
        info = dict(self._active_task_info) if isinstance(self._active_task_info, dict) else {}
        body_name = info.get('object_body_name', '')
        if body_name and hasattr(self._simu, 'get_object_position'):
            info['object_pos'] = np.asarray(self._simu.get_object_position(body_name), dtype=float)
        return info

    def move_current_task_to_pre_grasp(self) -> bool:
        if self._grasp_executor is None:
            return False
        waypoints = self._grasp_executor.get_waypoints()
        if not waypoints:
            return False
        pose = np.asarray(waypoints[0][0], dtype=float)
        self._log("执行 IK 到预抓取位", "INFO")
        return bool(self._grasp_executor._move_to_position(pose))

    def move_current_task_to_grasp(self) -> bool:
        if self._grasp_executor is None:
            return False
        waypoints = self._grasp_executor.get_waypoints()
        if len(waypoints) < 2:
            return False
        pose = np.asarray(waypoints[1][0], dtype=float)
        self._log("执行 IK 到抓取位", "INFO")
        return bool(self._grasp_executor._move_to_position(pose))

    def move_current_task_to_lift(self) -> bool:
        if self._grasp_executor is None:
            return False
        waypoints = self._grasp_executor.get_waypoints()
        if len(waypoints) < 3:
            return False
        pose = np.asarray(waypoints[2][0], dtype=float)
        self._log("执行 IK 到抬起位", "INFO")
        return bool(self._grasp_executor._move_to_position(pose))

    def move_current_task_to_pre_place(self) -> bool:
        if self._grasp_executor is None:
            return False
        waypoints = self._grasp_executor.get_waypoints()
        if len(waypoints) < 4:
            return False
        pose = np.asarray(waypoints[3][0], dtype=float)
        self._log("执行 IK 到预放置位", "INFO")
        return bool(self._grasp_executor._move_to_position(pose))

    def move_current_task_to_place(self) -> bool:
        if self._grasp_executor is None:
            return False
        waypoints = self._grasp_executor.get_waypoints()
        if len(waypoints) < 5:
            return False
        pose = np.asarray(waypoints[4][0], dtype=float)
        self._log("执行 IK 到放置位", "INFO")
        return bool(self._grasp_executor._move_to_position(pose))

    def move_to_home_pose(self) -> bool:
        if self._grasp_executor is None:
            return False
        home_pose = np.zeros(6)
        return bool(self._grasp_executor._move_to_position(home_pose))

    def get_gripper_close_position(self) -> float:
        if self._grasp_executor is None:
            return 0.65
        return float(self._grasp_executor._gripper_close_position)

    def set_mock_adhesion_targets(
        self,
        object_body_name: str,
        object_pos: np.ndarray,
        plate_pos: np.ndarray,
        attach_distance: float = 0.06,
        detach_distance: float = 0.05,
        gripper_close_threshold: float = 0.3,
    ):
        """设置Mock模式吸附目标"""
        self._mock_adhesion_object_body_name = object_body_name or "cube"
        self._mock_adhesion_object_pos = np.asarray(object_pos[:3], dtype=float)
        self._mock_adhesion_plate_pos = np.asarray(plate_pos[:3], dtype=float)
        self._mock_attach_distance = float(max(0.005, attach_distance))
        self._mock_detach_distance = float(max(0.005, detach_distance))
        self._mock_gripper_close_threshold = float(np.clip(gripper_close_threshold, 0.0, 1.0))
        self._mock_adhesion_enabled = True
        self._mock_adhesion_attached = False
        self._log(f"Mock吸附目标已设置: object={object_body_name}, attach_dist={self._mock_attach_distance}m", "INFO")

    def clear_mock_adhesion_targets(self):
        """清除Mock模式吸附目标"""
        self._mock_adhesion_enabled = False
        self._mock_adhesion_attached = False
        self._log("Mock吸附目标已清除", "INFO")

    def is_mock_adhesion_attached(self) -> bool:
        """获取Mock模式吸附状态"""
        return self._mock_adhesion_attached

    def update_mock_adhesion(self):
        """更新Mock模式吸附状态（应在关节/夹爪控制后调用）"""
        if not self._mock_adhesion_enabled:
            return
        if self._simu is None:
            return
        if not hasattr(self._simu, 'get_tcp_position'):
            return

        tcp_pos = self._simu.get_tcp_position()
        if tcp_pos is None:
            return
        tcp_pos = np.asarray(tcp_pos[:3], dtype=float)

        body_name = self._mock_adhesion_object_body_name
        obj_pos = np.asarray(self._simu.get_object_position(body_name), dtype=float)

        gripper_state = float(self._simu.get_gripper_state())
        gripper_is_closed = gripper_state > self._mock_gripper_close_threshold

        if not self._mock_adhesion_attached:
            dist_to_object = np.linalg.norm(tcp_pos - obj_pos)
            if dist_to_object <= self._mock_attach_distance and gripper_is_closed:
                self._mock_adhesion_attached = True
                self._log(f"Mock物体已吸附: body={body_name}, dist={dist_to_object:.4f}m, gripper={gripper_state:.3f}", "INFO")

        if self._mock_adhesion_attached:
            if gripper_is_closed:
                self._simu.set_object_position(body_name, tcp_pos, reset_z=False)
            else:
                dist_to_plate = np.linalg.norm(obj_pos - self._mock_adhesion_plate_pos)
                if dist_to_plate <= self._mock_detach_distance:
                    self._mock_adhesion_attached = False
                    self._simu.set_object_position(body_name, self._mock_adhesion_plate_pos, reset_z=True)
                    self._log(f"Mock物体已脱附并放置: body={body_name}, dist_to_plate={dist_to_plate:.4f}m", "INFO")
                else:
                    self._mock_adhesion_attached = False
                    self._log(f"Mock物体已脱附(夹爪松开): body={body_name}, gripper={gripper_state:.3f}", "INFO")

    def manual_attach_object(self) -> bool:
        """手动强制吸附物体"""
        if not self._mock_adhesion_enabled:
            self._log("吸附未启用，请先设置吸附目标", "WARNING")
            return False
        self._mock_adhesion_attached = True
        self._log("手动强制吸附物体", "INFO")
        return True

    def manual_detach_object(self) -> bool:
        """手动强制脱附物体"""
        if not self._mock_adhesion_attached:
            return False
        self._mock_adhesion_attached = False
        if self._simu is not None:
            body_name = self._mock_adhesion_object_body_name
            self._simu.set_object_position(body_name, self._mock_adhesion_plate_pos, reset_z=True)
        self._log("手动强制脱附物体", "INFO")
        return True

    def stop(self):
        """停止数据收集"""
        self._running = False
        self._paused = False
        
        if self._sync_controller:
            self._sync_controller.stop_sync()
        
        # 结束并保存 episode
        if self._real_data_collector:
            if self._real_data_collector._current_episode_info:
                self._real_data_collector.end_episode(self._current_episode, True)
                self._log(f"Episode {self._current_episode} 已保存", "SUCCESS")
            self._real_data_collector.stop_collection()
        
        if self._simu_data_collector:
            if self._simu_data_collector._current_episode_info:
                self._simu_data_collector.end_episode(self._current_episode, True)
            self._simu_data_collector.stop_collection()
        
        self._log("数据收集已停止", "WARNING")
        self._update_status("已停止", "#ff4444")
    
    def pause(self):
        """暂停数据收集"""
        if not self._running:
            self._log("数据收集未启动", "WARNING")
            return
        
        if self._waiting_for_next_task:
            self._log("任务执行中，请执行完毕后再暂停", "WARNING")
            return
        
        self._paused = True
        self._log("数据收集已暂停", "WARNING")
        self._update_status("已暂停", "#ffa500")
    
    def resume(self):
        """继续数据收集"""
        if not self._running:
            self._log("数据收集未启动", "WARNING")
            return
        
        self._paused = False
        self._log("数据收集已继续", "INFO")
        self._update_status("运行中", "#44ff44")
    
    def skip_current_task(self):
        """跳过当前任务"""
        if not self._running:
            self._log("数据收集未启动", "WARNING")
            return
        
        if self._waiting_for_next_task:
            self._log("任务执行中，请执行完毕后再跳过", "WARNING")
            return
        
        # 停止数据记录
        if self._real_data_collector is not None:
            self._real_data_collector.stop_recording()
        if self._simu_data_collector is not None:
            self._simu_data_collector.stop_recording()

        # Mock 模式：关闭 GLFW 查看器
        if not self._use_real and self._show_simu_viewer:
            try:
                self._simu.close_glfw_viewer()
                self._log("GLFW 查看器已关闭", "INFO")
            except Exception as e:
                self._log(f"关闭 GLFW 查看器失败: {e}", "WARNING")

        self._task_in_progress = False
        self._active_task_info = {}
        self._current_task_index += 1
        self._log("已跳过当前任务", "WARNING")
    
    def cleanup(self):
        """清理资源"""
        self.stop()
        
        if self._simu:
            # 关闭 GLFW 查看器 (Mock 模式)
            try:
                self._simu.close_glfw_viewer()
            except Exception as e:
                pass
            self._simu.stop_render_process()
            self._simu.disconnect()
        
        if self._real:
            self._real.disconnect()
        
        self._log("系统已清理", "INFO")


def main():
    script_dir = Path(__file__).parent
    default_config = str(script_dir / "config" / "tasks_config.yaml")
    
    parser = argparse.ArgumentParser(description="Kortex 数据收集系统 - Qt GUI")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="配置文件路径"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="使用真实机器人（实机同步模式）"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用模拟接口（测试用）"
    )
    args = parser.parse_args()

    if args.real:
        use_real = True
        mode_name = "实机同步模式"
    elif args.mock:
        use_real = False
        mode_name = "模拟模式"
    else:
        use_real = False
        mode_name = "模拟模式 (默认)"

    print(f"启动模式: {mode_name}")

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow(args.config, mock_mode=not use_real)
    window.show()

    system = DataCollectionSystem(
        args.config,
        use_real=use_real,
        show_simu_viewer=not use_real
    )

    window.set_data_system(system)

    system.set_callbacks(
        log_callback=window.log,
        status_callback=window.update_status,
        task_callback=window.update_task_info,
    )

    window.log(f"正在初始化系统 ({mode_name})...", "INFO")
    if not system.initialize():
        window.log("系统初始化失败", "ERROR")
        return 1

    if use_real:
        window.setup_cameras(
            system.get_real_camera_names(),
            system.get_simu_camera_names(),
            system.get_real_camera_config(),
            system.get_simu_camera_config(),
        )

    window.log("系统初始化完成", "SUCCESS")

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
