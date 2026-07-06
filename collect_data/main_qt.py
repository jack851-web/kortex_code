"""
数据收集系统主程序 - Qt GUI 版本
"""
import sys
import argparse
import socket
import time
import logging
import traceback
import threading
import yaml
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # kortex_code

# 项目根目录 (kortex_code)
PROJECT_ROOT = Path(__file__).parent.parent

from gui import MainWindow, MockTaskTunerWindow
from PyQt5.QtWidgets import QApplication
from scripts import (
    RealInterface,
    SimuInterface, SimuStubInterface,
    SyncController, GraspExecutor,
    RealDataCollector, SimuDataCollector,
    MessageBroker,
    SimuManager, SimuPublisher, RealPublisher,
    resolve_path, resolve_config_paths,
)


class DataCollectionSystem:
    """数据收集系统 - Qt 版本

    支持三种模式:
    1. 实机模式 (--real): 仅采集真实机器人数据
    2. 模拟模式 (--mock): 使用仿真接口测试
    3. 遥操作模式 (--teleop): 手机ARCore遥操作控制
    """

    def __init__(self, config_path: str, use_real: bool = True, show_simu_viewer: bool = False, teleop_mode: bool = False):
        self._config_path = config_path
        self._use_real = use_real
        self._show_simu_viewer = bool(show_simu_viewer)
        self._teleop_mode = teleop_mode
        self._config = None
        
        # 消息总线（全局单例）
        self._broker = MessageBroker.instance()
        
        # 接口
        self._real: Optional[RealInterface] = None
        self._simu: Optional[SimuInterface] = None
        self._sync_controller: Optional[SyncController] = None
        self._grasp_executor: Optional[GraspExecutor] = None
        
        # 仿真管理器（接管 SimuInterface 的生命周期）
        self._simu_manager = SimuManager(self._broker)
        
        # 数据发布者
        self._simu_publisher: Optional[SimuPublisher] = None
        self._real_publisher: Optional[RealPublisher] = None
        
        # 数据收集器
        self._real_data_collector: Optional[RealDataCollector] = None
        self._simu_data_collector: Optional[SimuDataCollector] = None
        
        # 状态
        self._running = False
        self._paused = False
        self._current_task_index = 0
        self._current_episode = 1
        self._waiting_for_next_task = False  # 等待用户按键执行下一个任务
        self._real_connected = False  # 真实机器人连接状态
        
        # GUI 回调
        self._log_callback = None
        self._status_callback = None
        self._task_callback = None
        
        # 任务线程
        self._task_thread: Optional[threading.Thread] = None

        # tqdm 进度重定向（LeRobot 内部使用 tqdm，重定向到 Qt 日志）
        self._redirect_tqdm_to_log()
        
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
        self._finish_lock = threading.Lock()  # 防 finish_current_task 重入

        # 遥操作模式
        self._ws_server = None
        self._teleop_controller = None
        self._teleop_axis_mapping: Dict[str, str] = {}  # 缓存手机端标定结果
        self._teleop_rot_mapping: Dict[str, str] = {}  # 缓存手机端姿态标定结果
        self._calibration_mode = False  # 标定模式标志
        self._state_sync_timer: Optional[threading.Timer] = None  # 定期状态同步定时器
    
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

    def _generate_random_position(
        self,
        existing_position: Optional[np.ndarray] = None,
        min_distance: float = 0.15,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """在工作空间内生成随机位置
        
        Args:
            existing_position: 已有位置（如object_position），生成的位置需保持最小距离
            min_distance: 与已有位置的最小距离（当existing_position不为None时生效）
            seed: 随机种子（可选，用于调试）
        
        Returns:
            [x, y, z] 随机位置数组
        """
        rng = np.random.RandomState(seed)
        
        # 从配置获取工作空间参数（支持新旧两种格式）
        workspace = self._config.get('workspace', {})
        safety_margin = workspace.get('safety_margin', 0.05)

        if 'x_range' in workspace and 'y_range' in workspace and 'z_range' in workspace:
            # 新格式：直接 xyz 坐标范围 [min, max]
            x_range = workspace['x_range']
            y_range = workspace['y_range']
            z_range = workspace['z_range']
        elif 'table_bounds' in workspace:
            # 兼容旧格式：[x_min, x_max, y_min, y_max, z_surface]
            tb = workspace['table_bounds']
            x_range = [tb[0], tb[1]]
            y_range = [tb[2], tb[3]]
            z_range = [tb[4], tb[4]]
        else:
            x_range = [0.25, 0.50]
            y_range = [-0.25, 0.25]
            z_range = [0.44, 0.44]

        x_min, x_max = x_range[0] + safety_margin, x_range[1] - safety_margin
        y_min, y_max = y_range[0] + safety_margin, y_range[1] - safety_margin
        z_min, z_max = z_range[0], z_range[1]

        max_attempts = 100
        for _ in range(max_attempts):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            z = rng.uniform(z_min, z_max) if z_min != z_max else z_min

            new_pos = np.array([x, y, z])

            # 如果有已有位置，检查距离
            if existing_position is not None:
                distance = np.linalg.norm(new_pos[:2] - existing_position[:2])  # 仅检查xy平面
                if distance < min_distance:
                    continue  # 距离太近，重新生成

            self._log(f"生成随机位置: [{x:.3f}, {y:.3f}, {z:.3f}]", "INFO")
            return new_pos

        # 超过最大尝试次数，返回一个默认位置
        z_default = (z_min + z_max) / 2
        self._log(f"随机位置生成失败（尝试{max_attempts}次），使用默认位置", "WARNING")
        return np.array([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            z_default
        ])
    
    def _resolve_position(
        self,
        position_config,
        existing_position: Optional[np.ndarray] = None,
        min_distance: float = 0.15
    ) -> np.ndarray:
        """解析位置配置，支持坐标数组或"random"字符串
        
        Args:
            position_config: 位置配置，可以是[x,y,z]数组或"random"字符串
            existing_position: 已有位置（用于random时保持最小距离）
            min_distance: 最小距离
        
        Returns:
            [x, y, z] 位置数组
        """
        if position_config is None:
            return np.array([0.35, 0.0, 0.44])
        
        if isinstance(position_config, str) and position_config.lower() == "random":
            return self._generate_random_position(
                existing_position=existing_position,
                min_distance=min_distance
            )
        
        # 假设是坐标数组
        return np.array(position_config[:3], dtype=float)

    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            # 解析配置中的相对路径为绝对路径
            self._config = resolve_config_paths(self._config, PROJECT_ROOT)
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
        _simu_cam_cfg = self._config.get('cameras', {}).get('simu', [])
        # cameras.simu 支持列表 [name, ...] 或字典 {name: {...}} 格式
        if isinstance(_simu_cam_cfg, list):
            self._simu_camera_names = _simu_cam_cfg if _simu_cam_cfg else ['agentview']
        else:
            self._simu_camera_names = list(_simu_cam_cfg.keys()) if _simu_cam_cfg else ['agentview']
        simu_config = self._config.get('simulation', {})
        grasp_config = self._config.get('grasp', {})
        dataset_config = self._config.get('dataset', {})

        self._simu_base_xml_path = simu_config.get('xml_path', '')
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

        # 仿真初始夹爪 (0=张开, 1=闭合)
        self._sim_initial_gripper = float(simu_config.get('initial_gripper', 0.0))
        
        # 创建接口
        if self._use_real:
            # 实机模式: 仅真实机器人 + 真实相机
            self._log("使用真实机器人接口（纯实机模式）", "INFO")
            self._real = RealInterface(camera_config=real_camera_config)

            # 带超时的连接（最多等待5秒）
            real_connected = False
            connect_timeout = robot_config.get('connect_timeout', 5.0)
            robot_ip = robot_config.get('ip', '192.168.1.10')

            connect_result = {'done': False, 'success': False}

            def _connect_thread():
                try:
                    connect_result['success'] = self._real.connect(
                        robot_ip,
                        username=robot_config.get('username'),
                        password=robot_config.get('password'),
                    )
                except Exception as e:
                    self._log(f"连接异常: {e}", "WARNING")
                    connect_result['success'] = False
                finally:
                    connect_result['done'] = True

            t = threading.Thread(target=_connect_thread, daemon=True)
            t.start()
            t.join(timeout=connect_timeout)

            if connect_result['done'] and connect_result['success']:
                real_connected = True
                self._log("真实机器人已连接", "SUCCESS")
            else:
                self._log(f"真实机器人连接超时或失败（等待{connect_timeout}秒），将以离线模式运行", "WARNING")
                real_connected = False

            self._real_connected = real_connected

            if real_connected:
                # 启动真实机器人发布者
                self._real_publisher = RealPublisher(self._real, self._broker, fps=20)
                self._real_publisher.start()
                # 机器人连上后，再连相机（相机初始化可能较慢，不阻塞机器人连接超时）
                self._log("连接相机...", "INFO")
                self._real.connect_cameras()

            # 实机模式：不启动仿真
            self._simu = None
            self._log("实机模式：不启动仿真", "INFO")
        else:
            # 仿真独立模式: 不立即启动仿真，等待标定或任务开始时再启动
            # 这样启动时不会显示关节状态，只有在标定/任务时才会有
            self._log("使用仿真独立模式（IK）", "INFO")
            self._real = None
            self._simu = None
            self._log("仿真将在标定或任务开始时启动", "INFO")

        # 读取坐标变换参数（仅仿真模式使用）
        self._coord_rotation_z = simu_config.get('coord_rotation_z', 0.0)

        # 计算视频帧率
        video_fps = 30
        if real_camera_config:
            first_cam = list(real_camera_config.values())[0]
            video_fps = first_cam.get('fps', 30)

        if self._use_real:
            # 实机模式：仅创建实机数据收集器
            real_data_root = dataset_config.get('real_data_root', './data/Real/realdata')
            real_cam_names = list(real_camera_config.keys()) if real_camera_config else ['cam_0']
            self._real_data_collector = RealDataCollector(
                self._real,
                data_root=real_data_root,
                fps=dataset_config.get('fps', 20),
                video_fps=video_fps,
                broker=self._broker,
                camera_names=real_cam_names,
            )
            self._simu_data_collector = None
            self._sync_controller = None

            # 读取初始关节位置
            init_joints = robot_config.get('initial_joints', None)
            init_gripper = robot_config.get('initial_gripper', 0.0)
            if init_joints is not None:
                initial_joints = np.array(init_joints[:6], dtype=float)
                self._log(f"初始关节(deg): {initial_joints.tolist()}, 夹爪: {init_gripper}", "INFO")
            else:
                initial_joints = np.zeros(6, dtype=float)

            # 创建抓取执行器（实机模式：不传 simu）
            pre_grasp_offs = [0, 0, float(grasp_config.get('pre_grasp_height', 0.05))]
            self._grasp_executor = GraspExecutor(
                self._real,
                None,
                home_position=[0, 0, 0, 0, 0, 0],
                pre_grasp_offset=pre_grasp_offs,
                lift_height=grasp_config.get('lift_height', 0.20),
                approach_height=grasp_config.get('approach_height', 0.05),
                use_simulation=False,
                initial_joints=initial_joints,
                initial_gripper=init_gripper,
            )
        else:
            # 仿真模式：仅仿真数据收集器（延迟绑定simu接口，等仿真启动后再设置）
            simu_data_root = dataset_config.get('mock_simu_data_root', dataset_config.get('simu_data_root', './data/Simu/simu_data'))
            self._real_data_collector = None
            self._simu_data_collector = SimuDataCollector(
                None,  # 初始化时simu还未启动，后续启动仿真后再绑定
                data_root=simu_data_root,
                fps=dataset_config.get('fps', 20),
                video_fps=video_fps,
                run_in_thread=True,
                broker=self._broker,
            )
            self._sync_controller = None

            pre_grasp_offs = [0, 0, float(grasp_config.get('pre_grasp_height', 0.05))]
            # 创建抓取执行器（仿真模式：使用 IK，simu接口延迟绑定）
            self._grasp_executor = GraspExecutor(
                self._real,
                None,  # 初始化时simu还未启动，后续启动仿真后再绑定
                home_position=[0, 0, 0, 0, 0, 0],
                pre_grasp_offset=pre_grasp_offs,
                lift_height=grasp_config.get('lift_height', 0.20),
                approach_height=grasp_config.get('approach_height', 0.05),
                use_simulation=True,
            )

        # 设置每物体抓取参数
        object_profiles = grasp_config.get('object_profiles', {})
        if object_profiles:
            self._grasp_executor.set_object_profiles(object_profiles)

        # 设置放置时的抬升高度
        micro_lift = grasp_config.get('micro_lift_height', 0.02)
        release_lift = grasp_config.get('release_lift_height', 0.08)
        self._grasp_executor.set_release_lift_heights(micro_lift, release_lift)

        if self._simu_data_collector is not None:
            self._grasp_executor.set_frame_callback(self._simu_data_collector.collect_frame)
        
        # 加载进度（恢复上次收集到的 episode 编号）
        progress_collector = self._real_data_collector if self._real_data_collector is not None else self._simu_data_collector
        saved_count = progress_collector.load_progress() if progress_collector is not None else 0
        if saved_count > 0:
            self._current_episode = saved_count + 1
            self._log(f"从进度恢复，下一个 Episode 将为 {self._current_episode} (已有 {saved_count} 条)", "INFO")

        # 遥操作模式初始化
        if self._teleop_mode:
            self._init_teleop_mode()

        return True
    
    def set_callbacks(self, log_callback=None, status_callback=None, task_callback=None):
        """设置 GUI 回调"""
        self._log_callback = log_callback
        self._status_callback = status_callback
        self._task_callback = task_callback

    # ========== 遥操作模式 ==========

    def _init_teleop_mode(self):
        """初始化遥操作模式（WebSocket服务端 + TeleopController）"""
        from scripts.phone_remote import WebSocketServer, TeleopController

        teleop_config = self._config.get('teleop', {})

        # 工作空间限制：将字典格式转换为扁平数组 [xmin, xmax, ymin, ymax, zmin, zmax]
        ws_limits_raw = teleop_config.get('workspace_limits')
        ws_limits_flat = None
        if ws_limits_raw is not None:
            if isinstance(ws_limits_raw, dict):
                # 字典格式: {'x': [min, max], 'y': [min, max], 'z': [min, max]}
                ws_limits_flat = [
                    ws_limits_raw['x'][0], ws_limits_raw['x'][1],
                    ws_limits_raw['y'][0], ws_limits_raw['y'][1],
                    ws_limits_raw['z'][0], ws_limits_raw['z'][1],
                ]
            elif isinstance(ws_limits_raw, (list, tuple)) and len(ws_limits_raw) == 6:
                ws_limits_flat = list(ws_limits_raw)
            self._log(f"工作空间限制: {ws_limits_flat}", "INFO")

        # 创建遥操作控制器
        # 新增参数(vr-teleop-kit 风格):
        #   - pos_reach_limit / rot_reach_limit:增量累积 Reach Limit,鼠标到边语义
        #   - control_rate_hz:控制循环频率(原 20Hz → 默认 50Hz,跟手更紧)
        #   - max_dq_pos / max_dq_rot:每 tick 关节限幅,防奇异处追跑
        #   - ik_fail_reset:连续失败 N 次自动重置目标
        self._teleop_controller = TeleopController(
            robot_interface=self._real,
            simu_interface=self._simu,
            position_scale=teleop_config.get('position_scale', 1.0),
            orientation_scale=teleop_config.get('orientation_scale', 0.5),
            max_linear_speed=teleop_config.get('max_linear_speed', 0.5),
            max_angular_speed=teleop_config.get('max_angular_speed', 30.0),
            workspace_limits=ws_limits_flat,
            initial_joints=self._grasp_executor.initial_joints if self._grasp_executor else None,
            initial_gripper=self._grasp_executor.initial_gripper if self._grasp_executor else 0.0,
            frame_callback=self._simu_data_collector.collect_frame if self._simu_data_collector else None,
            pos_reach_limit=teleop_config.get('pos_reach_limit', 0.25),
            rot_reach_limit=teleop_config.get('rot_reach_limit', 0.6),
            control_rate_hz=teleop_config.get('control_rate_hz', 50.0),
            max_dq_pos=teleop_config.get('max_dq_pos', 0.04),
            max_dq_rot=teleop_config.get('max_dq_rot', 0.10),
            ik_fail_reset=teleop_config.get('ik_fail_reset', 30),
        )

        # 创建 WebSocket 服务
        ws_config = teleop_config.get('websocket', {})
        ws_host = ws_config.get('host', '0.0.0.0')
        ws_port = ws_config.get('port', 8765)
        self._ws_server = WebSocketServer(
            host=ws_host,
            port=ws_port,
            on_pose_delta=self._on_teleop_pose_delta,
            on_gripper=self._on_teleop_gripper,
            on_command=self._on_teleop_command,
            on_connect=self._on_teleop_connect,
            on_disconnect=self._on_teleop_disconnect,
            on_align_direction=self._on_teleop_align_direction,
            on_calib_obs=self._on_teleop_calib_obs,
            on_wrist_pivot=self._on_teleop_wrist_pivot,
            heartbeat_interval=float(ws_config.get('heartbeat_interval', 3.0)),
            heartbeat_timeout=float(ws_config.get('heartbeat_timeout', 6.0)),
        )
        if not self._ws_server.start(wait_startup=2.0):
            self._log(
                f"[ERROR] WebSocket 服务启动失败，端口 {ws_port} 可能被占用。"
                f"请检查端口占用或修改配置文件中的端口。", "ERROR"
            )
            self._ws_server = None
            return

        mode = "simu" if self._simu is not None else "real"
        # 打印所有候选局域网 IP，便于用户在手机端填写
        lan_ips = self._get_lan_ips()
        if ws_host in ('0.0.0.0', '::'):
            ip_hint = "、".join(lan_ips) if lan_ips else "(未找到局域网 IP)"
            self._log(
                f"遥操作模式已启动 ({mode})，监听 {ws_host}:{ws_port}，"
                f"手机请连接到: {ip_hint}", "INFO"
            )
        else:
            self._log(
                f"遥操作模式已启动 ({mode})，监听 {ws_host}:{ws_port}，"
                f"手机请连接到同一 IP", "INFO"
            )

    @staticmethod
    def _get_lan_ips() -> list:
        """获取本机所有 IPv4 地址（排除回环），用于在日志中提示手机端连接 IP"""
        ips = []
        try:
            import socket
            # 通过 UDP 连接获取本机出口 IP（不真正发包）
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                ips.append(s.getsockname()[0])
            finally:
                s.close()
            # 再补充 gethostbyname_ex 的所有地址
            hostname = socket.gethostname()
            try:
                _, _, addrlist = socket.gethostbyname_ex(hostname)
                for ip in addrlist:
                    if ip not in ips and not ip.startswith("127."):
                        ips.append(ip)
            except Exception:
                pass
        except Exception:
            pass
        return ips

    def _on_teleop_connect(self):
        """手机连接成功回调"""
        self._log("手机已连接", "SUCCESS")
        # 发送当前模式
        mode = "simu" if self._simu is not None else "real"
        if self._ws_server:
            self._ws_server.send({"type": "mode", "value": mode})
        # Publisher 接管 step(改造后:控制循环只算 IK 不 step)
        self._set_publisher_auto_step(True)
        # 同步当前状态给手机端
        self._sync_state_to_phone()
        # 启动定期状态同步（每2秒），防止网络抖动导致状态不一致
        self._start_periodic_state_sync()

    def _on_teleop_disconnect(self):
        """手机断开回调"""
        self._log("手机已断开", "WARNING")
        self._stop_periodic_state_sync()
        # Publisher 接管 step,保持画面刷新
        self._set_publisher_auto_step(True)

    def _set_publisher_auto_step(self, auto_step: bool):
        """切换 Publisher 的 auto_step 模式

        改造后(vr-teleop-kit 风格):
          - TeleopController 控制循环只算 IK + set_joint_target,**不 step**
          - SimuPublisher 的 auto_step 负责 mj_step 1 步/帧
          - 两个线程短锁互补,无 simtime 回溯,无锁竞争
        - 真正暂停采集/暂停步进时,临时关闭 auto_step 即可
        """
        if self._simu_manager and self._simu_manager._publisher:
            self._simu_manager._publisher.set_auto_step(auto_step)

    def _on_teleop_pose_delta(self, delta):
        """处理 ARCore 增量位姿：只有在标定模式或任务执行中才转发给控制器"""
        # 提前判断：如果既不在标定模式，也不在任务执行中，直接丢弃
        if not (self._calibration_mode or self._task_in_progress):
            return
        # 检查控制器是否就绪
        if not self._teleop_controller:
            return
        # 检查仿真是否就绪
        if self._simu is None:
            return
        self._teleop_controller.apply_delta(delta)
        # 低频回传实际末端位姿给手机端（每10次回传1次，约3Hz）
        # 关键：使用独立线程异步读取TCP位姿，避免在WebSocket事件循环中阻塞
        # get_tcp_position 内部持仿真锁，同步调用会阻塞事件循环
        self._tcp_feedback_counter = getattr(self, '_tcp_feedback_counter', 0) + 1
        if self._tcp_feedback_counter % 10 == 0 and self._ws_server:
            import threading
            def _send_tcp_feedback():
                try:
                    tcp = self._teleop_controller.get_current_tcp_pose()
                    if tcp is not None:
                        self._ws_server.send({
                            "type": "tcp_pose",
                            "x": float(tcp[0]),
                            "y": float(tcp[1]),
                            "z": float(tcp[2]),
                            "rx": float(tcp[3]) if len(tcp) > 3 else 0.0,
                            "ry": float(tcp[4]) if len(tcp) > 4 else 0.0,
                            "rz": float(tcp[5]) if len(tcp) > 5 else 0.0,
                        })
                except Exception:
                    pass
            threading.Thread(target=_send_tcp_feedback, daemon=True).start()

    def _on_teleop_gripper(self, value: float):
        """处理夹爪控制"""
        logger.info(f"_on_teleop_gripper: value={value}, teleop_controller={self._teleop_controller is not None}")
        if self._teleop_controller:
            self._teleop_controller.apply_gripper(value)
        else:
            self._log(f"[遥操作] 夹爪控制失败: TeleopController 未初始化", "WARNING")

    def _on_teleop_command(self, cmd: dict):
        """处理手机端控制指令

        统一状态流转:
        idle -> start_task -> collection_started
        collection_started -> start_episode -> preparing -> episode_running
        episode_running -> end_episode -> collection_started
        collection_started/episode_running -> end_task -> idle
        idle/collection_started -> calibration_start -> calibrating
        calibrating -> calibration_end -> idle
        """
        cmd_type = cmd.get("type", "")
        # 用 print + flush 确保命令接收一定能被观测到（不依赖日志级别配置）
        print(f"[遥操作] 收到手机指令: {cmd_type}", flush=True)
        self._log(f"[遥操作] 收到手机指令: {cmd_type}", "INFO")

        if cmd_type == "start_task":
            self.start_collection()

        elif cmd_type == "start_episode":
            # 遥操作模式：执行任务（创建场景），对齐和开始采集在 _execute_current_task 中完成
            self.execute_next_task()

        elif cmd_type == "end_episode":
            # 异步执行，避免 finish_current_task 中的 stop_simulation 阻塞 WebSocket 事件循环
            def _end_episode_thread():
                try:
                    # 先停止采集活跃状态
                    if self._teleop_controller:
                        self._teleop_controller.set_collecting_active(False)
                        self._set_publisher_auto_step(True)
                    # 先回到初始位（finish_current_task 会关闭仿真，所以必须先回位）
                    if self._teleop_controller and self._simu_manager and self._simu_manager._simu is not None:
                        self._teleop_controller.move_to_initial()
                    # 结束当前任务（会关闭仿真、保存数据）
                    self.finish_current_task()
                    # 通知手机 episode 已保存
                    if self._ws_server:
                        self._ws_server.send({
                            "type": "episode_saved",
                            "id": self._current_episode,
                            "success": True,
                        })
                    self._sync_state_to_phone()
                except Exception as e:
                    self._log(f"结束任务异常: {e}", "ERROR")
            threading.Thread(target=_end_episode_thread, daemon=True).start()

        elif cmd_type == "retry_episode":
            # 异步执行，避免 stop_simulation 阻塞 WebSocket 事件循环
            def _retry_episode_thread():
                try:
                    if self._teleop_controller:
                        self._teleop_controller.set_collecting_active(False)
                        self._set_publisher_auto_step(True)
                    self.retry_current_task()
                    self._sync_state_to_phone()
                except Exception as e:
                    self._log(f"重做任务异常: {e}", "ERROR")
            threading.Thread(target=_retry_episode_thread, daemon=True).start()

        elif cmd_type == "end_task":
            # 异步执行，避免 stop() 中的 stop_simulation 阻塞 WebSocket 事件循环
            def _end_task_thread():
                try:
                    if self._teleop_controller:
                        self._teleop_controller.set_collecting_active(False)
                        self._set_publisher_auto_step(True)
                    self.stop()
                    if self._ws_server:
                        self._ws_server.send({"type": "task_complete"})
                except Exception as e:
                    self._log(f"结束收集异常: {e}", "ERROR")
            threading.Thread(target=_end_task_thread, daemon=True).start()

        elif cmd_type == "emergency_stop":
            if self._teleop_controller:
                self._teleop_controller.emergency_stop()
            self.stop()
            # stop 内部已调用 _sync_state_to_phone

        elif cmd_type == "request_state_sync":
            # 手机端主动请求状态同步（如页面切换、重连后）
            self._sync_state_to_phone()
            # 同时发送当前模式
            mode = "simu" if self._simu is not None else "real"
            if self._ws_server:
                self._ws_server.send({"type": "mode", "value": mode})

        elif cmd_type == "calibration_start":
            self._calibration_mode = True
            self._teleop_axis_mapping = {}
            self._teleop_rot_mapping = {}
            self._sync_state_to_phone()  # 立即同步标定状态
            self._start_calibration_mode()
            self._log("进入标定模式，移动手机观察机械臂方向", "INFO")

        elif cmd_type == "calibration_end":
            self._log("=== 退出标定模式 (开始) ===", "INFO")
            self._log(f"退出标定前状态: _calibration_mode={self._calibration_mode}, "
                     f"_simu={self._simu is not None}, "
                     f"_simu_manager={self._simu_manager is not None}", "INFO")

            self._calibration_mode = False
            # 立即同步状态，让手机端知道已退出标定
            self._sync_state_to_phone()

            # 在独立线程中执行耗时的清理操作，避免阻塞 WebSocket 事件循环导致心跳超时断连
            def _exit_calibration_thread():
                try:
                    # 1. 必须先停止控制循环（释放对 _lock 的占用），再停止 Publisher
                    # 否则 Publisher.stop() 的 join 会等 Publisher 线程退出，
                    # 而 Publisher 线程在等 _lock，_lock 被控制循环持有 → 死锁
                    if self._teleop_controller:
                        self._log("标定模式: 调用end_calibration()停止控制循环...", "INFO")
                        self._teleop_controller.end_calibration()
                        self._log("标定模式: 控制循环已停止", "INFO")

                    # 2. 标记 Publisher 停止但不 join，让 Publisher 线程自然退出
                    #    控制循环已停止，_lock 已释放，Publisher 能正常获取锁并退出
                    self._set_publisher_auto_step(True)
                    if self._simu_manager and self._simu_manager._publisher:
                        self._simu_manager._publisher._fps = 20

                    # 3. 移动到初始位（控制循环已停，这里同步执行）
                    if self._teleop_controller and self._simu is not None:
                        try:
                            self._teleop_controller.move_to_initial()
                            self._log("标定模式: 已回到初始位", "INFO")
                        except Exception as e:
                            self._log(f"标定模式: move_to_initial 异常: {e}", "WARNING")

                    # 4. 停止仿真（此时控制循环已停，_lock 无竞争，Publisher 也能正常退出）
                    if self._simu_manager is not None:
                        try:
                            if self._simu_manager.is_running:
                                self._log("标定模式: 正在调用 stop_simulation()...", "INFO")
                                self._simu_manager.stop_simulation()
                                self._log("标定模式: stop_simulation() 完成", "INFO")
                        except Exception as e:
                            tb = traceback.format_exc()
                            self._log(f"标定模式: 停止仿真异常: {e}\n{tb}", "ERROR")
                        finally:
                            self._simu = None
                            self._log("标定模式: _simu 已设为 None", "INFO")
                    else:
                        self._simu = None

                    # 5. 应用映射
                    rot_map = self._teleop_rot_mapping
                    if self._teleop_controller:
                        self._teleop_controller.set_axis_mapping(
                            self._teleop_axis_mapping,
                            rot_mapping=rot_map
                        )
                        pos_count = len(self._teleop_axis_mapping)
                        rot_count = len(rot_map) if rot_map else 0
                        self._log(f"标定映射已应用: 位置={pos_count}/6, 姿态={rot_count}/6", "SUCCESS")

                    self._sync_state_to_phone()
                    self._log("=== 退出标定模式 (完成) ===", "INFO")
                except Exception as e:
                    tb = traceback.format_exc()
                    self._log(f"退出标定模式异常: {e}\n{tb}", "ERROR")

            threading.Thread(target=_exit_calibration_thread, daemon=True).start()

    def _on_teleop_calib_obs(self, msg: dict):
        """处理交互式智能标定消息

        消息类型:
        - calib_obs_start: 开始观察 {"mode": "position"/"orientation"}
        - calib_obs_stop: 停止观察，返回机械臂实际运动方向
        - calib_apply_mapping: 应用映射 {"phone_dir": "z+", "expected_dir": "x+", "mode": "position"}
        - calib_reset_position: 重置机械臂到标定起始位置
        """
        if not self._teleop_controller:
            self._log("[标定] 控制器未初始化", "ERROR")
            return

        msg_type = msg.get("type", "")
        self._log(f"[交互式标定] 收到: {msg_type}", "INFO")

        if msg_type == "calib_obs_start":
            mode = msg.get("mode", "position")
            self._teleop_controller.start_calibration_observation(mode)
            self._log(f"[标定] 开始观察 {mode}", "INFO")

        elif msg_type == "calib_obs_stop":
            result = self._teleop_controller.stop_calibration_observation()
            if self._ws_server:
                self._ws_server.send({
                    "type": "calib_obs_result",
                    **result
                })
            self._log(f"[标定] 观察结果: {result}", "INFO")

        elif msg_type == "calib_apply_mapping":
            phone_dir = msg.get("phone_dir", "")
            expected_dir = msg.get("expected_dir", "")
            mode = msg.get("mode", "position")
            success = self._teleop_controller.build_mapping_from_observation(
                phone_dir, expected_dir, mode
            )
            if success:
                # 同步到 main_qt 的映射字典
                mapping = self._teleop_controller.get_current_mapping()
                self._teleop_axis_mapping = mapping["position_mapping"]
                self._teleop_rot_mapping = mapping["rotation_mapping"]
                # 通知手机端
                if self._ws_server:
                    self._ws_server.send({
                        "type": "calib_mapping_updated",
                        "success": True,
                        "position_mapping": self._teleop_axis_mapping,
                        "rotation_mapping": self._teleop_rot_mapping,
                    })
                self._log(f"[标定] 映射已更新: phone_{phone_dir} -> robot_{expected_dir}", "INFO")
            else:
                if self._ws_server:
                    self._ws_server.send({
                        "type": "calib_mapping_updated",
                        "success": False,
                        "error": "映射建立失败",
                    })
                self._log(f"[标定] 映射建立失败", "ERROR")

        elif msg_type == "calib_reset_position":
            self._teleop_controller.move_to_initial()
            self._teleop_controller.align()
            self._log("[标定] 已重置到标定起始位置", "INFO")

    def _on_teleop_wrist_pivot(self, msg_type: str, msg: dict):
        """处理 wrist-pivot 校准消息(仿 vr-teleop-kit)

        流程:
          1. 手机端检测到双手 grip → 发 wrist_pivot_start
          2. PC 端 5s 期间记录 controller._target(arm base 帧)样本
          3. 5s 结束后 LS 求解 offset
          4. 回 wrist_pivot_result 给手机
        """
        if not self._teleop_controller:
            self._log("[WristPivot] 控制器未初始化", "ERROR")
            return

        if msg_type == "wrist_pivot_start":
            duration_s = float(msg.get("duration_s", 5.0))

            def _on_result(result: dict):
                if result.get("ok"):
                    self._log(
                        f"[WristPivot] 校准成功 offset={result['o']}, "
                        f"rms={result['rms'] * 1000:.1f} mm, n={result['n']}",
                        "SUCCESS",
                    )
                else:
                    self._log(
                        f"[WristPivot] 校准失败: {result.get('reason')}",
                        "WARNING",
                    )
                # 推回手机端(供 UI 显示)
                if self._ws_server:
                    try:
                        self._ws_server.send({
                            "type": "wrist_pivot_result",
                            "ok": result.get("ok", False),
                            "offset": result.get("o", [0, 0, 0]),
                            "rms": result.get("rms", 0.0),
                            "n": result.get("n", 0),
                            "reason": result.get("reason"),
                        })
                    except Exception as e:
                        logger.debug(f"send wrist_pivot_result: {e}", exc_info=True)

            ok = self._teleop_controller.start_wrist_pivot_calibration(
                on_result=_on_result,
                duration_s=duration_s,
            )
            if ok:
                self._log(
                    f"[WristPivot] 开始校准,时长 {duration_s}s,"
                    f"请双手握紧手柄并自由转动腕部",
                    "INFO",
                )
            else:
                self._log("[WristPivot] 启动失败(未对齐?)", "ERROR")

        elif msg_type == "wrist_pivot_stop":
            self._teleop_controller.stop_wrist_pivot_calibration()
            self._log("[WristPivot] 已停止校准", "INFO")

    def _on_teleop_align_direction(self, axis: str, phone_axis: str):
        """处理方向标定结果

        Args:
            axis: 机械臂方向 (如 "x+", "y-", "z+", "rx+", "ry-", "rz+")
            phone_axis: 手机方向 (如 "x+", "y-", "z+")
        """
        # 区分位置映射和姿态映射
        is_rotation = axis.startswith('r')  # "rx+", "ry-", "rz+" 等

        if is_rotation:
            # 姿态映射：存储到单独的字典
            # 统一key格式：去掉前缀'r'，使key为 "x+"/"y-"/"z+" 等
            # 这样与 DEFAULT_ROT_MAPPING 和 apply_delta 中的格式一致
            normalized_axis = axis[1:]  # "rx+" -> "x+", "ry-" -> "y-"
            self._teleop_rot_mapping[normalized_axis] = phone_axis
            self._log(f"姿态标定: {axis}(存储为{normalized_axis}) -> {phone_axis} "
                     f"(当前已标定 {len(self._teleop_rot_mapping)}/6)", "INFO")

            # 实时更新控制器姿态映射
            if self._teleop_controller and len(self._teleop_rot_mapping) > 0:
                self._teleop_controller.set_axis_mapping(
                    self._teleop_axis_mapping,
                    rot_mapping=self._teleop_rot_mapping
                )
        else:
            # 位置映射
            self._teleop_axis_mapping[axis] = phone_axis
            self._log(f"位置标定: {axis} -> {phone_axis} (当前已标定 {len(self._teleop_axis_mapping)}/6)", "INFO")

            # 实时更新控制器位置映射
            if self._teleop_controller and len(self._teleop_axis_mapping) > 0:
                rot_map = self._teleop_rot_mapping
                self._teleop_controller.set_axis_mapping(
                    self._teleop_axis_mapping,
                    rot_mapping=rot_map
                )

    def _start_calibration_mode(self):
        """启动标定模式：仿真=启动仿真不录制，实机=控制实机

        标定模式不进行任何数据录制，仅让用户移动手机观察机械臂方向，
        以确认轴映射是否正确。
        """
        def _calibration_thread():
            try:
                self._log("标定模式: 开始初始化...", "INFO")

                # 检查可用接口
                has_simu = self._simu is not None or (self._simu_base_xml_path and self._simu_manager)
                has_real = self._real_connected

                self._log(f"标定模式: has_simu={has_simu}, _simu exists={self._simu is not None}, "
                         f"_simu_base_xml_path={bool(self._simu_base_xml_path)}, "
                         f"_simu_manager exists={self._simu_manager is not None}", "INFO")
                self._log(f"标定模式: has_real={has_real}, _teleop_controller exists={self._teleop_controller is not None}", "INFO")

                if has_simu:
                    # 仿真模式：启动一个简单仿真场景（无物体、不录制）
                    if self._simu is None and self._simu_manager:
                        calib_config = {
                            'base_scene_xml': self._simu_base_xml_path,
                            'camera_names': self.get_simu_camera_names() if hasattr(self, 'get_simu_camera_names') else ['top'],
                            'use_ik': True,
                            'fps': 20,
                            'show_viewer': False,  # 不启动被动查看器，避免与GLFW查看器冲突
                        }
                        if self._sim_initial_joints_deg is not None:
                            calib_config['initial_joints_deg'] = self._sim_initial_joints_deg.tolist()
                            self._log(f"标定模式: 初始关节={calib_config['initial_joints_deg']}", "INFO")
                        else:
                            self._log("标定模式: 未配置初始关节，将使用全0姿态", "WARNING")
                        calib_config['initial_gripper'] = getattr(self, '_sim_initial_gripper', 0.0)

                        self._log(f"标定模式: 正在启动仿真, xml={calib_config['base_scene_xml']}", "INFO")
                        if not self._simu_manager.start_task_simulation(calib_config):
                            self._log("标定模式: 仿真启动失败", "ERROR")
                            self._calibration_mode = False
                            self._sync_state_to_phone()
                            return

                        self._simu = self._simu_manager.simu
                        self._log(f"标定模式: 仿真已启动, simu={self._simu is not None}", "INFO")

                        if self._simu is not None:
                            joints = self._simu.get_joint_state()  # 度数
                            pos = self._simu.get_tcp_position()
                            self._log(f"标定模式: 当前关节(deg)={joints.tolist() if joints is not None else 'None'}", "INFO")
                            self._log(f"标定模式: 当前位置={pos.tolist() if pos is not None else 'None'}", "INFO")

                    # 启动 GLFW 查看器（遥操作模式下强制启动，非遥操作模式按配置）
                    if self._simu and (self._teleop_mode or self._show_simu_viewer):
                        self._log("标定模式: 启动仿真查看器...", "INFO")
                        viewer_started = self._simu.start_glfw_viewer(width=1200, height=900, title="Calibration Mode")
                        if viewer_started:
                            self._log("标定模式: GLFW 查看器已启动", "SUCCESS")
                        else:
                            self._log("标定模式: GLFW 查看器启动失败", "WARNING")

                    # 更新 TeleopController 的仿真接口
                    if self._teleop_controller:
                        self._log("标定模式: 配置 TeleopController...", "INFO")
                        self._teleop_controller.set_simu_interface(self._simu)
                        # 禁用 Publisher 自动推进，由控制循环负责 step()
                        self._set_publisher_auto_step(False)
                        # 标定模式降低 publisher 频率到5Hz，减少相机渲染锁竞争，提高控制循环响应速度
                        if self._simu_manager and self._simu_manager._publisher:
                            self._simu_manager._publisher._fps = 5
                            self._log("标定模式: Publisher 频率降为5Hz减少锁竞争", "INFO")
                        # 调用start_calibration启动标定模式（自动对齐+启动控制循环）
                        self._teleop_controller.start_calibration()
                        self._log(f"标定模式(仿真): aligned={self._teleop_controller.is_aligned}, "
                                 f"control_running={self._teleop_controller.is_control_running}, "
                                 f"calibrating={self._calibration_mode}", "INFO")

                elif has_real:
                    # 实机模式：直接控制实机
                    self._log("标定模式: 使用实机模式", "INFO")
                    if self._teleop_controller:
                        self._teleop_controller.set_simu_interface(None)
                        # 调用start_calibration启动标定模式
                        self._teleop_controller.start_calibration()
                        self._log("标定模式(实机): 已启动，移动手机观察机械臂", "SUCCESS")
                else:
                    self._log("标定模式: 无仿真或实机可用", "ERROR")
                    self._calibration_mode = False

                # 标定模式启动后同步状态
                self._sync_state_to_phone()
                self._log("标定模式: 初始化完成", "INFO")

            except Exception as e:
                tb = traceback.format_exc()
                self._log(f"标定模式启动失败: {e}\n{tb}", "ERROR")
                self._calibration_mode = False
                self._sync_state_to_phone()

        threading.Thread(target=_calibration_thread, daemon=True).start()

    def is_teleop_mode(self) -> bool:
        """是否为遥操作模式"""
        return self._teleop_mode

    def is_phone_connected(self) -> bool:
        """手机是否已连接"""
        return self._ws_server is not None and self._ws_server.is_connected

    def _sync_state_to_phone(self):
        """将当前 PC 端状态同步给手机端，避免双端操作冲突

        统一状态定义:
        - idle: 未开始收集
        - collection_started: 已开启收集，等待开启任务/标定
        - preparing: 正在准备场景（创建仿真、加载模型）
        - calibrating: 标定模式中（仿真已启动，可移动手机调试）
        - episode_running: 任务执行中（仿真已启动、已对齐、控制循环运行，可遥操作）
        """
        if not self._ws_server or not self._ws_server.is_connected:
            return

        # 标定模式优先
        if self._calibration_mode:
            pc_state = "calibrating"
        elif not self._running:
            pc_state = "idle"
        elif self._task_in_progress:
            pc_state = "episode_running"
        elif self._waiting_for_next_task:
            pc_state = "preparing"
        else:
            pc_state = "collection_started"

        # 获取遥操作控制器状态
        ctrl = self._teleop_controller
        simu_ready = (self._simu is not None)
        aligned = ctrl.is_aligned if ctrl else False
        control_running = ctrl.is_control_running if ctrl else False
        can_control = simu_ready and aligned and control_running and not (ctrl.is_emergency_stopped if ctrl else True)

        self._ws_server.send({
            "type": "state_sync",
            "state": pc_state,
            "episode": self._current_episode,
            "task_index": self._current_task_index,
            "task_in_progress": self._task_in_progress,
            "collecting": self._running,
            "calibrating": self._calibration_mode,
            "simu_ready": simu_ready,
            "aligned": aligned,
            "control_running": control_running,
            "can_control": can_control,
        })

    def _start_periodic_state_sync(self):
        """启动定期状态同步（每2秒），防止网络抖动导致双端状态不一致"""
        self._stop_periodic_state_sync()
        self._state_sync_timer = threading.Timer(2.0, self._periodic_state_sync_tick)
        self._state_sync_timer.daemon = True
        self._state_sync_timer.start()

    def _periodic_state_sync_tick(self):
        """定期同步一次状态，然后重新设置定时器"""
        if self._ws_server and self._ws_server.is_connected:
            self._sync_state_to_phone()
            self._state_sync_timer = threading.Timer(2.0, self._periodic_state_sync_tick)
            self._state_sync_timer.daemon = True
            self._state_sync_timer.start()

    def _stop_periodic_state_sync(self):
        """停止定期状态同步"""
        if self._state_sync_timer is not None:
            self._state_sync_timer.cancel()
            self._state_sync_timer = None

    def get_local_ip(self) -> str:
        """获取本机局域网 IP 地址"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    def get_teleop_port(self) -> int:
        """获取遥操作 WebSocket 端口"""
        if self._ws_server is not None:
            return self._ws_server._port
        return 8765

    def _redirect_tqdm_to_log(self):
        """将 tqdm 进度条（LeRobot 内部使用）重定向到 Qt 日志面板"""
        from tqdm import tqdm as _tqdm_original
        import tqdm as _tqdm_module

        # 闭包捕获 self，避免全局变量
        _self = self

        class _TqdmToLog(_tqdm_original):
            """自定义 tqdm：进度更新时通过 _log 输出到 Qt 面板"""
            def __init__(self, *args, **kwargs):
                kwargs.setdefault('file', None)  # 禁用默认文件输出
                super().__init__(*args, **kwargs)
                self._last_log_time = 0

            def display(self, msg=None, pos=None):
                """覆盖显示方法：将进度信息转发到日志面板"""
                import time
                now = time.time()
                # 限流：最多每 0.3s 输出一次，避免刷屏
                if now - self._last_log_time < 0.3 and not (msg and '100%' in str(msg)):
                    return
                self._last_log_time = now

                try:
                    # 构造简短的进度描述
                    desc = self.desc or "Progress"
                    pct = self.n / self.total * 100 if self.total else 0
                    info = f"{desc}: {pct:.0f}% ({self.n}/{self.total})"
                    if len(info) > 100:
                        info = info[:97] + "..."
                    _self._log(info, "INFO")
                except Exception:
                    logger.debug("Frame callback error", exc_info=True)

        # 仅替换当前模块引用，不影响其他库的 tqdm 使用
        import sys
        _tqdm_module.tqdm = _TqdmToLog
        # 保存原始 tqdm 以便恢复
        self._original_tqdm = _tqdm_original
    
    def _log(self, message: str, level: str = "INFO"):
        """输出日志"""
        log_level = getattr(logging, level, logging.INFO)
        logger.log(log_level, message)
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

    def is_real_connected(self) -> bool:
        """检查真实机器人是否已连接"""
        return self._real_connected
    
    def get_real_camera_names(self) -> list:
        """获取真实相机名称列表"""
        if not self._real_connected:
            return []
        real_camera_config = self._config.get('cameras', {}).get('real', {})
        return list(real_camera_config.keys()) if real_camera_config else []
    
    def get_simu_camera_names(self) -> list:
        """获取仿真相机名称列表"""
        return self._simu_camera_names

    def get_real_camera_config(self) -> dict:
        """获取真实相机配置"""
        if not self._real_connected:
            return {}
        return self._config.get('cameras', {}).get('real', {}) or {}

    def get_simu_camera_config(self) -> dict:
        """获取仿真相机配置（仅返回名称列表的兼容结构）"""
        return {name: {} for name in self._simu_camera_names}

    def _apply_sim_initial_joints(self):
        if self._simu is None or self._sim_initial_joints_deg is None:
            return
        ok = self._simu.set_joint_positions(self._sim_initial_joints_deg, gripper=self._sim_initial_gripper)
        if ok:
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

    def _infer_object_body_name(self, object_name: str, task_config: dict) -> str:
        """推断仿真物体 body name"""
        object_cfg = self._object_library.get(object_name, {}) if isinstance(self._object_library, dict) else {}
        object_model_xml = object_cfg.get('model_xml_path', '') if isinstance(object_cfg, dict) else ''
        cfg_body_name = object_cfg.get('body_name', '') if isinstance(object_cfg, dict) else ''
        inferred = self._infer_object_body_name_from_xml(object_model_xml, fallback='cube')
        return task_config.get('sim_body_name', cfg_body_name or inferred)

    def _infer_plate_body_name(self) -> str:
        """推断仿真盘子 body name"""
        plate_target_cfg = self._config.get('simulation', {}).get('plate_target', {}) if isinstance(self._config, dict) else {}
        if not plate_target_cfg:
            plate_target_cfg = self._config.get('plate_target', {}) if isinstance(self._config, dict) else {}
        return plate_target_cfg.get('body_name', 'body_obj_plate') if isinstance(plate_target_cfg, dict) else 'body_obj_plate'

    def _get_object_model_xml(self, object_name: str) -> str:
        """获取物体模型 XML 路径"""
        object_cfg = self._object_library.get(object_name, {}) if isinstance(self._object_library, dict) else {}
        return object_cfg.get('model_xml_path', '') if isinstance(object_cfg, dict) else ''

    def _get_plate_model_xml(self) -> str:
        """获取盘子模型 XML 路径"""
        plate_target_cfg = self._config.get('simulation', {}).get('plate_target', {}) if isinstance(self._config, dict) else {}
        if not plate_target_cfg:
            plate_target_cfg = self._config.get('plate_target', {}) if isinstance(self._config, dict) else {}
        return plate_target_cfg.get('model_xml_path', '') if isinstance(plate_target_cfg, dict) else ''

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
        if self._grasp_executor is None:
            self._log('系统未初始化，无法进入标定模式', 'ERROR')
            return False

        # 如果仿真尚未启动，先启动基础仿真
        if self._simu is None:
            self._log('prepare_object_tuning: 仿真未启动，先启动基础仿真', 'INFO')
            if self._simu_manager is None or not self._simu_base_xml_path:
                self._log('prepare_object_tuning: 无法启动仿真，缺少 SimuManager 或 xml_path', 'ERROR')
                return False

            calib_config = {
                'base_scene_xml': self._simu_base_xml_path,
                'camera_names': self.get_simu_camera_names() if hasattr(self, 'get_simu_camera_names') else ['top'],
                'use_ik': True,
                'fps': 20,
                'show_viewer': False,
            }
            if self._sim_initial_joints_deg is not None:
                calib_config['initial_joints_deg'] = self._sim_initial_joints_deg.tolist()
            calib_config['initial_gripper'] = getattr(self, '_sim_initial_gripper', 0.0)

            if not self._simu_manager.start_task_simulation(calib_config):
                self._log('prepare_object_tuning: 仿真启动失败', 'ERROR')
                return False
            self._simu = self._simu_manager.simu
            self._log(f'prepare_object_tuning: 仿真已启动, simu={self._simu is not None}', 'INFO')

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
            pre_ori = self._grasp_executor.euler_xyz_deg_to_rotmat(pre_pose[3:6]) if len(pre_pose) >= 6 else None

            self._log(f'标定模式: 先执行IK到目标上方，task={task_id_resolved}, object={obj_name}', 'INFO')
            ok = self._simu.move_to_cartesian(pre_pos, orientation=pre_ori, duration=2.0, steps=100)
            if not ok:
                self._log('IK到预抓取位失败，可直接手动拖动关节继续标定', 'WARNING')

            self._simu.step(30)
            return True
        except Exception as e:
            self._log(f'prepare_object_tuning 失败: {e}', 'ERROR')
            return False

    def set_tuning_joints_deg(self, joints_deg: np.ndarray) -> bool:
        if self._simu is None:
            return False
        ok = self._simu.set_joint_target(np.asarray(joints_deg[:6], dtype=float))
        if ok:
            self._simu.step(10)
        return ok

    def set_tuning_gripper(self, gripper: float) -> bool:
        if self._simu is None:
            return False
        ok = self._simu.set_gripper(float(np.clip(gripper, 0.0, 1.0)))
        if ok:
            self._simu.step(10)
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
                # 限制欧拉角范围，避免姿态突变
                euler = np.asarray(pose[3:6], dtype=float)
                # 将欧拉角归一化到 [-180, 180]
                euler = np.mod(euler + 180, 360) - 180
                target_ori = self._grasp_executor.euler_xyz_deg_to_rotmat(euler)
            
            success = self._simu.move_to_cartesian(
                target_pos,
                orientation=target_ori,
                duration=0.15,  # 缩短时间，加快响应
                steps=8,       # 减少步数，加速 IK 求解
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
                    logger.debug("Viewer check failed", exc_info=True)
            if self._real_connected:
                # 实机已连接：启动被动查看器
                if hasattr(self._simu, 'start_viewer'):
                    self._simu.start_viewer()
                    return True
            else:
                # 离线模式：启动 GLFW 查看器
                if self._show_simu_viewer and hasattr(self._simu, 'start_glfw_viewer'):
                    return self._simu.start_glfw_viewer()
            return False
        except Exception:
            return False

    def get_tuning_state(self) -> Dict[str, Any]:
        if self._simu is None:
            return {}

        try:
            # 不在这里调用 step()，避免干扰正在进行的 IK 运动
            # 只同步控制状态（如果需要）
            if hasattr(self._simu, 'sync_control_to_current_state'):
                try:
                    self._simu.sync_control_to_current_state()
                except Exception:
                    logger.debug("sync_control_to_current_state failed", exc_info=True)
        except Exception:
            logger.debug("Simu state sync failed", exc_info=True)

        joints_deg = self._simu.get_joint_state()  # 度数
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
            try:
                body_pos = np.asarray(self._simu.get_object_position(self._tuning_object_body_name), dtype=float)
                if float(np.linalg.norm(body_pos)) > 1e-9:
                    obj_pos = body_pos
            except Exception:
                logger.debug("Failed to get object position", exc_info=True)

        # 计算抓取偏移：TCP 相对于物体中心的偏移
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

    def start_collection(self, start_task_index: int = 0, episode: int = None):
        """开始数据收集 - 每个任务是一个独立的 episode"""
        if self._running:
            self._log("数据收集已在进行中", "WARNING")
            return
        self._current_task_index = start_task_index
        if episode is not None:
            self._current_episode = episode
        self._running = True
        self._paused = False
        self._waiting_for_next_task = False

        # 开始收集前，机械臂先复位到初始关节位置（后台线程）
        if self._real_connected and self._grasp_executor is not None:
            self._log("机械臂复位到初始关节位置...", "INFO")
            _gs = self._grasp_executor
            _ls = self._log
            def _reset():
                _gs.move_to_initial_joints()
                _ls("机械臂已复位", "SUCCESS")
            threading.Thread(target=_reset, daemon=True).start()
        
        self._log(f"开始收集，Episode 编号: {self._current_episode}，任务索引: {start_task_index}", "INFO")
        
        # 启动数据收集器（不预启动 episode，每个任务单独 start_episode）
        if self._real_data_collector is not None:
            self._real_data_collector.start_collection()
        if self._simu_data_collector is not None:
            self._simu_data_collector.start_collection()
        self._log(f"数据收集已启动，当前 Episode 编号: {self._current_episode}", "SUCCESS")
        
        self._update_status("等待执行任务", "#ffa500")
        self._sync_state_to_phone()
    
    def execute_next_task(self):
        """执行下一个任务（由 GUI 按钮触发）"""
        if not self._running:
            self._log("数据收集未启动", "WARNING")
            return

        if self._waiting_for_next_task:
            self._log("任务准备中，请稍候", "WARNING")
            return

        if self._task_in_progress:
            self._log('当前任务尚未点击"抓取任务完毕"', "WARNING")
            return

        tasks = self._config.get('tasks', {})
        task_list = list(tasks.items())

        self._log(f"当前任务索引: {self._current_task_index}, 总任务数: {len(task_list)}", "INFO")

        if self._current_task_index >= len(task_list):
            self._log("所有任务已完成，请停止收集", "SUCCESS")
            return

        self._waiting_for_next_task = True
        self._sync_state_to_phone()  # 立即同步 preparing 状态
        task_thread = threading.Thread(target=self._execute_current_task, daemon=True)
        task_thread.start()

    def _execute_current_task(self):
        """准备并自动执行当前任务"""
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
                if self._teleop_mode:
                    self._log('任务已准备就绪，请移动手机开始遥操作', "INFO")
                    self._update_status("遥操作采集中", "#44ff44")
                    # 遥操作时降低 Publisher 频率到10Hz，减少相机渲染锁竞争，提高控制循环响应速度
                    if self._simu_manager and self._simu_manager._publisher:
                        self._simu_manager._publisher._fps = 10
                        self._log("遥操作模式: Publisher 频率降为10Hz减少锁竞争", "INFO")
                else:
                    self._log('任务已启动，自动执行中...', "INFO")
                    self._update_status("自动执行中", "#44ff44")
            else:
                self._task_in_progress = False
                self._log("任务准备失败", "WARNING")
                self._update_status("准备失败", "#ff4444")
            self._sync_state_to_phone()

        except Exception as e:
            self._log(f"任务错误: {e}", "ERROR")
            logger.error(f"Failed to connect to real robot: {e}")
            logger.debug(traceback.format_exc())
            self._waiting_for_next_task = False
            self._task_in_progress = False
            self._update_status("错误", "#ff0000")
            self._sync_state_to_phone()

    def retry_current_task(self):
        """重做当前任务（仅任务执行中可用）

        丢弃当前数据并重新执行，避免文件写入后再删除的复杂性。
        """
        if not self._running:
            self._log("数据收集未启动", "WARNING")
            return

        if self._waiting_for_next_task:
            self._log("任务准备中，请稍后再重做", "WARNING")
            return

        if not self._task_in_progress:
            self._log("当前没有进行中的任务，无法重做", "WARNING")
            return

        self._log("丢弃当前数据并重新执行...", "WARNING")
        self._update_status("重做任务中", "#ffa500")

        # 停止数据记录
        if self._real_data_collector is not None:
            self._real_data_collector.stop_recording()
        if self._simu_data_collector is not None:
            self._simu_data_collector.stop_recording()

        # 丢弃当前任务数据
        if self._real_data_collector is not None:
            self._real_data_collector.discard_current_task()
        if self._simu_data_collector is not None:
            self._simu_data_collector.discard_current_task()
        self._log("当前任务数据已丢弃", "SUCCESS")

        # 关闭仿真窗口（仅仿真模式需要重建）
        if self._simu is not None and self._simu_manager.is_running:
            try:
                self._log("正在关闭仿真窗口...", "INFO")
                self._simu_manager.stop_simulation()
                self._simu = None
                self._log("仿真窗口已关闭", "SUCCESS")
            except Exception as e:
                self._log(f"关闭仿真失败: {e}", "WARNING")

        # 重置任务状态
        self._task_in_progress = False
        self._active_task_info = {}
        self._sync_state_to_phone()

        # 重新执行当前任务
        tasks = self._config.get('tasks', {})
        task_list = list(tasks.items())
        if self._current_task_index >= len(task_list):
            self._current_task_index = max(0, len(task_list) - 1)

        task_id, task_config = task_list[self._current_task_index]
        task_name = task_config.get('description', task_id)
        self._log(f"重新执行任务 {self._current_task_index + 1}/{len(task_list)}: {task_name} (Episode {self._current_episode})", "WARNING")

        self._waiting_for_next_task = True
        task_thread = threading.Thread(target=self._execute_current_task, daemon=True)
        task_thread.start()

    def finish_current_task(self):
        """确认当前任务完成（防重入，自动线程和手动按钮均安全）"""
        if not self._finish_lock.acquire(blocking=False):
            self._log("任务正在提交中，请稍候", "WARNING")
            return

        try:
            if not self._running:
                self._log("数据收集未启动", "WARNING")
                return

            if not self._task_in_progress:
                self._log("当前没有进行中的任务", "WARNING")
                return

            self._task_in_progress = False  # 先标记，防止状态混乱
            self._update_status("等待数据处理完成", "#ffa500")
            self._log("正在停止数据记录...", "INFO")
            if self._real_data_collector is not None:
                self._real_data_collector.stop_recording()
            if self._simu_data_collector is not None:
                self._simu_data_collector.stop_recording()
            self._log("数据记录已停止，正在保存 Episode...", "INFO")

            success = self._evaluate_task_success()
            self._log(f"任务完成: {'成功' if success else '失败'}", "SUCCESS" if success else "WARNING")

            if self._real_data_collector is not None:
                self._real_data_collector.end_episode(self._current_episode, success)
            if self._simu_data_collector is not None:
                self._simu_data_collector.end_episode(self._current_episode, success)
            self._log(f"Episode {self._current_episode} 数据已保存完毕", "SUCCESS")
            self._current_episode += 1

            progress_collector = self._real_data_collector if self._real_data_collector is not None else self._simu_data_collector
            if progress_collector is not None:
                progress_collector.save_progress()

            # 机械臂复位到初始位置（后台线程，不阻塞）
            if self._real_connected and self._grasp_executor is not None:
                self._log("机械臂复位到初始关节位置...", "INFO")
                _gs = self._grasp_executor
                _ls = self._log
                def _reset():
                    _gs.move_to_initial_joints()
                    _ls("机械臂已复位", "SUCCESS")
                threading.Thread(target=_reset, daemon=True).start()

            # 任务结束：仅在仿真模式下重建仿真
            if self._simu is not None and self._simu_manager.is_running:
                try:
                    self._log("正在关闭仿真窗口...", "INFO")
                    self._simu_manager.stop_simulation()
                    self._simu = None
                    self._log("仿真窗口已关闭", "SUCCESS")
                except Exception as e:
                    self._log(f"关闭仿真失败: {e}", "WARNING")

            self._active_task_info = {}
            self._current_task_index += 1

            tasks = self._config.get('tasks', {})
            task_total = len(list(tasks.items()))
            if self._current_task_index >= task_total:
                self._log(f"所有任务已完成！共收集 {self._current_episode - 1} 个 episode", "SUCCESS")
                self._update_status("任务完成", "#00aa00")
            else:
                self._log("等待下一个任务...", "INFO")
                self._update_status("等待执行任务", "#ffa500")
            self._sync_state_to_phone()
        finally:
            self._finish_lock.release()
    
    def stop(self):
        """停止数据收集"""
        self._log("正在停止数据收集...", "WARNING")
        self._running = False
        self._paused = False
        
        # 如果当前有任务正在进行，停止记录并丢弃未完成的数据
        if self._task_in_progress:
            if self._real_data_collector is not None:
                self._real_data_collector.stop_recording()
                self._real_data_collector.discard_current_task()
            if self._simu_data_collector is not None:
                self._simu_data_collector.stop_recording()
                self._simu_data_collector.discard_current_task()
            self._task_in_progress = False
            self._active_task_info = {}
            self._log("当前任务数据已丢弃（任务未完成）", "WARNING")
        
        if self._real_data_collector:
            self._real_data_collector.stop_collection()
        
        if self._simu_data_collector:
            self._simu_data_collector.stop_collection()
        
        # 保存进度
        progress_collector = self._real_data_collector if self._real_data_collector is not None else self._simu_data_collector
        if progress_collector is not None:
            progress_collector.save_progress()
        
        # 停止仿真（仅仿真模式）
        if self._simu is not None and self._simu_manager.is_running:
            try:
                self._log("正在关闭仿真窗口...", "INFO")
                self._simu_manager.stop_simulation()
                self._simu = None
                self._log("仿真窗口已关闭", "SUCCESS")
            except Exception as e:
                self._log(f"关闭仿真失败: {e}", "WARNING")
        
        self._log(f"数据收集已停止 (共 {self._current_episode} 个 episode)", "WARNING")
        self._update_status("已停止", "#ff4444")
        self._sync_state_to_phone()
    
    def _execute_task(self, task_id: str, task_config: dict) -> bool:
        """准备并执行单个任务
        
        遥操作模式流程（完全独立，无自动执行）：
        1. 检查手机连接状态
        2. 停止旧仿真，创建新仿真场景
        3. 强制启动仿真查看器
        4. 配置TeleopController
        5. 执行对齐
        6. 启动数据录制
        7. 设置collecting_active=True，等待手机遥操作
        """
        task_name = task_config.get('description', task_id)
        object_name = task_config.get('object_name', 'cube')
        
        self._log(f"═══════════════════════════════════════════════════════", "INFO")
        self._log(f"开始执行任务: {task_name}", "INFO")
        self._log(f"遥操作模式: {self._teleop_mode}", "INFO")
        
        phone_connected = False
        if self._ws_server is not None:
            try:
                # is_connected 是 @property，不是方法，不能用 is_connected()
                phone_connected = self._ws_server.is_connected
            except Exception:
                phone_connected = False
        self._log(f"手机连接状态: {phone_connected}", "INFO")
        
        try:
            # ==============================================================
            # 遥操作模式：优先处理，提前返回，绝不执行后面的自动执行逻辑
            # ==============================================================
            if self._teleop_mode:
                self._log("【遥操作模式】开始准备任务场景...", "INFO")
                
                if not phone_connected:
                    self._log("遥操作模式错误：手机未连接，无法开始任务", "ERROR")
                    return False
                
                if not self._simu_base_xml_path or not self._simu_manager:
                    self._log("遥操作模式错误：缺少仿真配置或SimuManager", "ERROR")
                    return False
                
                if self._teleop_controller is None:
                    self._log("遥操作模式错误：TeleopController未初始化", "ERROR")
                    return False
                
                plate_pos_config = task_config.get('plate_position', [0.35, -0.15, 0.44])
                plate_pos = self._resolve_position(plate_pos_config)
                min_dist = self._config.get('workspace', {}).get('min_object_plate_distance', 0.15)
                object_pos_config = task_config.get('object_position', [0.35, 0.15, 0.44])
                object_pos = self._resolve_position(object_pos_config, existing_position=plate_pos, min_distance=min_dist)
                
                self._grasp_executor.set_object_type(object_name)
                self._log(f"物体类型: {object_name}", "INFO")
                
                self._grasp_executor.set_target(
                    object_position=object_pos.tolist(),
                    place_position=plate_pos.tolist(),
                )
                
                self._log("清理之前的仿真...", "INFO")
                if self._simu is not None:
                    try:
                        if self._simu_manager.is_running:
                            self._simu_manager.stop_simulation()
                    except Exception:
                        pass
                    self._simu = None
                    self._log("旧仿真已停止", "INFO")
                
                object_body_name = self._infer_object_body_name(object_name, task_config)
                plate_body_name = self._infer_plate_body_name()
                
                self._log(f"创建任务仿真场景: object={object_name}", "INFO")
                task_simu_config = {
                    'base_scene_xml': self._simu_base_xml_path,
                    'object_model_xml': self._get_object_model_xml(object_name),
                    'object_body_name': object_body_name,
                    'camera_names': self.get_simu_camera_names(),
                    'use_ik': True,
                    'fps': 20,
                    'show_viewer': False,
                    'use_process_renderer': False,
                }
                
                plate_model_xml = self._get_plate_model_xml()
                if plate_model_xml:
                    task_simu_config['plate_model_xml'] = plate_model_xml
                    task_simu_config['plate_body_name'] = plate_body_name
                
                simu_object_pos = self._transform_position(object_pos)
                task_simu_config['object_position'] = simu_object_pos.tolist()
                if self._sim_initial_joints_deg is not None:
                    task_simu_config['initial_joints_deg'] = self._sim_initial_joints_deg.tolist()
                task_simu_config['initial_gripper'] = getattr(self, '_sim_initial_gripper', 0.0)
                
                simu_plate_pos = self._transform_position(plate_pos)
                if plate_model_xml and plate_body_name:
                    task_simu_config['plate_position'] = simu_plate_pos.tolist()
                
                self._log("启动仿真...", "INFO")
                if not self._simu_manager.start_task_simulation(task_simu_config):
                    self._log(f"仿真场景创建失败: {object_name}", "ERROR")
                    return False
                self._simu = self._simu_manager.simu
                self._log(f"仿真启动成功，simu={self._simu is not None}", "SUCCESS")
                
                self._grasp_executor.set_simu_interface(self._simu)
                if self._simu_data_collector is not None:
                    self._simu_data_collector.set_simu_interface(self._simu)
                self._grasp_executor.set_sim_object_body_name(object_body_name)
                self._simu.set_active_object_body_name(object_body_name)
                self._teleop_controller.set_simu_interface(self._simu)
                self._teleop_controller.set_frame_callback(None)
                
                self._log("启动仿真查看器...", "INFO")
                viewer_started = False
                try:
                    if hasattr(self._simu, 'start_glfw_viewer'):
                        viewer_started = self._simu.start_glfw_viewer(
                            width=1200,
                            height=900,
                            title=f"遥操作 - {task_name}"
                        )
                except Exception as e:
                    self._log(f"启动查看器异常: {e}", "WARNING")
                
                if viewer_started:
                    self._log("仿真查看器已启动", "SUCCESS")
                else:
                    self._log("警告：仿真查看器启动失败，但控制仍可继续", "WARNING")
                
                self._active_task_info = {
                    'task_id': task_id,
                    'task_name': task_name,
                    'object_name': object_name,
                    'object_body_name': object_body_name,
                    'object_pos': np.asarray(simu_object_pos, dtype=float),
                    'target_pos': np.asarray(simu_plate_pos, dtype=float),
                    'target_pos_user': np.asarray(plate_pos, dtype=float),
                }
                
                single_task_info = {
                    'task_id': task_id,
                    'task_name': task_name,
                    'description': task_config.get('description', task_id),
                }
                
                ik_ok = False
                if self._simu is not None:
                    try:
                        ik_ok = self._simu.is_ik_available()
                    except Exception:
                        ik_ok = False
                self._log(f"IK求解器可用: {ik_ok}", "INFO" if ik_ok else "ERROR")
                if not ik_ok:
                    self._log("错误：IK不可用，机械臂无法移动", "ERROR")
                    return False
                
                self._update_status("对齐中...", "#ffa500")
                self._sync_state_to_phone()
                self._log("执行对齐...", "INFO")
                
                align_ok = False
                try:
                    align_ok = self._teleop_controller.align()
                except Exception as e:
                    self._log(f"对齐异常: {e}", "ERROR")
                    align_ok = False
                self._log(f"对齐结果: {'成功' if align_ok else '失败'}, aligned={self._teleop_controller.is_aligned}", 
                         "SUCCESS" if align_ok else "ERROR")
                
                if not align_ok:
                    self._log("对齐失败，无法开始任务", "ERROR")
                    self._update_status("对齐失败", "#ff4444")
                    self._sync_state_to_phone()
                    return False
                
                if self._simu_data_collector is not None:
                    try:
                        self._simu_data_collector.start_episode(
                            self._current_episode,
                            self.get_simu_camera_names(),
                            single_task_info,
                        )
                        self._simu_data_collector.start_recording()
                        self._teleop_controller.set_frame_callback(self._simu_data_collector.collect_frame)
                    except Exception as e:
                        self._log(f"启动数据录制失败: {e}", "WARNING")
                self._log(f"Episode {self._current_episode} 已启动，等待手机遥操作", "INFO")
                
                self._set_publisher_auto_step(False)
                self._teleop_controller.set_collecting_active(True)
                
                self._log("任务就绪，可以开始遥操作采集", "SUCCESS")
                self._update_status("遥操作采集中", "#44ff44")
                self._sync_state_to_phone()
                return True

            # ==============================================================
            # 非遥操作模式：离线/实机自动执行
            # ==============================================================
            self._log("【非遥操作模式】执行自动流程...", "INFO")
            
            plate_pos_config = task_config.get('plate_position', [0.35, -0.15, 0.44])
            plate_pos = self._resolve_position(plate_pos_config)
            min_dist = self._config.get('workspace', {}).get('min_object_plate_distance', 0.15)
            object_pos_config = task_config.get('object_position', [0.35, 0.15, 0.44])
            object_pos = self._resolve_position(object_pos_config, existing_position=plate_pos, min_distance=min_dist)
            
            self._grasp_executor.set_object_type(object_name)
            self._log(f"物体类型: {object_name}", "INFO")
            
            self._grasp_executor.set_target(
                object_position=object_pos.tolist(),
                place_position=plate_pos.tolist(),
            )
            
            has_simu = self._simu is not None or (self._simu_base_xml_path and self._simu_manager)
            if has_simu:
                object_body_name = self._infer_object_body_name(object_name, task_config)
                plate_body_name = self._infer_plate_body_name()
                
                self._log(f"重建仿真场景: object={object_name}", "INFO")
                task_simu_config = {
                    'base_scene_xml': self._simu_base_xml_path,
                    'object_model_xml': self._get_object_model_xml(object_name),
                    'object_body_name': object_body_name,
                    'camera_names': self.get_simu_camera_names(),
                    'use_ik': not self._real_connected,
                    'fps': 20,
                    'show_viewer': self._show_simu_viewer,
                    'use_process_renderer': self._real_connected,
                }
                
                plate_model_xml = self._get_plate_model_xml()
                if plate_model_xml:
                    task_simu_config['plate_model_xml'] = plate_model_xml
                    task_simu_config['plate_body_name'] = plate_body_name
                
                simu_object_pos = self._transform_position(object_pos)
                task_simu_config['object_position'] = simu_object_pos.tolist()
                if self._sim_initial_joints_deg is not None:
                    task_simu_config['initial_joints_deg'] = self._sim_initial_joints_deg.tolist()
                task_simu_config['initial_gripper'] = getattr(self, '_sim_initial_gripper', 0.0)
                
                simu_plate_pos = self._transform_position(plate_pos)
                if plate_model_xml and plate_body_name:
                    task_simu_config['plate_position'] = simu_plate_pos.tolist()
                
                if not self._simu_manager.start_task_simulation(task_simu_config):
                    self._log(f"动态加载物体失败: {object_name}", "ERROR")
                    return False
                self._simu = self._simu_manager.simu
                
                self._grasp_executor.set_simu_interface(self._simu)
                if self._simu_data_collector is not None:
                    self._simu_data_collector.set_simu_interface(self._simu)
                self._grasp_executor.set_sim_object_body_name(object_body_name)
                self._simu.set_active_object_body_name(object_body_name)
                
                self._active_task_info = {
                    'task_id': task_id,
                    'task_name': task_name,
                    'object_name': object_name,
                    'object_body_name': object_body_name,
                    'object_pos': np.asarray(simu_object_pos, dtype=float),
                    'target_pos': np.asarray(simu_plate_pos, dtype=float),
                    'target_pos_user': np.asarray(plate_pos, dtype=float),
                }
            else:
                self._active_task_info = {
                    'task_id': task_id,
                    'task_name': task_name,
                    'object_name': object_name,
                    'description': task_config.get('description', task_id),
                }
            
            single_task_info = {
                'task_id': task_id,
                'task_name': task_config.get('description', task_id),
                'description': task_config.get('description', task_id),
            }
            
            if self._real_connected:
                self._log("实机模式: 自动执行抓取流程...", "INFO")
                self._update_status("自动抓取中", "#44ff44")
                if self._real_data_collector is not None:
                    self._real_data_collector.start_episode(
                        self._current_episode,
                        self.get_real_camera_names(),
                        single_task_info,
                    )
                    self._real_data_collector.start_recording()
                if self._simu_data_collector is not None:
                    self._simu_data_collector.start_episode(
                        self._current_episode,
                        self.get_simu_camera_names(),
                        single_task_info,
                    )
                    self._simu_data_collector.start_recording()
                self._log(f"Episode {self._current_episode} 已启动", "INFO")
                
                def _auto_execute():
                    success = self._grasp_executor.execute()
                    self._log(f"自动抓取流程{'完成' if success else '未完全成功'}", "SUCCESS" if success else "WARNING")
                    self.finish_current_task()
                threading.Thread(target=_auto_execute, daemon=True).start()
            elif self._simu is not None:
                self._log("仿真离线模式: 准备自动执行", "INFO")
                if self._real_data_collector is not None:
                    self._real_data_collector.start_episode(
                        self._current_episode,
                        self.get_real_camera_names(),
                        single_task_info,
                    )
                    self._real_data_collector.start_recording()
                if self._simu_data_collector is not None:
                    self._simu_data_collector.start_episode(
                        self._current_episode,
                        self.get_simu_camera_names(),
                        single_task_info,
                    )
                    self._simu_data_collector.start_recording()
                self._log(f"Episode {self._current_episode} 已启动", "INFO")
                if self._show_simu_viewer:
                    self._simu.start_glfw_viewer(
                        width=1200,
                        height=900,
                        title=f"离线模式 - {task_name}"
                    )
                self.move_current_task_to_pre_grasp()
            
            return True
            
        except Exception as e:
            if self._real_data_collector:
                try:
                    self._real_data_collector.stop_recording()
                except Exception:
                    pass
            if self._simu_data_collector:
                try:
                    self._simu_data_collector.stop_recording()
                except Exception:
                    pass
            self._log(f"任务执行错误: {e}", "ERROR")
            logger.error(f"Task execution error: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _evaluate_task_success(self) -> bool:
        """判定任务是否成功：物体是否在盘子半径范围内（仅仿真模式下可验证）"""
        if not self._active_task_info:
            return False
        if self._simu is None or not hasattr(self._simu, 'get_object_position'):
            return True  # 实机模式无法自动验证，默认成功

        # 获取物体当前位置
        object_body_name = self._active_task_info.get('object_body_name', 'cube')
        object_pos = np.asarray(self._simu.get_object_position(object_body_name), dtype=float)

        # 获取盘子位置（target_pos 是仿真坐标系中的盘子位置）
        plate_pos = np.asarray(self._active_task_info.get('target_pos', np.zeros(3)), dtype=float)

        # 计算物体到盘子中心的 XY 平面距离
        distance_xy = np.linalg.norm(object_pos[:2] - plate_pos[:2])

        # 从配置获取盘子半径，默认 0.09m
        plate_target_cfg = self._config.get('simulation', {}).get('plate_target', {})
        if not plate_target_cfg:
            plate_target_cfg = self._config.get('plate_target', {})
        plate_radius = plate_target_cfg.get('radius', 0.09)

        self._log(f"任务判定: object_pos={object_pos}, plate_pos={plate_pos}", "INFO")
        self._log(f"任务判定: distance_xy={distance_xy:.4f}m, plate_radius={plate_radius:.2f}m", "INFO")

        # 成功条件：物体在盘子半径范围内
        success = bool(distance_xy < plate_radius)
        self._log(f"任务判定结果: {'成功' if success else '失败'}", "SUCCESS" if success else "WARNING")

        return success

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
        return bool(self._grasp_executor.move_to_position(pose))

    def move_current_task_to_grasp(self) -> bool:
        if self._grasp_executor is None:
            return False
        waypoints = self._grasp_executor.get_waypoints()
        if len(waypoints) < 2:
            return False
        pose = np.asarray(waypoints[1][0], dtype=float)
        self._log("执行 IK 到抓取位", "INFO")
        return bool(self._grasp_executor.move_to_position(pose))

    def move_current_task_to_lift(self) -> bool:
        if self._grasp_executor is None:
            return False
        waypoints = self._grasp_executor.get_waypoints()
        if len(waypoints) < 3:
            return False
        pose = np.asarray(waypoints[2][0], dtype=float)
        self._log("执行 IK 到抬起位", "INFO")
        return bool(self._grasp_executor.move_to_position(pose))

    def move_current_task_to_pre_place(self) -> bool:
        if self._grasp_executor is None:
            return False
        waypoints = self._grasp_executor.get_waypoints()
        if len(waypoints) < 4:
            return False
        pose = np.asarray(waypoints[3][0], dtype=float)
        self._log("执行 IK 到预放置位", "INFO")
        return bool(self._grasp_executor.move_to_position(pose))

    def move_current_task_to_place(self) -> bool:
        if self._grasp_executor is None:
            return False
        waypoints = self._grasp_executor.get_waypoints()
        if len(waypoints) < 5:
            return False
        pose = np.asarray(waypoints[4][0], dtype=float)
        self._log("执行 IK 到放置位", "INFO")
        return bool(self._grasp_executor.move_to_position(pose))

    def move_to_home_pose(self) -> bool:
        if self._grasp_executor is None:
            return False
        self._grasp_executor.move_to_initial_joints()
        return True

    def get_gripper_close_position(self) -> float:
        if self._grasp_executor is None:
            return 0.65
        return float(self._grasp_executor.gripper_close_position)

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
        
        # 停止数据记录并丢弃当前 episode（不保存）
        if self._real_data_collector is not None:
            self._real_data_collector.stop_recording()
            self._real_data_collector.discard_current_task()
        if self._simu_data_collector is not None:
            self._simu_data_collector.stop_recording()
            self._simu_data_collector.discard_current_task()
        self._log("已停止数据记录（当前 episode 已丢弃）", "WARNING")

        # 任务跳过：停止仿真（下次任务时重建）
        if self._simu_manager.is_running:
            try:
                self._log("正在关闭仿真窗口...", "INFO")
                self._simu_manager.stop_simulation()
                self._simu = None
                self._log("仿真窗口已关闭", "SUCCESS")
            except Exception as e:
                self._log(f"关闭仿真失败: {e}", "WARNING")

        self._task_in_progress = False
        self._active_task_info = {}
        self._current_task_index += 1
        self._log(f"已跳过当前任务，等待下一个任务... (索引: {self._current_task_index})", "WARNING")
    
    def cleanup(self):
        """清理资源"""
        self.stop()

        # 停止遥操作服务
        if self._ws_server is not None:
            self._ws_server.stop()
            self._ws_server = None
        self._teleop_controller = None

        # 停止发布者
        if self._real_publisher is not None:
            self._real_publisher.stop()
            self._real_publisher = None

        # 通过 SimuManager 清理仿真
        self._simu_manager.stop_simulation()
        self._simu = None

        # 清理真实机器人
        if self._real:
            self._real.disconnect()

        self._log("系统已清理", "INFO")


def main():
    # 配置日志：让 INFO 级别诊断日志能输出到控制台
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    script_dir = Path(__file__).parent
    real_config = str(script_dir / "config" / "real_config.yaml")
    simu_config = str(script_dir / "config" / "simu_config.yaml")
    default_config = str(script_dir / "config" / "tasks_config.yaml")  # 兼容旧配置
    
    parser = argparse.ArgumentParser(description="Kortex 数据收集系统 - Qt GUI")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（不指定则根据模式自动选择）"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="使用真实机器人（实机同步模式），自动使用 real_config.yaml"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用模拟接口（测试用）"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线模式：跳过实机连接，仅使用仿真（调试用），自动使用 simu_config.yaml"
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="手机遥操作模式：启用 WebSocket 服务端，接收手机 ARCore 增量位姿控制机械臂"
    )
    args = parser.parse_args()

    teleop_mode = args.teleop

    if args.real:
        use_real = True
        mode_name = "实机同步模式"
        default_config = real_config
    elif args.mock or args.offline:
        use_real = False
        mode_name = "模拟模式"
        default_config = simu_config
    else:
        use_real = False
        mode_name = "模拟模式 (默认)"
        default_config = simu_config

    if teleop_mode:
        mode_name += " + 遥操作"
    
    # 如果用户指定了配置文件，优先使用
    config_path = args.config if args.config else default_config

    logger.info(f"启动模式: {mode_name}")
    logger.info(f"配置文件: {config_path}")

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow(config_path, mock_mode=not use_real)
    window.show()

    system = DataCollectionSystem(
        config_path,
        use_real=use_real,
        show_simu_viewer=not use_real,
        teleop_mode=teleop_mode,
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

    # 检查实机连接状态
    if use_real and not system.is_real_connected():
        window.log("[!] 实机未连接，以离线模式运行（仅仿真）", "WARNING")
        window.update_status("离线模式（实机未连接）", "#ffa500")

    # 始终设置相机显示（模拟模式下实机相机为空列表）
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
