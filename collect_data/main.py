import argparse
import sys
import time
import yaml
import numpy as np
from pathlib import Path

from scripts import (
    RealInterface,
    MockRealInterface,
    SimuInterface,
    MockSimuInterface,
    SyncController,
    GraspExecutor,
)
from scripts.real.data_collector import RealDataCollector
from scripts.simu.data_collector import SimuDataCollector

# 项目根目录 (kortex_code)
PROJECT_ROOT = Path(__file__).parent.parent


def resolve_path(path_str: str, base_dir: Path = None) -> str:
    """解析路径，支持相对路径和绝对路径"""
    if not path_str:
        return path_str
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    base = base_dir or PROJECT_ROOT
    return str((base / path_str).resolve())


def resolve_config_paths(config: dict, base_dir: Path = None) -> dict:
    """递归解析配置中的路径字段"""
    path_keys = {
        'xml_path', 'model_xml_path', 'scene_base_xml_path',
        'real_data_root', 'simu_data_root', 'mock_simu_data_root',
        'data_root', 'output_path', 'log_path',
    }
    
    def _resolve_value(key: str, value):
        if isinstance(value, str):
            is_path = (
                key in path_keys or 
                key.endswith('_path') or 
                key.endswith('_root') or
                key.endswith('_xml') or
                key.endswith('_dir') or
                '.xml' in value.lower() or
                '.yaml' in value.lower() or
                '.json' in value.lower()
            )
            if is_path:
                return resolve_path(value, base_dir)
            return value
        elif isinstance(value, dict):
            return {k: _resolve_value(k, v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_resolve_value(key, item) for item in value]
        return value
    
    if config is None:
        return config
    return {k: _resolve_value(k, v) for k, v in config.items()}


class DataCollectionSystem:
    def __init__(self, config_path: str, use_mock: bool = True):
        self._config_path = config_path
        self._use_mock = use_mock
        self._config = None
        self._real = None
        self._simu = None
        self._sync_controller = None
        self._grasp_executor = None
        self._real_data_collector = None
        self._simu_data_collector = None
        self._tasks = []
        self._current_task_index = 0
        self._running = False

        self._simu_base_xml_path = ""
        self._object_library = {}
        self._sim_initial_joints_deg = None

    def load_config(self) -> bool:
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            
            # 解析配置中的相对路径为绝对路径
            self._config = resolve_config_paths(self._config, PROJECT_ROOT)
            
            tasks_dict = self._config.get("tasks", {})
            self._tasks = []
            for task_name, task_data in tasks_dict.items():
                self._tasks.append({
                    "name": task_name,
                    "object_name": task_data.get("object_name", "cube"),
                    "sim_body_name": task_data.get("sim_body_name", "cube"),
                    "object_position": task_data.get("object_position", [0.3, 0.0, 0.05]),
                    "plate_position": task_data.get("plate_position", [0.4, 0.0, 0.05]),
                    "description": task_data.get("description", "")
                })
            return True
        except Exception as e:
            print(f"Failed to load config: {e}")
            return False

    def initialize(self) -> bool:
        robot_config = self._config.get("robot", {})
        simu_config = self._config.get("simulation", {})
        grasp_config = self._config.get("grasp", {})
        dataset_config = self._config.get("dataset", {})
        camera_config = self._config.get("cameras", {})

        self._simu_base_xml_path = simu_config.get("xml_path", "")
        self._object_library = simu_config.get("object_library", {})

        # 仿真初始关节（可在 config 里设置，支持 rad/deg）
        self._sim_initial_joints_deg = None
        sim_init_joints = simu_config.get("initial_joints", None)
        sim_init_unit = str(simu_config.get("initial_joints_unit", "rad")).lower()
        if isinstance(sim_init_joints, list) and len(sim_init_joints) >= 6:
            arr = np.array(sim_init_joints[:6], dtype=float)
            if sim_init_unit in ("rad", "radian", "radians"):
                self._sim_initial_joints_deg = np.rad2deg(arr)
            else:
                self._sim_initial_joints_deg = arr
            print(f"[Config] simulation.initial_joints loaded ({sim_init_unit}): {sim_init_joints[:6]}")
        
        real_camera_config = camera_config.get("real", {})
        _simu_cam_cfg = camera_config.get("simu", [])
        # cameras.simu 支持列表 [name, ...] 或字典 {name: {...}} 格式
        if isinstance(_simu_cam_cfg, list):
            self._simu_camera_names = _simu_cam_cfg if _simu_cam_cfg else ['agentview', 'topview']
        else:
            self._simu_camera_names = list(_simu_cam_cfg.keys()) if _simu_cam_cfg else ['agentview', 'topview']
        
        # 保存相机名称供后续使用
        self._real_camera_names = list(real_camera_config.keys()) if real_camera_config else []

        if self._use_mock:
            print("Using simulation-only mode (IK enabled)")
            self._real = MockRealInterface(camera_names=list(real_camera_config.keys()))
            self._real.connect("mock")
            self._simu = SimuInterface(
                simu_config.get("xml_path", ""),
                camera_names=self._simu_camera_names,
                use_ik=True,
            )
        else:
            print("Using real interfaces")
            real_cam_cfg = {}
            for cam_name, cam_cfg in real_camera_config.items():
                real_cam_cfg[cam_name] = {
                    "index": cam_cfg.get("index", 0),
                    "width": cam_cfg.get("width", 640),
                    "height": cam_cfg.get("height", 480),
                    "fps": cam_cfg.get("fps", 30)
                }
            self._real = RealInterface(camera_config=real_cam_cfg)
            if not self._real.connect(robot_config.get("ip", "192.168.1.10")):
                print("Failed to connect to real robot")
                return False
            self._simu = SimuInterface(
                simu_config.get("xml_path", ""),
                camera_names=self._simu_camera_names,
                use_ik=False,
            )

        if not self._simu.initialize(simu_config.get("xml_path", "")):
            print("Failed to initialize simulation")
            return False

        self._apply_sim_initial_joints()

        simu_camera_names = self._simu_camera_names
        self._simu.set_display_cameras(simu_camera_names)

        # 启动进程渲染器（提高性能且避免缓冲区共享）
        print("\nStarting render process...")
        self._simu.start_render_process()

        print("\nStarting viewers...")
        print("  - Starting simulation viewer (MuJoCo window)")
        self._simu.start_viewer()
        
        if simu_camera_names:
            print(f"  - Starting simulation camera viewer: {simu_camera_names}")
            self._simu.start_camera_viewer(simu_camera_names)
        
        real_camera_names = list(real_camera_config.keys()) if real_camera_config else []
        if (not self._use_mock) and real_camera_names:
            print(f"  - Starting real camera viewer: {real_camera_names}")
            self._real.start_camera_viewer()
        
        print("All viewers started. Press 'q' in camera windows to close them.\n")

        # 创建数据收集器（先创建，再设置回调）
        # 从相机配置中获取视频帧率（使用第一个真实相机的帧率，默认为30）
        real_camera_config = camera_config.get("real", {})
        video_fps = 30
        if real_camera_config:
            first_cam = list(real_camera_config.values())[0]
            video_fps = first_cam.get("fps", 30)
        
        # 创建数据收集器
        if not self._use_mock:
            self._real_data_collector = RealDataCollector(
                self._real,
                data_root=dataset_config.get("real_data_root", "./data/real_data"),
                fps=dataset_config.get("fps", 20),
                video_fps=video_fps,
            )
        else:
            self._real_data_collector = None

        self._simu_data_collector = SimuDataCollector(
            self._simu,
            data_root=dataset_config.get("simu_data_root", "./data/simu_data"),
            fps=dataset_config.get("fps", 20),
            video_fps=video_fps,
            run_in_thread=False,
        )

        # 仅实机模式创建同步控制器；仿真独立模式不做实机同步
        if not self._use_mock:
            self._sync_controller = SyncController(
                self._real,
                self._simu,
                on_sync_callback=self._simu_data_collector.collect_frame
            )
        else:
            self._sync_controller = None

        self._grasp_executor = GraspExecutor(
            self._real,
            self._simu,
            home_position=[0, 0, 0, 0, 0, 0],
            pre_grasp_offset=grasp_config.get("pre_grasp_offset", [0, 0, 0.15]),
            lift_height=grasp_config.get("lift_height", 0.20),
            approach_height=grasp_config.get("approach_height", 0.05),
        )
        
        if "default_orientation" in grasp_config:
            self._grasp_executor.set_default_orientation(grasp_config["default_orientation"])
        if "grasp_offsets" in grasp_config:
            self._grasp_executor.set_grasp_offsets(grasp_config["grasp_offsets"])
        if "gripper_positions" in grasp_config:
            self._grasp_executor.set_gripper_positions_by_object(grasp_config["gripper_positions"])
        if "object_profiles" in grasp_config:
            self._grasp_executor.set_object_profiles(grasp_config["object_profiles"])

        if self._use_mock and self._simu_data_collector is not None:
            self._grasp_executor.set_frame_callback(self._simu_data_collector.collect_frame)

        # 数据收集器已在前面创建
        progress_collector = self._real_data_collector if self._real_data_collector is not None else self._simu_data_collector
        saved_count = progress_collector.load_progress() if progress_collector is not None else 0
        if saved_count > 0:
            print(f"Resumed from episode {saved_count}")
            self._current_task_index = saved_count

        return True

    def run(self):
        if not self.load_config():
            return False
        if not self.initialize():
            return False

        self._running = True

        try:
            if self._sync_controller is not None:
                self._sync_controller.start_sync()
            if self._real_data_collector is not None:
                self._real_data_collector.start_collection()
            self._simu_data_collector.start_collection()

            print(f"\n{'='*60}")
            print("Data Collection System Started")
            print(f"Total tasks: {len(self._tasks)}")
            print(f"Starting from task: {self._current_task_index + 1}")
            print(f"{'='*60}\n")

            for i in range(self._current_task_index, len(self._tasks)):
                if not self._running:
                    break
                task = self._tasks[i]
                self._execute_task(i + 1, task)

            print(f"\n{'='*60}")
            print("All tasks completed")
            episode_count = self._real_data_collector._episode_count if self._real_data_collector is not None else self._simu_data_collector._episode_count
            print(f"Total episodes collected: {episode_count}")
            print(f"{'='*60}\n")

            return True

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return False
        finally:
            self._cleanup()

    def _execute_task(self, task_id: int, task: dict):
        task_name = task.get("name", f"task{task_id}")
        description = task.get("description", "")
        object_name = task.get("object_name", "cube")
        object_body_name = task.get("sim_body_name", "cube")
        object_pos = task.get("object_position", [0.3, 0.0, 0.05])
        plate_pos = task.get("plate_position", [0.4, 0.0, 0.05])

        print(f"\n--- {task_name} ({task_id}/{len(self._tasks)}) ---")
        print(f"Description: {description}")
        print(f"Object: {object_name}, body: {object_body_name}")
        print(f"Object Position: {object_pos}")
        print(f"Plate Position: {plate_pos}")

        object_cfg = self._object_library.get(object_name, {}) if isinstance(self._object_library, dict) else {}
        object_model_xml = object_cfg.get("model_xml_path", "")
        object_body_name = object_cfg.get("body_name", object_body_name)

        if object_model_xml and self._simu_base_xml_path:
            if not self._simu.reload_scene_with_object(
                self._simu_base_xml_path,
                object_model_xml,
                object_body_name=object_body_name,
                show_viewer=False,
            ):
                print(f"[ERROR] Failed to load object model: {object_name}")
                return
            self._apply_sim_initial_joints()

        self._grasp_executor.set_object_type(object_name)
        self._grasp_executor.set_sim_object_body_name(object_body_name)
        self._simu.set_active_object_body_name(object_body_name)

        # 先按config放置x/y，并按模型几何自动对齐z（底部高度）
        self._simu.set_object_position(
            object_body_name,
            object_pos,
            reset_z=True,
        )

        # 预热渲染进程，确保首任务录制前已切到新场景并拿到新物体帧
        for _ in range(3):
            self._simu.get_camera_images(self._simu_camera_names)
            time.sleep(0.03)

        # 再读取任务目标
        self._grasp_executor.set_task(task_id, object_pos, plate_pos)

        task_info = {
            "task_id": task_id,
            "task_name": task_name,
            "description": description,
            "object_position": object_pos,
            "plate_position": plate_pos,
        }

        base_episode_count = self._real_data_collector._episode_count if self._real_data_collector is not None else self._simu_data_collector._episode_count
        episode_id = base_episode_count + 1

        if self._real_data_collector is not None:
            self._real_data_collector.start_episode(episode_id, self._real_camera_names, task_info)
        self._simu_data_collector.start_episode(episode_id, self._simu_camera_names, task_info)
        print(f"Episode {episode_id} started")

        success = self._grasp_executor.execute(
            progress_callback=self._on_waypoint_progress
        )

        if self._real_data_collector is not None:
            self._real_data_collector.end_episode(episode_id, success)
        self._simu_data_collector.end_episode(episode_id, success)
        print(f"Episode {episode_id} completed - Success: {success}")

        if task_id < len(self._tasks):
            print("\n[INFO] Please change the object position for the next task.")
            print("[INFO] Press Enter to continue...")
            try:
                input()
            except EOFError:
                pass

    def _apply_sim_initial_joints(self):
        if self._simu is None or self._sim_initial_joints_deg is None:
            return
        ok = self._simu.set_joint_target(self._sim_initial_joints_deg)
        if ok:
            # 推进仿真使控制量真正生效
            self._simu.step(300)
            print(f"[Config] Applied simulation initial joints (deg): {self._sim_initial_joints_deg.tolist()}")

    def _on_waypoint_progress(self, waypoint_name: str, current: int, total: int):
        print(f"  Progress: [{current}/{total}] {waypoint_name}")

    def _cleanup(self):
        print("\nCleaning up...")
        if self._real_data_collector:
            self._real_data_collector.stop_collection()
        if self._simu_data_collector:
            self._simu_data_collector.stop_collection()
        if self._sync_controller:
            self._sync_controller.stop_sync()
        if self._real:
            self._real.disconnect()
        if self._simu:
            self._simu.stop_render_process()
            self._simu.close()
        print("Cleanup completed")

    def stop(self):
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="Kortex Data Collection System")
    parser.add_argument(
        "--config",
        type=str,
        default="config/tasks_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real robot interfaces instead of mock",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    system = DataCollectionSystem(str(config_path), use_mock=not args.real)
    success = system.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
