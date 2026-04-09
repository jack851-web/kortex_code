import numpy as np
from typing import Optional, Dict
import yaml
from pathlib import Path
import cv2
import threading
import time
from .simple_camera import CameraManager


class RobotNotConnectedError(Exception):
    pass


class RealInterface:
    def __init__(self, config=None, camera_config: Dict = None):
        self._robot = None
        self._config = config
        self._connected = False
        self._Gen3Lite = None
        self._Gen3LiteConfig = None
        self._camera_config = camera_config
        self._camera_names = list(camera_config.keys()) if camera_config else []
        self._camera_manager: Optional[CameraManager] = None
        self._viewer_running = False
        self._viewer_thread = None

    @classmethod
    def from_config_file(cls, config_path: str) -> "RealInterface":
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        camera_config = {}
        for cam_name, cam_cfg in cfg.get("cameras", {}).items():
            camera_config[cam_name] = {
                "index": cam_cfg.get("index", 0),
                "width": cam_cfg.get("width", 640),
                "height": cam_cfg.get("height", 480),
                "fps": cam_cfg.get("fps", 30)
            }
        
        return cls(camera_config=camera_config)

    def _ensure_import(self):
        if self._Gen3Lite is not None:
            return True
        try:
            import sys
            sys.path.insert(0, r'D:\VLA\kortex_code\kortex_real')
            from gen3 import Gen3Lite, Gen3LiteConfig
            self._Gen3Lite = Gen3Lite
            self._Gen3LiteConfig = Gen3LiteConfig
            return True
        except ImportError as e:
            raise ImportError(f"Failed to import Gen3Lite: {e}")

    def _check_connection(self):
        if not self._connected or self._robot is None:
            raise RobotNotConnectedError("Robot is not connected. Call connect() first.")

    def connect(self, ip: str, username: str = "admin", password: str = "admin") -> bool:
        try:
            self._ensure_import()
        except ImportError as e:
            print(f"Failed to import robot module: {e}")
            return False
        try:
            self._config = self._Gen3LiteConfig(
                ip_address=ip,
                username=username,
                password=password,
                gripper_enabled=True,
                cameras={}
            )
            self._robot = self._Gen3Lite(self._config)
            self._robot.connect(calibrate=False)
            self._connected = True
            
            if self._camera_config:
                print("Connecting cameras...")
                self._camera_manager = CameraManager(self._camera_config)
                if not self._camera_manager.connect():
                    print("Warning: Some cameras failed to connect")
            
            return True
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            return False

    def get_joint_state(self) -> np.ndarray:
        self._check_connection()
        try:
            obs = self._robot.get_observation()
            joints_deg = np.array([
                obs.get(f"joint_{i}.pos", 0.0) for i in range(1, 7)
            ])
            return joints_deg
        except Exception as e:
            raise RuntimeError(f"Failed to get joint state: {e}")

    def get_cartesian_pose(self) -> np.ndarray:
        self._check_connection()
        try:
            obs = self._robot.get_observation()
            pose = np.array([
                obs.get("ee.x", 0.0),
                obs.get("ee.y", 0.0),
                obs.get("ee.z", 0.0),
                obs.get("ee.wx", 0.0),
                obs.get("ee.wy", 0.0),
                obs.get("ee.wz", 0.0),
            ])
            return pose
        except Exception as e:
            raise RuntimeError(f"Failed to get cartesian pose: {e}")
    
    def get_full_state(self) -> tuple:
        """一次性获取所有状态，减少通信开销"""
        self._check_connection()
        try:
            obs = self._robot.get_observation()
            joints_deg = np.array([
                obs.get(f"joint_{i}.pos", 0.0) for i in range(1, 7)
            ])
            pose = np.array([
                obs.get("ee.x", 0.0),
                obs.get("ee.y", 0.0),
                obs.get("ee.z", 0.0),
                obs.get("ee.wx", 0.0),
                obs.get("ee.wy", 0.0),
                obs.get("ee.wz", 0.0),
            ])
            gripper = obs.get("gripper.pos", 0.0)
            return joints_deg, pose, gripper
        except Exception as e:
            raise RuntimeError(f"Failed to get full state: {e}")

    def set_joint_target(self, joints: np.ndarray) -> bool:
        self._check_connection()
        if len(joints) != 6:
            raise ValueError(f"Expected 6 joint values, got {len(joints)}")
        try:
            action = {f"joint_{i}.pos": joints[i-1] for i in range(1, 7)}
            self._robot.send_action(action)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to set joint target: {e}")

    def move_cartesian(self, pose: np.ndarray) -> bool:
        self._check_connection()
        if len(pose) < 6:
            current_pose = self.get_cartesian_pose()
            pose = np.concatenate([pose, current_pose[3:6]])
        try:
            return self._robot.arm_move_cartesian(pose.tolist())
        except Exception as e:
            raise RuntimeError(f"Failed to move cartesian: {e}")

    def get_gripper_state(self) -> float:
        self._check_connection()
        try:
            obs = self._robot.get_observation()
            return obs.get("gripper.pos", 0.0)
        except Exception as e:
            raise RuntimeError(f"Failed to get gripper state: {e}")

    def set_gripper(self, position: float) -> bool:
        self._check_connection()
        print(f"[RealInterface] set_gripper called with position={position}")
        position = np.clip(position, 0.0, 1.0)
        print(f"[RealInterface] After clip: position={position}")
        try:
            action = {"gripper.pos": position}
            self._robot.send_action(action)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to set gripper: {e}")
    
    def is_at_position(self, target_pose: np.ndarray, position_tol: float = 0.005, orientation_tol: float = 1.0) -> bool:
        """检查是否到达目标位置"""
        current_pose = self.get_cartesian_pose()
        pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
        ori_diff = np.linalg.norm(current_pose[3:6] - target_pose[3:6])
        return pos_diff < position_tol and ori_diff < orientation_tol
    
    def wait_for_arrival(self, target_pose: np.ndarray, timeout: float = 10.0, 
                         position_tol: float = 0.005, orientation_tol: float = 1.0,
                         callback: callable = None) -> bool:
        """等待到达目标位置"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_pose = self.get_cartesian_pose()
            pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
            ori_diff = np.linalg.norm(current_pose[3:6] - target_pose[3:6])
            
            if callback:
                callback(current_pose, target_pose, pos_diff, ori_diff)
            
            if pos_diff < position_tol and ori_diff < orientation_tol:
                return True
            time.sleep(0.05)
        return False

    def get_camera_images(self) -> Dict[str, np.ndarray]:
        self._check_connection()
        if self._camera_manager is not None:
            all_images = self._camera_manager.get_images()
            return {name: all_images.get(name, np.zeros((480, 640, 3), dtype=np.uint8)) 
                    for name in self._camera_names}
        return {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in self._camera_names}

    def disconnect(self):
        self.stop_camera_viewer()
        if self._camera_manager is not None:
            self._camera_manager.disconnect()
            self._camera_manager = None
        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception:
                pass
        self._connected = False
        self._robot = None

    def clear_faults(self) -> bool:
        self._check_connection()
        try:
            return self._robot.clear_faults()
        except Exception as e:
            raise RuntimeError(f"Failed to clear faults: {e}")

    def is_connected(self) -> bool:
        return self._connected

    def show_camera_images(self, wait_key: int = 1) -> bool:
        self._check_connection()
        try:
            images = self.get_camera_images()
            for cam_name, img in images.items():
                cv2.imshow(f"Camera: {cam_name}", img)
            key = cv2.waitKey(wait_key)
            return key != ord('q')
        except Exception as e:
            raise RuntimeError(f"Failed to show camera images: {e}")

    def _viewer_loop(self):
        while self._viewer_running and self._connected:
            try:
                self.show_camera_images(wait_key=30)
            except Exception:
                break
        cv2.destroyAllWindows()

    def start_camera_viewer(self):
        if self._viewer_running:
            return
        self._viewer_running = True
        self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self._viewer_thread.start()

    def stop_camera_viewer(self):
        self._viewer_running = False
        if self._viewer_thread is not None:
            self._viewer_thread.join(timeout=1.0)
            self._viewer_thread = None
        cv2.destroyAllWindows()



class MockRealInterface:
    def __init__(self, camera_names: list = None):
        self._connected = False
        self._gripper_position = 0.0
        self._joint_state = np.zeros(6)
        self._cartesian_pose = np.array([0.4, 0.0, 0.3, 180.0, 0.0, 0.0])
        self._viewer_running = False
        self._viewer_thread = None
        self._camera_names = camera_names or ["top"]

    def _check_connection(self):
        if not self._connected:
            raise RobotNotConnectedError("Mock robot is not connected. Call connect() first.")

    def connect(self, ip: str, username: str = "admin", password: str = "admin") -> bool:
        self._connected = True
        return True

    def disconnect(self):
        self.stop_camera_viewer()
        self._connected = False

    def clear_faults(self) -> bool:
        return True

    def is_connected(self) -> bool:
        return self._connected

    def get_joint_state(self) -> np.ndarray:
        self._check_connection()
        return self._joint_state.copy()

    def get_cartesian_pose(self) -> np.ndarray:
        self._check_connection()
        return self._cartesian_pose.copy()

    def get_full_state(self) -> tuple:
        """一次性获取所有状态，减少通信开销"""
        self._check_connection()
        return self._joint_state.copy(), self._cartesian_pose.copy(), self._gripper_position

    def set_joint_target(self, joints: np.ndarray) -> bool:
        self._check_connection()
        if len(joints) != 6:
            raise ValueError(f"Expected 6 joint values, got {len(joints)}")
        self._joint_state = np.array(joints)
        return True

    def move_cartesian(self, pose: np.ndarray) -> bool:
        self._check_connection()
        if len(pose) < 6:
            current_pose = self.get_cartesian_pose()
            pose = np.concatenate([pose, current_pose[3:6]])
        self._cartesian_pose = np.array(pose)
        return True

    def get_gripper_state(self) -> float:
        self._check_connection()
        return self._gripper_position

    def set_gripper(self, position: float) -> bool:
        self._check_connection()
        self._gripper_position = np.clip(position, 0.0, 1.0)
        return True

    def get_camera_images(self) -> Dict[str, np.ndarray]:
        self._check_connection()
        return {name: np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                for name in self._camera_names}

    def show_camera_images(self, wait_key: int = 1) -> bool:
        self._check_connection()
        try:
            images = self.get_camera_images()
            for cam_name, img in images.items():
                cv2.imshow(f"Camera: {cam_name}", img)
            key = cv2.waitKey(wait_key)
            return key != ord('q')
        except Exception as e:
            raise RuntimeError(f"Failed to show camera images: {e}")

    def _viewer_loop(self):
        while self._viewer_running and self._connected:
            try:
                self.show_camera_images(wait_key=30)
            except Exception:
                break
        cv2.destroyAllWindows()

    def start_camera_viewer(self):
        if self._viewer_running:
            return
        self._viewer_running = True
        self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self._viewer_thread.start()

    def stop_camera_viewer(self):
        self._viewer_running = False
        if self._viewer_thread is not None:
            self._viewer_thread.join(timeout=1.0)
            self._viewer_thread = None
        cv2.destroyAllWindows()
