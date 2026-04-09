"""
仿真机器人数据收集器 - 完全独立
"""
import json
import time
import numpy as np
import threading
import cv2
from typing import Dict, List, Any
from pathlib import Path


def create_video_from_images(image_list: List[bytes], output_path: Path, fps: int = 30, width: int = 640, height: int = 480):
    """将图像列表转换为视频文件"""
    if not image_list:
        return False
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"[SimuVideo] Failed to open video writer for {output_path}")
            return False
        
        for img_bytes in image_list:
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img_array = img_array.reshape(height, width, 3)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)
        
        writer.release()
        print(f"[SimuVideo] Saved: {output_path} ({len(image_list)} frames)")
        return True
    except Exception as e:
        print(f"[SimuVideo] Error: {e}")
        return False


class SimuDataCollector:
    """仿真机器人数据收集器 - 完全独立"""
    
    def __init__(self, simu_interface, data_root: str, fps: int = 20, video_fps: int = 30, run_in_thread: bool = False):
        self._simu = simu_interface
        self._data_root = Path(data_root)
        self._fps = fps
        self._video_fps = video_fps
        self._episode_count = 0
        self._is_collecting = False
        self._is_recording = False  # 是否正在记录数据（任务执行期间）
        self._run_in_thread = run_in_thread
        self._collect_thread = None
        self._stop_event = threading.Event()
        self._data_lock = threading.Lock()
        self._min_frame_interval = (1.0 / float(self._fps)) if self._fps > 0 else 0.0
        self._last_collect_ts = 0.0
        
        # 任务数据保存点（用于重做任务）
        self._task_save_point = {"states": 0, "actions": 0, "images": {}}
        self._last_task_end_point = {"states": 0, "actions": 0, "images": {}}
        
        # 当前episode数据
        self._current_data = {"observations": {}, "actions": []}
        self._current_episode_info = {}
        self._camera_names = []
        
        self._save_progress()
    
    def _get_episode_dir(self, episode_id: int) -> Path:
        return self._data_root / f"episode_{episode_id:03d}"
    
    def _ensure_directories(self, episode_id: int):
        episode_dir = self._get_episode_dir(episode_id)
        obs_dir = episode_dir / "observation"
        obs_dir.mkdir(parents=True, exist_ok=True)
        
        for cam_name in self._camera_names:
            (obs_dir / f"{cam_name}_image").mkdir(exist_ok=True)
        (episode_dir / "action").mkdir(exist_ok=True)
        (episode_dir / "video").mkdir(exist_ok=True)
    
    def start_collection(self):
        """开始收集"""
        if self._is_collecting:
            return
        self._is_collecting = True

        if self._run_in_thread:
            self._stop_event.clear()
            self._collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
            self._collect_thread.start()
            print(f"[SimuDataCollector] Started (thread mode, target FPS: {self._fps})")
        else:
            print("[SimuDataCollector] Started (callback mode)")

    def stop_collection(self):
        if not self._is_collecting:
            return

        self._is_collecting = False
        self._stop_event.set()

        if self._collect_thread is not None:
            self._collect_thread.join(timeout=2.0)
            self._collect_thread = None

        self._is_recording = False
        print("[SimuDataCollector] Stopped")

    def _collect_loop(self):
        while not self._stop_event.is_set():
            loop_start = time.time()
            self.collect_frame()
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, (1.0 / self._fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def collect_frame(self):
        """收集一帧数据（可被回调调用，也可在线程中调用）"""
        if not self._is_collecting or not self._is_recording:
            return

        try:
            timestamp = time.time()
            if self._min_frame_interval > 0 and (timestamp - self._last_collect_ts) < self._min_frame_interval:
                return
            self._last_collect_ts = timestamp

            images = self._simu.get_camera_images(self._camera_names)
            joints = self._simu.get_joint_state()
            gripper = self._simu.get_gripper_state()
            tcp_pos = self._simu.get_tcp_position() if hasattr(self._simu, 'get_tcp_position') else None

            with self._data_lock:
                for cam_name, img in images.items():
                    key = f"{cam_name}_images"
                    if key not in self._current_data["observations"]:
                        self._current_data["observations"][key] = []
                    self._current_data["observations"][key].append(img.tobytes())

                state = {
                    "joint_positions": joints.tolist(),
                    "gripper": float(gripper),
                    "timestamp": timestamp,
                }
                if tcp_pos is not None:
                    state["cartesian_position"] = np.asarray(tcp_pos, dtype=float).tolist()

                if "states" not in self._current_data["observations"]:
                    self._current_data["observations"]["states"] = []
                self._current_data["observations"]["states"].append(state)

                self._current_data["actions"].append(state.copy())

        except Exception as e:
            print(f"[SimuDataCollector] Error: {e}")
    
    def start_episode(self, episode_id: int, camera_names: list, task_info: Dict[str, Any]):
        self._episode_count = episode_id
        self._camera_names = camera_names
        self._ensure_directories(episode_id)
        
        with self._data_lock:
            self._current_data = {"observations": {}, "actions": []}
            self._current_episode_info = {
                "episode_id": episode_id,
                "task_id": task_info.get("task_id", episode_id),
                "task_name": task_info.get("task_name", ""),
                "description": task_info.get("description", ""),
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        print(f"[SimuDataCollector] Episode {episode_id} started")
    
    def start_recording(self):
        """开始记录数据（任务执行期间）"""
        with self._data_lock:
            # 保存当前数据长度作为保存点
            self._task_save_point["states"] = len(self._current_data["observations"].get("states", []))
            self._task_save_point["actions"] = len(self._current_data["actions"])
            # 保存所有图像键的长度
            self._task_save_point["images"] = {}
            for key in self._current_data["observations"]:
                if key.endswith("_images"):
                    self._task_save_point["images"][key] = len(self._current_data["observations"][key])
            # 如果还没有图像数据，为每个相机初始化保存点为0
            for cam_name in self._camera_names:
                key = f"{cam_name}_images"
                if key not in self._task_save_point["images"]:
                    self._task_save_point["images"][key] = 0
        self._is_recording = True
        self._last_collect_ts = 0.0
        print(f"[SimuDataCollector] Recording started, save point: states={self._task_save_point['states']}, images={self._task_save_point['images']}")
    
    def stop_recording(self):
        """停止记录数据（任务执行期间）"""
        self._is_recording = False
        # 保存任务结束时的数据长度（用于重做任务）
        with self._data_lock:
            self._last_task_end_point["states"] = len(self._current_data["observations"].get("states", []))
            self._last_task_end_point["actions"] = len(self._current_data["actions"])
            self._last_task_end_point["images"] = {}
            for key in self._current_data["observations"]:
                if key.endswith("_images"):
                    self._last_task_end_point["images"][key] = len(self._current_data["observations"][key])
        print("[SimuDataCollector] Recording stopped")
    
    def discard_current_task(self):
        """丢弃当前任务的数据（用于重做任务）"""
        with self._data_lock:
            # 截断数据到任务开始前的保存点
            states = self._current_data["observations"].get("states", [])
            if len(states) > self._task_save_point["states"]:
                self._current_data["observations"]["states"] = states[:self._task_save_point["states"]]
            
            actions = self._current_data["actions"]
            if len(actions) > self._task_save_point["actions"]:
                self._current_data["actions"] = actions[:self._task_save_point["actions"]]
            
            # 截断所有图像数据
            for key in list(self._current_data["observations"].keys()):
                if key.endswith("_images"):
                    save_len = self._task_save_point["images"].get(key, 0)
                    images = self._current_data["observations"][key]
                    if len(images) > save_len:
                        self._current_data["observations"][key] = images[:save_len]
            
            print(f"[SimuDataCollector] Current task data discarded, reverted to states={self._task_save_point['states']}, images={self._task_save_point['images']}")
    
    def end_episode(self, episode_id: int, success: bool = True):
        self._is_recording = False  # 确保停止记录
        with self._data_lock:
            self._current_episode_info["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._current_episode_info["success"] = success
            return self._save_episode(episode_id)
    
    def _save_episode(self, episode_id: int) -> bool:
        try:
            episode_dir = self._get_episode_dir(episode_id)
            obs_dir = episode_dir / "observation"
            video_dir = episode_dir / "video"
            action_dir = episode_dir / "action"

            # 防御性创建目录：避免 episode 目录被外部清理后写入失败
            obs_dir.mkdir(parents=True, exist_ok=True)
            video_dir.mkdir(parents=True, exist_ok=True)
            action_dir.mkdir(parents=True, exist_ok=True)
            
            num_frames = len(self._current_data["observations"].get("states", []))
            print(f"[SimuDataCollector] Saving episode {episode_id} ({num_frames} frames)")
            
            if num_frames == 0:
                print("[SimuDataCollector] Warning: No frames!")
                return False
            
            # 保存图像和视频
            for key, value in self._current_data["observations"].items():
                if key == "states" or not isinstance(value, list) or not value:
                    continue
                
                cam_name = key.replace("_images", "")
                img_dir = obs_dir / f"{cam_name}_image"
                img_dir.mkdir(parents=True, exist_ok=True)

                # 保存为PNG格式
                for i, img_bytes in enumerate(value):
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img_array = img_array.reshape(480, 640, 3)
                    # OpenCV使用BGR格式，需要转换
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(img_dir / f"frame_{i:06d}.png"), img_bgr)
                
                # 保存视频
                video_path = video_dir / f"{cam_name}.mp4"
                create_video_from_images(value, video_path, fps=self._video_fps)
            
            # 保存状态和动作
            with open(obs_dir / "state.json", "w") as f:
                json.dump(self._current_data["observations"]["states"], f, indent=2)
            
            with open(action_dir / "action.json", "w") as f:
                json.dump(self._current_data["actions"], f, indent=2)
            
            with open(episode_dir / "metadata.json", "w") as f:
                json.dump(self._current_episode_info, f, indent=2)
            
            self._save_progress()
            print(f"[SimuDataCollector] Episode {episode_id} saved")
            return True
            
        except Exception as e:
            print(f"[SimuDataCollector] Save error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_progress(self):
        self._data_root.mkdir(parents=True, exist_ok=True)
        with open(self._data_root / "progress.json", "w") as f:
            json.dump({"episode_count": self._episode_count}, f, indent=2)
    
    def load_progress(self) -> int:
        progress_file = self._data_root / "progress.json"
        if progress_file.exists():
            with open(progress_file, "r") as f:
                self._episode_count = json.load(f).get("episode_count", 0)
        return self._episode_count
