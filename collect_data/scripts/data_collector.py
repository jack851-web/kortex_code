import json
import os
import time
import numpy as np
import threading
import cv2
from typing import Dict, List, Optional, Any
from pathlib import Path


def create_video_from_images(image_list: List[bytes], output_path: Path, fps: int = 20, width: int = 640, height: int = 480):
    """将图像列表转换为视频文件"""
    if not image_list:
        return False
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用 mp4v 编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"[Video] Failed to open video writer for {output_path}")
            return False
        
        for img_bytes in image_list:
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img_array = img_array.reshape(height, width, 3)
            # OpenCV 使用 BGR 格式，需要转换
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)
        
        writer.release()
        print(f"[Video] Saved video: {output_path} ({len(image_list)} frames)")
        return True
    except Exception as e:
        print(f"[Video] Error creating video: {e}")
        return False


class DataCollector:
    def __init__(
        self,
        real_interface,
        simu_interface,
        real_data_root: str,
        simu_data_root: str,
        repo_name: str,
        fps: int = 20,
        video_fps: int = 30,
    ):
        self._real = real_interface
        self._simu = simu_interface
        self._real_data_root = Path(real_data_root)
        self._simu_data_root = Path(simu_data_root)
        self._repo_name = repo_name
        self._fps = fps  # 数据收集帧率
        self._video_fps = video_fps  # 视频播放帧率（使用相机原始帧率）
        self._episode_count = 0
        self._is_collecting = False
        self._collect_thread = None
        self._stop_event = threading.Event()
        
        self._current_real_data = {
            "observations": {"top_images": [], "wrist_images": [], "states": []},
            "actions": [],
        }
        self._current_simu_data = {
            "observations": {"agentview_images": [], "topview_images": [], "states": []},
            "actions": [],
        }
        self._episode_lock = threading.Lock()
        self._save_progress()

    def _get_episode_dir(self, episode_id: int, is_real: bool = True) -> Path:
        root = self._real_data_root if is_real else self._simu_data_root
        episode_dir = root / f"episode_{episode_id:03d}"
        return episode_dir

    def _ensure_directories(self, episode_id: int, real_camera_names: list = None, simu_camera_names: list = None):
        real_episode_dir = self._get_episode_dir(episode_id, is_real=True)
        real_obs_dir = real_episode_dir / "observation"
        real_obs_dir.mkdir(parents=True, exist_ok=True)
        # 根据实际相机创建文件夹
        for cam_name in (real_camera_names or ["top"]):
            (real_obs_dir / f"{cam_name}_image").mkdir(exist_ok=True)
        (real_episode_dir / "action").mkdir(exist_ok=True)
        
        simu_episode_dir = self._get_episode_dir(episode_id, is_real=False)
        simu_obs_dir = simu_episode_dir / "observation"
        simu_obs_dir.mkdir(parents=True, exist_ok=True)
        # 根据实际相机创建文件夹
        for cam_name in (simu_camera_names or ["agentview"]):
            (simu_obs_dir / f"{cam_name}_image").mkdir(exist_ok=True)
        (simu_episode_dir / "action").mkdir(exist_ok=True)

    def start_collection(self):
        if self._is_collecting:
            print("[DataCollector] Already collecting")
            return
        self._stop_event.clear()
        self._is_collecting = True
        self._collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collect_thread.start()
        print("[DataCollector] Collection thread started")

    def stop_collection(self):
        if not self._is_collecting:
            return
        self._is_collecting = False
        self._stop_event.set()
        if self._collect_thread is not None:
            self._collect_thread.join(timeout=2.0)

    def _collect_loop(self):
        print(f"[DataCollector] Collection loop started (target FPS: {self._fps})")
        frame_count = 0
        last_print_time = time.time()
        
        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                timestamp = time.time()
                
                # 获取数据
                t1 = time.time()
                real_images = self._real.get_camera_images()
                t2 = time.time()
                # 使用 get_full_state 一次性获取所有状态，减少通信开销
                real_joints, real_cartesian, real_gripper = self._real.get_full_state()
                t3 = time.time()
                
                simu_images = self._simu.get_camera_images()
                t4 = time.time()
                simu_joints = self._simu.get_joint_state()
                simu_gripper = self._simu.get_gripper_state()
                t5 = time.time()
                
                # 动态保存所有可用相机图像
                t6 = time.time()
                
                with self._episode_lock:
                    # 保存所有真实相机图像（添加 real_ 前缀避免冲突）
                    for cam_name, img in real_images.items():
                        key = f"real_{cam_name}_images"
                        if key not in self._current_real_data["observations"]:
                            self._current_real_data["observations"][key] = []
                        self._current_real_data["observations"][key].append(img.tobytes())
                    
                    real_state = {
                        "joint_positions": real_joints.tolist(),
                        "cartesian_pose": real_cartesian.tolist(),
                        "gripper": float(real_gripper),
                        "timestamp": timestamp,
                    }
                    self._current_real_data["observations"]["states"].append(real_state)
                    
                    real_action = {
                        "joint_positions": real_joints.tolist(),
                        "cartesian_pose": real_cartesian.tolist(),
                        "gripper": float(real_gripper),
                        "timestamp": timestamp,
                    }
                    self._current_real_data["actions"].append(real_action)
                    
                    # 保存所有仿真相机图像
                    for cam_name, img in simu_images.items():
                        key = f"{cam_name}_images"
                        if key not in self._current_simu_data["observations"]:
                            self._current_simu_data["observations"][key] = []
                        self._current_simu_data["observations"][key].append(img.tobytes())
                    
                    simu_state = {
                        "joint_positions": simu_joints.tolist(),
                        "gripper": float(simu_gripper),
                        "timestamp": timestamp,
                    }
                    self._current_simu_data["observations"]["states"].append(simu_state)
                    
                    simu_action = {
                        "joint_positions": simu_joints.tolist(),
                        "gripper": float(simu_gripper),
                        "timestamp": timestamp,
                    }
                    self._current_simu_data["actions"].append(simu_action)
                    
                    frame_count += 1
                t7 = time.time()
                
                # 每秒打印一次性能统计
                if t7 - last_print_time >= 1.0:
                    actual_fps = frame_count / (t7 - last_print_time) if t7 > last_print_time else 0
                    print(f"[DataCollector] Frames: {frame_count}, Actual FPS: {actual_fps:.1f}")
                    print(f"  Timing - real_get_img: {(t2-t1)*1000:.1f}ms, real_get_state: {(t3-t2)*1000:.1f}ms, "
                          f"simu_get_img: {(t4-t3)*1000:.1f}ms, simu_get_state: {(t5-t4)*1000:.1f}ms, "
                          f"img_proc: {(t6-t5)*1000:.1f}ms, lock_save: {(t7-t6)*1000:.1f}ms")
                    last_print_time = t7
                    frame_count = 0
                    
            except Exception as e:
                print(f"[DataCollector] Error collecting data: {e}")
            
            # 计算睡眠时间以维持目标帧率
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / self._fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        print(f"[DataCollector] Collection loop stopped. Total frames: {frame_count}")

    def start_episode(self, task_info: Dict[str, Any], real_camera_names: list = None, simu_camera_names: list = None):
        with self._episode_lock:
            self._episode_count += 1
            episode_id = self._episode_count
            self._ensure_directories(episode_id, real_camera_names, simu_camera_names)
            
            # 动态创建数据结构，根据实际相机
            real_obs = {"states": []}
            for cam_name in (real_camera_names or ["top"]):
                real_obs[f"{cam_name}_images"] = []
            
            simu_obs = {"states": []}
            for cam_name in (simu_camera_names or ["agentview"]):
                simu_obs[f"{cam_name}_images"] = []
            
            self._current_real_data = {
                "observations": real_obs,
                "actions": [],
            }
            self._current_simu_data = {
                "observations": simu_obs,
                "actions": [],
            }
            
            self._current_episode_info = {
                "episode_id": episode_id,
                "task_id": task_info.get("task_id", episode_id),
                "task_name": task_info.get("task_name", ""),
                "description": task_info.get("description", ""),
                "object_position": task_info.get("object_position", []),
                "plate_position": task_info.get("plate_position", []),
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            print(f"[DataCollector] Episode {episode_id} started, data reset")
            return episode_id

    def end_episode(self, episode_id: int, success: bool = True) -> bool:
        with self._episode_lock:
            self._current_episode_info["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._current_episode_info["success"] = success
            self._current_episode_info["episode_id"] = episode_id
            return self._save_episode(episode_id)

    def _save_episode(self, episode_id: int) -> bool:
        try:
            num_real_frames = len(self._current_real_data["observations"]["states"])
            num_simu_frames = len(self._current_simu_data["observations"]["states"])
            
            print(f"[DataCollector] Saving episode {episode_id}")
            print(f"  - Real data: {num_real_frames} frames -> {self._real_data_root}")
            print(f"  - Simu data: {num_simu_frames} frames -> {self._simu_data_root}")
            
            if num_real_frames == 0 and num_simu_frames == 0:
                print("[DataCollector] Warning: No frames collected!")
                return False
            
            self._save_real_data(episode_id)
            self._save_simu_data(episode_id)
            
            self._save_progress()
            print(f"[DataCollector] Episode {episode_id} saved successfully")
            return True
        except Exception as e:
            print(f"[DataCollector] Error saving episode: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_real_data(self, episode_id: int):
        episode_dir = self._get_episode_dir(episode_id, is_real=True)
        obs_dir = episode_dir / "observation"
        video_dir = episode_dir / "video"
        video_dir.mkdir(exist_ok=True)
        
        # 动态保存所有相机的图像和视频
        for key, value in self._current_real_data["observations"].items():
            if key == "states" or not isinstance(value, list) or not value:
                continue
            
            cam_name = key.replace("_images", "")
            img_dir = obs_dir / f"{cam_name}_image"
            
            # 保存为 .npy 文件
            for i, img_bytes in enumerate(value):
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img_array = img_array.reshape(480, 640, 3)
                np.save(img_dir / f"frame_{i:06d}.npy", img_array)
            
            # 生成视频
            video_path = video_dir / f"{cam_name}.mp4"
            create_video_from_images(value, video_path, fps=self._video_fps)
        
        with open(obs_dir / "state.json", "w") as f:
            json.dump(self._current_real_data["observations"]["states"], f, indent=2)
        
        action_dir = episode_dir / "action"
        with open(action_dir / "action.json", "w") as f:
            json.dump(self._current_real_data["actions"], f, indent=2)
        
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(self._current_episode_info, f, indent=2)

    def _save_simu_data(self, episode_id: int):
        episode_dir = self._get_episode_dir(episode_id, is_real=False)
        obs_dir = episode_dir / "observation"
        video_dir = episode_dir / "video"
        video_dir.mkdir(exist_ok=True)
        
        # 动态保存所有相机的图像和视频
        for key, value in self._current_simu_data["observations"].items():
            if key == "states" or not isinstance(value, list) or not value:
                continue
            
            cam_name = key.replace("_images", "")
            img_dir = obs_dir / f"{cam_name}_image"
            
            # 保存为 .npy 文件
            for i, img_bytes in enumerate(value):
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img_array = img_array.reshape(480, 640, 3)
                np.save(img_dir / f"frame_{i:06d}.npy", img_array)
            
            # 生成视频
            video_path = video_dir / f"{cam_name}.mp4"
            create_video_from_images(value, video_path, fps=self._video_fps)
        
        with open(obs_dir / "state.json", "w") as f:
            json.dump(self._current_simu_data["observations"]["states"], f, indent=2)
        
        action_dir = episode_dir / "action"
        with open(action_dir / "action.json", "w") as f:
            json.dump(self._current_simu_data["actions"], f, indent=2)
        
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(self._current_episode_info, f, indent=2)

    def _save_progress(self):
        for root in [self._real_data_root, self._simu_data_root]:
            root.mkdir(parents=True, exist_ok=True)
            progress_file = root / "progress.json"
            progress_data = {
                "episode_count": self._episode_count,
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(progress_file, "w") as f:
                json.dump(progress_data, f, indent=2)

    def load_progress(self) -> int:
        progress_file = self._real_data_root / "progress.json"
        if progress_file.exists():
            with open(progress_file, "r") as f:
                progress_data = json.load(f)
                self._episode_count = progress_data.get("episode_count", 0)
                return self._episode_count
        return 0

    def get_episode_count(self) -> int:
        return self._episode_count

    def is_collecting(self) -> bool:
        return self._is_collecting

    def save_dataset_info(self):
        for root in [self._real_data_root, self._simu_data_root]:
            dataset_info = {
                "repo_name": self._repo_name,
                "episode_count": self._episode_count,
                "fps": self._fps,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(root / "dataset_info.json", "w") as f:
                json.dump(dataset_info, f, indent=2)
