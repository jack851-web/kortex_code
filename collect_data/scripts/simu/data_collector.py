"""
仿真机器人数据收集器 - LeRobot API 直录版

核心设计：
- 使用 LeRobotDataset.create() + streaming_encoding 直接写入 LeRobot v3.0 格式
- 每帧调用 add_frame()，视频实时流式编码为 MP4（libsvtav1/h264）
- observation.state: [6关节角(度), 1夹爪] = 7维 float32
- action: [6关节增量(度), 1夹爪增量] = 7维 float32（= 下一帧state - 当前帧state）
- end_episode 时调用 save_episode()，停止时调用 finalize()
- 保持外部接口与旧版完全兼容
"""
import json
import time
import gc
import numpy as np
import threading
from typing import Dict, List, Any
from pathlib import Path


# 单个 Episode 最大帧数限制
MAX_EPISODE_FRAMES = 5000

# 相机名称到 LeRobot feature key 的映射
# 例如: "top" -> "observation.images.top"
CAMERA_KEY_PREFIX = "observation.images"


def _build_features(camera_names: List[str], image_height: int = 480, image_width: int = 640) -> dict:
    """根据相机列表构建 LeRobot features 字典"""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),  # 6关节角(度) + 1夹爪
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),  # 6关节增量(度) + 1夹爪增量
            "names": ["dj1", "dj2", "dj3", "dj4", "dj5", "dj6", "dgripper"],
        },
    }
    for cam_name in camera_names:
        key = f"{CAMERA_KEY_PREFIX}.{cam_name}"
        features[key] = {
            "dtype": "video",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class SimuDataCollector:
    """仿真机器人数据收集器 - LeRobot API 直录版

    使用 LeRobotDataset.create(streaming_encoding=True) 直接写入 LeRobot v3.0 格式。
    外部接口（start_episode, end_episode, start_recording, stop_recording,
    collect_frame, discard_current_task）与旧版完全兼容。

    数据源：
    - broker 模式：从 MessageBroker 订阅 SimuPublisher 发布的数据
    - 直接模式：直接调用 SimuInterface.get_camera_images()
    """

    def __init__(self, simu_interface, data_root: str, fps: int = 20, video_fps: int = 30,
                 run_in_thread: bool = False, broker=None,
                 repo_id: str = "kortex_simu_dataset",
                 image_size: tuple = (480, 640),
                 use_videos: bool = True):
        self._simu = simu_interface
        self._data_root = Path(data_root)
        self._fps = fps
        self._video_fps = video_fps
        self._use_videos = use_videos
        self._image_height, self._image_width = image_size
        self._episode_count = 0
        self._is_collecting = False
        self._is_recording = False
        self._run_in_thread = run_in_thread
        self._collect_thread = None
        self._stop_event = threading.Event()
        self._data_lock = threading.Lock()
        self._min_frame_interval = (1.0 / float(self._fps)) if self._fps > 0 else 0.0
        self._last_collect_ts = 0.0
        self._repo_id = repo_id

        # Broker 模式
        self._broker = broker
        self._latest_broker_images = None
        self._latest_broker_joints = None
        self._latest_broker_gripper = None
        self._latest_broker_tcp = None

        if self._broker is not None:
            from scripts.core.topic_defs import SIMU_IMAGES, SIMU_JOINTS, SIMU_GRIPPER, SIMU_TCP_POSE
            self._broker.subscribe(SIMU_IMAGES, self._on_broker_images)
            self._broker.subscribe(SIMU_JOINTS, self._on_broker_joints)
            self._broker.subscribe(SIMU_GRIPPER, self._on_broker_gripper)
            self._broker.subscribe(SIMU_TCP_POSE, self._on_broker_tcp)

        # 任务数据保存点（用于重做任务）
        self._task_save_point = 0
        self._last_task_end_point = 0

        # 帧计数
        self._frame_count = 0
        self._current_episode_info = {}
        self._camera_names = []

        # 上一帧的 state（用于计算 action = next_state - current_state）
        self._prev_state = None

        # LeRobot 数据集实例
        self._dataset = None

        # 任务描述（VLA 训练需要）
        self._task_description = ""

        self._save_progress()

    def set_simu_interface(self, simu_interface):
        """更新仿真接口引用（任务切换后 SimuInterface 被重建时调用）"""
        self._simu = simu_interface

    def _init_lerobot_dataset(self):
        """创建或恢复 LeRobot 数据集"""
        import os
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # 禁用 HuggingFace Hub 连接（本地模式）
        os.environ['HF_HUB_OFFLINE'] = '1'

        features = _build_features(
            self._camera_names,
            image_height=self._image_height,
            image_width=self._image_width,
        )

        # 视频特征不使用视频时改为 image
        if not self._use_videos:
            for cam_name in self._camera_names:
                key = f"{CAMERA_KEY_PREFIX}.{cam_name}"
                features[key]["dtype"] = "image"

        # 检查本地数据集是否有效（必须有 meta/info.json）
        info_path = self._data_root / "meta" / "info.json"
        dataset_exists = info_path.exists()

        if not dataset_exists:
            # 数据集不存在或无效，清理目录后创建新数据集
            if self._data_root.exists():
                import shutil
                print(f"[SimuDataCollector] Cleaning incomplete dataset at {self._data_root}")
                shutil.rmtree(self._data_root)

            try:
                self._dataset = LeRobotDataset.create(
                    repo_id=self._repo_id,
                    fps=self._fps,
                    features=features,
                    root=self._data_root,
                    robot_type="kortex",
                    use_videos=self._use_videos,
                    streaming_encoding=self._use_videos,  # 启用流式视频编码
                    vcodec="libsvtav1",
                    metadata_buffer_size=10,
                    encoder_queue_maxsize=30,
                )
                print(f"[SimuDataCollector] New LeRobot dataset created at {self._data_root}")
            except Exception as e:
                print(f"[SimuDataCollector] Failed to create dataset: {e}")
                raise
        else:
            # 数据集已存在且有效，加载现有数据集继续追加
            print(f"[SimuDataCollector] Loading existing LeRobot dataset from {self._data_root}")
            try:
                self._dataset = LeRobotDataset(
                    repo_id=self._repo_id,
                    root=self._data_root,
                    revision="local",  # 使用本地版本，避免连接 Hub
                )
                # 创建新的 episode buffer
                self._dataset.episode_buffer = self._dataset.create_episode_buffer()
            except Exception as e:
                print(f"[SimuDataCollector] Failed to load existing dataset: {e}")
                # 如果加载失败，删除并重新创建
                import shutil
                shutil.rmtree(self._data_root)
                self._dataset = LeRobotDataset.create(
                    repo_id=self._repo_id,
                    fps=self._fps,
                    features=features,
                    root=self._data_root,
                    robot_type="kortex",
                    use_videos=self._use_videos,
                    streaming_encoding=self._use_videos,
                    vcodec="libsvtav1",
                    metadata_buffer_size=10,
                    encoder_queue_maxsize=30,
                )
                print(f"[SimuDataCollector] Recreated LeRobot dataset at {self._data_root}")

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
        """停止收集，关闭数据集"""
        if not self._is_collecting:
            return

        self._is_collecting = False
        self._stop_event.set()

        if self._collect_thread is not None:
            self._collect_thread.join(timeout=2.0)
            self._collect_thread = None

        self._is_recording = False

        # 如果还有未保存的 episode buffer，先保存
        if self._dataset is not None and self._dataset.episode_buffer is not None:
            if self._dataset.episode_buffer.get("size", 0) > 0:
                try:
                    self._dataset.save_episode()
                    print("[SimuDataCollector] Saved remaining episode buffer on stop")
                except Exception as e:
                    print(f"[SimuDataCollector] Warning: failed to save remaining buffer: {e}")
            # 结束数据集
            try:
                self._dataset.finalize()
            except Exception as e:
                print(f"[SimuDataCollector] Warning: finalize error: {e}")
            self._dataset = None

        print("[SimuDataCollector] Stopped")

    def _collect_loop(self):
        if hasattr(self._simu, '_collector_thread_id'):
            self._simu._collector_thread_id = threading.current_thread()
        while not self._stop_event.is_set():
            loop_start = time.time()
            self.collect_frame()
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, (1.0 / self._fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ================================================================
    # Broker 回调
    # ================================================================

    def _on_broker_images(self, images):
        if images is not None:
            self._latest_broker_images = {k: np.array(v, copy=True) for k, v in images.items()}

    def _on_broker_joints(self, joints):
        if joints is not None:
            self._latest_broker_joints = np.copy(joints)

    def _on_broker_gripper(self, gripper):
        if gripper is not None:
            self._latest_broker_gripper = float(gripper)

    def _on_broker_tcp(self, tcp):
        if tcp is not None:
            pos, rot = tcp
            self._latest_broker_tcp = np.copy(pos)

    def collect_frame(self):
        """收集一帧数据，通过 LeRobot API 写入数据集"""
        if not self._is_collecting or not self._is_recording:
            return

        if self._frame_count >= MAX_EPISODE_FRAMES:
            return

        if self._dataset is None:
            return

        try:
            timestamp = time.time()
            if self._min_frame_interval > 0 and (timestamp - self._last_collect_ts) < self._min_frame_interval:
                return
            self._last_collect_ts = timestamp

            # 获取数据
            if self._broker is not None and self._latest_broker_images is not None:
                images = self._latest_broker_images
                joints_rad = self._latest_broker_joints
                gripper = self._latest_broker_gripper if self._latest_broker_gripper is not None else 0.0
                joints = np.rad2deg(joints_rad) if joints_rad is not None else np.zeros(6)
            else:
                images = self._simu.get_camera_images(self._camera_names)
                joints = self._simu.get_joint_state()
                gripper = self._simu.get_gripper_state()

            with self._data_lock:
                # 构建 state 向量: [6关节角(度), 1夹爪]
                state_vec = np.array(
                    list(joints) + [float(gripper)],
                    dtype=np.float32,
                )

                # 构建 action: 下一帧state - 当前帧state（增量）
                if self._prev_state is not None:
                    action_vec = (state_vec - self._prev_state).astype(np.float32)
                else:
                    action_vec = np.zeros(7, dtype=np.float32)
                self._prev_state = state_vec.copy()

                # 构建 LeRobot frame
                frame = {
                    "observation.state": state_vec,
                    "action": action_vec,
                    "task": self._task_description,
                }

                # 图像：LeRobot 要求 (H, W, C) RGB 格式
                # 仿真相机输出 RGB，无需转换
                for cam_name, img in images.items():
                    key = f"{CAMERA_KEY_PREFIX}.{cam_name}"
                    # 确保是 (H, W, C) 的 RGB numpy array
                    if isinstance(img, np.ndarray):
                        if img.ndim == 3 and img.shape[2] == 3:
                            # MuJoCo 渲染输出 RGB，直接使用
                            frame[key] = img.astype(np.uint8)
                        else:
                            frame[key] = img
                    else:
                        frame[key] = img

                # 通过 LeRobot API 添加帧
                self._dataset.add_frame(frame)
                self._frame_count += 1

        except Exception as e:
            print(f"[SimuDataCollector] Error: {e}")
            import traceback
            traceback.print_exc()

    def start_episode(self, episode_id: int, camera_names: list, task_info: Dict[str, Any]):
        """开始一个新的 episode

        Args:
            episode_id: episode 编号
            camera_names: 相机名称列表
            task_info: 任务信息，支持两种格式:
                - 旧格式: {"task_id": ..., "task_name": ..., "description": ...}
                - 新格式: {"tasks": [{"task_name": ..., "description": ...}, ...]}
        """
        self._episode_count = episode_id
        self._camera_names = camera_names

        # 解析任务描述
        self._task_description = self._parse_task_description(task_info)

        with self._data_lock:
            self._frame_count = 0
            self._prev_state = None
            self._current_episode_info = {
                "episode_id": episode_id,
                "task_id": task_info.get("task_id", episode_id),
                "task_name": task_info.get("task_name", ""),
                "description": task_info.get("description", ""),
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            # 初始化 LeRobot 数据集（仅首次）
            if self._dataset is None:
                self._init_lerobot_dataset()

            # 创建新的 episode buffer
            self._dataset.episode_buffer = self._dataset.create_episode_buffer()

        print(f"[SimuDataCollector] Episode {episode_id} started (LeRobot mode, task: {self._task_description})")

    def start_recording(self):
        """开始记录数据（任务执行期间）"""
        with self._data_lock:
            self._task_save_point = self._frame_count
        self._is_recording = True
        self._last_collect_ts = 0.0

        if hasattr(self._simu, 'set_collecting_active'):
            self._simu.set_collecting_active(True)
            self._simu._collector_thread_id = threading.current_thread()

        print(f"[SimuDataCollector] Recording started, save point: frame={self._task_save_point}")

    def stop_recording(self):
        """停止记录数据"""
        self._is_recording = False
        if hasattr(self._simu, 'set_collecting_active'):
            self._simu.set_collecting_active(False)
            self._simu._collector_thread_id = None
        with self._data_lock:
            self._last_task_end_point = self._frame_count
        print("[SimuDataCollector] Recording stopped")

    def discard_current_task(self):
        """丢弃当前任务的数据（用于重做任务）

        策略：重置 episode buffer 到保存点的状态。
        由于 LeRobot 的 streaming encoder 不支持截断，这里通过重建 episode buffer 实现。
        注意：已编码的视频帧无法回收，但 parquet 数据会以正确的帧数保存。
        """
        with self._data_lock:
            save_point = self._task_save_point
            self._frame_count = save_point
            self._prev_state = None

            # 重置 episode buffer（丢弃 add_frame 添加的数据）
            if self._dataset is not None:
                ep_idx = self._dataset.meta.total_episodes
                self._dataset.episode_buffer = self._dataset.create_episode_buffer(ep_idx)

            print(f"[SimuDataCollector] Current task data discarded, reverted to frame={save_point}")

    def end_episode(self, episode_id: int, success: bool = True):
        """结束 episode，保存到 LeRobot 数据集"""
        self._is_recording = False

        with self._data_lock:
            self._current_episode_info["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._current_episode_info["success"] = success

            t0 = time.time()

            num_frames = self._frame_count
            if self._dataset is not None and num_frames > 0:
                try:
                    self._dataset.save_episode()
                    self._patch_info_json()  # 补全 total_videos 等字段
                    print(f"[SimuDataCollector] Episode {episode_id} saved to LeRobot ({num_frames} frames)")
                except Exception as e:
                    print(f"[SimuDataCollector] Save error: {e}")
                    import traceback
                    traceback.print_exc()
            elif num_frames == 0:
                print(f"[SimuDataCollector] Episode {episode_id}: No frames collected, skipping save")

            elapsed = time.time() - t0
            print(f"[SimuDataCollector] Episode {episode_id} completed in {elapsed:.1f}s")

            # 重置状态
            self._frame_count = 0
            self._prev_state = None
            self._task_save_point = 0
            self._last_task_end_point = 0

            # 创建下一个 episode 的 buffer（为下一个 episode 准备好）
            if self._dataset is not None:
                self._dataset.episode_buffer = self._dataset.create_episode_buffer()

            gc.collect()

            return num_frames > 0

    def _patch_info_json(self):
        """补全 info.json 和缺失的元数据文件（rerun 可视化需要）"""
        info_path = self._data_root / "meta" / "info.json"
        if not info_path.exists():
            return

        try:
            with open(info_path, "r") as f:
                info = json.load(f)

            patched = False

            # total_videos: 视频特征数 × episodes
            if "total_videos" not in info:
                video_count = sum(1 for feat in info.get("features", {}).values()
                                  if isinstance(feat, dict) and feat.get("dtype") == "video")
                info["total_videos"] = video_count * info.get("total_episodes", 0)
                patched = True

            # total_chunks: 实际数据目录中的 chunk 文件夹数
            if "total_chunks" not in info:
                data_dir = self._data_root / "data"
                if data_dir.exists():
                    chunks = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")]
                    info["total_chunks"] = len(chunks)
                else:
                    info["total_chunks"] = 0
                patched = True

            if patched:
                with open(info_path, "w") as f:
                    json.dump(info, f, indent=2)
                print(f"[SimuDataCollector] Patched info.json")

            # 生成 episodes parquet（rerun 需要）
            self._write_episodes_parquet()

        except Exception as e:
            print(f"[SimuDataCollector] Warning: failed to patch metadata: {e}")

    def _write_episodes_parquet(self):
        """生成 meta/episodes/chunk-000/file-000.parquet（v3 格式，rerun 可视化需要）"""
        import pandas as pd

        episodes_dir = self._data_root / "meta" / "episodes"
        episodes_path = episodes_dir / "chunk-000" / "file-000.parquet"
        if episodes_path.exists():
            return

        tasks_path = self._data_root / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            return

        try:
            tasks_df = pd.read_parquet(tasks_path)
            total_episodes = len(tasks_df)
            total_frames = 0

            rows = []
            for idx, row in tasks_df.iterrows():
                ep_idx = int(row.get("episode", idx))
                task_desc = str(row.get("task", "")) or "Grasp the object"
                length = self._estimate_episode_length(idx, total_episodes)
                total_frames += length

                rows.append({
                    "episode_index": ep_idx,
                    "tasks": [task_desc],
                    "length": length,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                })

            df = pd.DataFrame(rows)
            df.to_parquet(episodes_path, index=False)

            # 同时删除旧的 .jsonl（如果存在）
            legacy_jsonl = self._data_root / "meta" / "episodes.jsonl"
            if legacy_jsonl.exists():
                legacy_jsonl.unlink()

            print(f"[SimuDataCollector] Generated {episodes_path} ({total_episodes} episodes)")
        except Exception as e:
            print(f"[SimuDataCollector] Warning: failed to generate episodes parquet: {e}")

    def _estimate_episode_length(self, episode_index, total_episodes):
        """估算单个 episode 的帧数"""
        stats_path = self._data_root / "meta" / "stats.json"
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                # stats 中有各 feature 的 count，取 observation.state 的总帧数 / episodes
                state_key = "observation.state"
                if state_key in stats:
                    return stats[state_key].get("count", 0) // total_episodes
            except Exception:
                pass
        return 0

    def finalize(self):
        """关闭数据集（在所有 episode 收集完毕后调用）"""
        if self._dataset is not None:
            try:
                self._dataset.finalize()
                print("[SimuDataCollector] Dataset finalized")
            except Exception as e:
                print(f"[SimuDataCollector] Finalize error: {e}")
            # 补全 rerun 需要的字段
            self._patch_info_json()
            self._dataset = None

    def _parse_task_description(self, task_info: Dict[str, Any]) -> str:
        """从 task_info 解析出任务描述字符串"""
        # 新格式: {"tasks": [...]}
        if "tasks" in task_info:
            tasks = task_info["tasks"]
            if isinstance(tasks, list) and len(tasks) > 0:
                # 取最后一个任务的描述（通常是当前正在执行的任务）
                last_task = tasks[-1]
                desc = last_task.get("task_name", "") or last_task.get("description", "")
                if desc:
                    return desc

        # 旧格式
        desc = task_info.get("task_name", "") or task_info.get("description", "")
        return desc if desc else "Grasp the object"

    def _save_progress(self):
        self._data_root.mkdir(parents=True, exist_ok=True)
        progress_file = self._data_root / "progress.json"
        with open(progress_file, "w") as f:
            json.dump({"episode_count": self._episode_count}, f, indent=2)

    def load_progress(self) -> int:
        progress_file = self._data_root / "progress.json"
        if progress_file.exists():
            with open(progress_file, "r") as f:
                self._episode_count = json.load(f).get("episode_count", 0)
        return self._episode_count
