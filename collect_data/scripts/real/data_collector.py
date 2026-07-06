"""
真实机器人数据收集器 - LeRobot API 直录版

核心设计：
- 使用 LeRobotDataset.create() + streaming_encoding 直接写入 LeRobot v3.0 格式
- 每帧调用 add_frame()，视频实时流式编码为 MP4
- observation.state: [6关节角(度, 0-360), 1夹爪(0~1), 3末端位姿(米)] = 10维 float32
- action: [6关节增量(度), 1夹爪增量, 3末端位姿增量(米)] = 10维 float32
- end_episode 时调用 save_episode()，停止时调用 finalize()
- 保持外部接口与旧版完全兼容
"""
import json
import time
import gc
import os
import shutil
import logging
import traceback
import numpy as np
import threading
from typing import Dict, List, Any
from pathlib import Path
from scripts.core.data_utils import parse_task_description, save_progress, load_progress

logger = logging.getLogger(__name__)


MAX_EPISODE_FRAMES = 5000
CAMERA_KEY_PREFIX = "observation.images"


def _build_features(camera_names: List[str], image_height: int = 480, image_width: int = 640) -> dict:
    """根据相机列表构建 LeRobot features 字典"""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (10,),  # 6关节角(度) + 1夹爪 + 3末端位姿(米)
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper", "ee_x", "ee_y", "ee_z"],
        },
        "action": {
            "dtype": "float32",
            "shape": (10,),  # 6关节增量(度) + 1夹爪增量 + 3末端位姿增量(米)
            "names": ["dj1", "dj2", "dj3", "dj4", "dj5", "dj6", "dgripper", "dee_x", "dee_y", "dee_z"],
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


class RealDataCollector:
    """真实机器人数据收集器 - LeRobot API 直录版

    使用 LeRobotDataset.create(streaming_encoding=True) 直接写入 LeRobot v3.0 格式。
    外部接口与旧版完全兼容。

    数据源：
    - broker 模式：从 MessageBroker 订阅 RealPublisher 发布的数据
    - 直接模式：直接调用 RealInterface.get_camera_images()
    """

    def __init__(self, real_interface, data_root: str, fps: int = 20, video_fps: int = 30,
                 broker=None, repo_id: str = "kortex_real_dataset",
                 image_size: tuple = (480, 640),
                 use_videos: bool = True,
                 camera_names: list = None):
        self._real = real_interface
        self._data_root = Path(data_root)
        self._fps = fps
        self._video_fps = video_fps
        self._use_videos = use_videos
        self._image_height, self._image_width = image_size
        self._episode_count = 0
        self._is_collecting = False
        self._is_recording = False
        self._collect_thread = None
        self._stop_event = threading.Event()
        self._data_lock = threading.Lock()
        self._repo_id = repo_id
        self._camera_names = camera_names or []

        # 任务数据保存点
        self._task_save_point = 0
        self._last_task_end_point = 0

        # 帧计数
        self._frame_count = 0
        self._current_episode_info = {}

        # 上一帧的 state（用于计算增量 action）
        self._prev_state = None

        # LeRobot 数据集实例
        self._dataset = None

        # 任务描述
        self._task_description = ""

        # Broker 模式
        self._broker = broker
        self._latest_broker_images = None
        self._latest_broker_joints = None
        self._latest_broker_cartesian = None
        self._latest_broker_gripper = None

        if self._broker is not None:
            from scripts.core.topic_defs import REAL_IMAGES, REAL_JOINTS, REAL_CARTESIAN, REAL_GRIPPER
            self._broker.subscribe(REAL_IMAGES, self._on_broker_images)
            self._broker.subscribe(REAL_JOINTS, self._on_broker_joints)
            self._broker.subscribe(REAL_CARTESIAN, self._on_broker_cartesian)
            self._broker.subscribe(REAL_GRIPPER, self._on_broker_gripper)

        # 恢复已有进度（不主动创建文件）
        self._episode_count = load_progress(self._data_root)

    # ================================================================
    # Broker 回调
    # ================================================================

    def _on_broker_images(self, images):
        if images is not None:
            self._latest_broker_images = {k: np.array(v, copy=True) for k, v in images.items()}

    def _on_broker_joints(self, joints):
        if joints is not None:
            self._latest_broker_joints = np.copy(joints)

    def _on_broker_cartesian(self, cartesian):
        if cartesian is not None:
            self._latest_broker_cartesian = np.copy(cartesian)

    def _on_broker_gripper(self, gripper):
        if gripper is not None:
            self._latest_broker_gripper = float(gripper)

    def _init_lerobot_dataset(self):
        """创建或恢复 LeRobot 数据集（纯本地模式，不连接 HuggingFace）"""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # 强制离线模式：覆盖已有设置，禁止任何 HF 网络请求
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        features = _build_features(
            self._camera_names,
            image_height=self._image_height,
            image_width=self._image_width,
        )

        if not self._use_videos:
            for cam_name in self._camera_names:
                key = f"{CAMERA_KEY_PREFIX}.{cam_name}"
                features[key]["dtype"] = "image"

        # 检查目录是否有效数据集
        info_path = self._data_root / "meta" / "info.json"
        tasks_path = self._data_root / "meta" / "tasks.parquet"

        needs_create = True
        if self._data_root.exists() and info_path.exists() and tasks_path.exists():
            needs_create = False

        if needs_create:
            # 清理旧目录后重新创建
            if self._data_root.exists():
                logger.info(f"[RealDataCollector] Removing incomplete dataset: {self._data_root}")
                try:
                    shutil.rmtree(self._data_root)
                except Exception as e:
                    logger.warning(f"[RealDataCollector] Failed to remove: {e}")

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
            logger.info(f"[RealDataCollector] LeRobot dataset created at {self._data_root}")
        else:
            logger.info(f"[RealDataCollector] Loading existing LeRobot dataset from {self._data_root}")
            self._dataset = LeRobotDataset(
                repo_id=self._repo_id,
                root=self._data_root,
            )
            self._dataset.episode_buffer = self._dataset.create_episode_buffer()
            # 修复：加载已有数据集时 _streaming_encoder 为 None，
            # 导致 add_frame 回退到 images/ 路径写 PNG 帧，需重新初始化
            if self._use_videos:
                from lerobot.datasets.video_encoder import StreamingVideoEncoder
                self._dataset._streaming_encoder = StreamingVideoEncoder(
                    fps=self._dataset.meta.fps,
                    vcodec=self._dataset.vcodec,
                    pix_fmt="yuv420p",
                    g=2,
                    crf=30,
                    preset=None,
                    queue_maxsize=30,
                )
                logger.info(f"[RealDataCollector] Re-initialized streaming encoder for existing dataset")

    def start_collection(self):
        """开始数据收集：初始化 LeRobot 数据集 + 启动采集线程"""
        if self._is_collecting:
            return

        # 一次性初始化数据集（不再在 start_episode 中懒加载）
        if self._dataset is None:
            self._init_lerobot_dataset()

        self._stop_event.clear()
        self._is_collecting = True
        self._collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collect_thread.start()
        logger.info("[RealDataCollector] Started (LeRobot mode)")

    def stop_collection(self):
        if not self._is_collecting:
            return
        self._is_collecting = False
        self._stop_event.set()

        if self._collect_thread:
            self._collect_thread.join(timeout=2.0)

        self._is_recording = False

        if self._dataset is not None and self._dataset.episode_buffer is not None:
            if self._dataset.episode_buffer.get("size", 0) > 0:
                try:
                    self._dataset.save_episode()
                    logger.info("[RealDataCollector] Saved remaining episode buffer on stop")
                except Exception as e:
                    logger.warning(f"[RealDataCollector] failed to save remaining buffer: {e}")
            try:
                self._dataset.finalize()
            except Exception as e:
                logger.warning(f"[RealDataCollector] finalize error: {e}")
            self._dataset = None

        logger.info("[RealDataCollector] Stopped")

    def _collect_loop(self):
        logger.info(f"[RealDataCollector] Loop started (target FPS: {self._fps})")

        while not self._stop_event.is_set():
            loop_start = time.time()

            try:
                if not self._is_recording:
                    time.sleep(0.01)
                    continue

                if self._frame_count >= MAX_EPISODE_FRAMES:
                    time.sleep(0.01)
                    continue

                timestamp = time.time()

                # 获取数据
                if self._broker is not None and self._latest_broker_images is not None:
                    images = self._latest_broker_images
                    joints = self._latest_broker_joints if self._latest_broker_joints is not None else np.zeros(6)
                    gripper = self._latest_broker_gripper if self._latest_broker_gripper is not None else 0.0
                    cartesian = self._latest_broker_cartesian  # [x, y, z] 或 None
                else:
                    images = self._real.get_camera_images()
                    joints, cartesian, gripper = self._real.get_full_state()

                with self._data_lock:
                    # 构建 state 向量: [6关节角(度, 0-360), 1夹爪(0~1), 3末端位姿(米)] = 10维
                    joints_raw = np.array(joints[:6], dtype=np.float32)
                    if cartesian is not None:
                        ee_raw = np.array(cartesian[:3], dtype=np.float32)
                    else:
                        ee_raw = np.zeros(3, dtype=np.float32)

                    state_vec = np.array(
                        list(joints_raw) + [float(gripper)] + list(ee_raw),
                        dtype=np.float32,
                    )

                    # 构建 action: 增量
                    if self._prev_state is not None:
                        action_vec = (state_vec - self._prev_state).astype(np.float32)
                    else:
                        action_vec = np.zeros(10, dtype=np.float32)
                    self._prev_state = state_vec.copy()

                    # 构建 LeRobot frame
                    frame = {
                        "observation.state": state_vec,
                        "action": action_vec,
                        "task": self._task_description,
                    }

                    # 图像：OpenCV 输出 BGR，LeRobot 需要 RGB
                    import cv2
                    for cam_name, img in images.items():
                        key = f"{CAMERA_KEY_PREFIX}.{cam_name}"
                        if isinstance(img, np.ndarray):
                            if img.ndim == 3 and img.shape[2] == 3:
                                # BGR -> RGB
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                frame[key] = img_rgb.astype(np.uint8)
                            else:
                                frame[key] = img
                        else:
                            frame[key] = img

                    self._dataset.add_frame(frame)
                    self._frame_count += 1

                    # 诊断：前5帧打印关节值，确认是原始范围还是0-1
                    if self._frame_count <= 5:
                        logger.info(f"[RealDataCollector] Frame {self._frame_count} raw state: "
                              f"j1={state_vec[0]:.1f}° j2={state_vec[1]:.1f}° "
                              f"j5={state_vec[4]:.1f}° j6={state_vec[5]:.1f}° "
                              f"gripper={state_vec[6]:.3f} ee_z={state_vec[9]:.4f}m")

            except Exception as e:
                logger.error(f"[RealDataCollector] Error: {e}")
                logger.debug(traceback.format_exc())

            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / self._fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    @property
    def episode_count(self) -> int:
        """当前已采集的 episode 数量"""
        return self._episode_count

    def start_episode(self, episode_id: int, camera_names: list, task_info: Dict[str, Any]):
        """开始一个新的 episode（数据集在 start_collection 时已初始化）"""
        self._episode_count = episode_id
        self._camera_names = camera_names

        # 解析任务描述
        self._task_description = parse_task_description(task_info)

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

            if self._dataset is None:
                self._init_lerobot_dataset()

            self._dataset.episode_buffer = self._dataset.create_episode_buffer()

        logger.info(f"[RealDataCollector] Episode {episode_id} started (LeRobot mode, task: {self._task_description})")

    def start_recording(self):
        with self._data_lock:
            self._task_save_point = self._frame_count
        self._is_recording = True
        logger.info(f"[RealDataCollector] Recording started, save point: frame={self._task_save_point}")

    def stop_recording(self):
        self._is_recording = False
        with self._data_lock:
            self._last_task_end_point = self._frame_count
        logger.info("[RealDataCollector] Recording stopped")

    def discard_current_task(self):
        """丢弃当前任务的数据"""
        with self._data_lock:
            save_point = self._task_save_point
            self._frame_count = save_point
            self._prev_state = None

            if self._dataset is not None:
                ep_idx = self._dataset.meta.total_episodes
                self._dataset.episode_buffer = self._dataset.create_episode_buffer(ep_idx)

            logger.info(f"[RealDataCollector] Current task data discarded, reverted to frame={save_point}")

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
                    logger.info(f"[RealDataCollector] Episode {episode_id} saved to LeRobot ({num_frames} frames)")
                except Exception as e:
                    logger.error(f"[RealDataCollector] Save error: {e}")
                    logger.debug(traceback.format_exc())
            elif num_frames == 0:
                logger.info(f"[RealDataCollector] Episode {episode_id}: No frames collected, skipping save")

            elapsed = time.time() - t0
            logger.info(f"[RealDataCollector] Episode {episode_id} completed in {elapsed:.1f}s")

            self._frame_count = 0
            self._prev_state = None
            self._task_save_point = 0
            self._last_task_end_point = 0

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
                logger.info(f"[RealDataCollector] Patched info.json")

            # 生成 episodes parquet（rerun 需要）
            self._write_episodes_parquet()

        except Exception as e:
            logger.warning(f"[RealDataCollector] failed to patch metadata: {e}")

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

            rows = []
            for idx, row in tasks_df.iterrows():
                ep_idx = int(row.get("episode", idx))
                task_desc = str(row.get("task", "")) or "Grasp the object"
                length = self._estimate_episode_length(idx, total_episodes)

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

            # 删除旧的 .jsonl
            legacy_jsonl = self._data_root / "meta" / "episodes.jsonl"
            if legacy_jsonl.exists():
                legacy_jsonl.unlink()

            logger.info(f"[RealDataCollector] Generated {episodes_path} ({total_episodes} episodes)")
        except Exception as e:
            logger.warning(f"[RealDataCollector] failed to generate episodes parquet: {e}")

    def _estimate_episode_length(self, episode_index, total_episodes):
        """估算单个 episode 的帧数"""
        stats_path = self._data_root / "meta" / "stats.json"
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                state_key = "observation.state"
                if state_key in stats:
                    return stats[state_key].get("count", 0) // total_episodes
            except Exception:
                logger.debug("Failed to read stats for frames_per_episode", exc_info=True)
        return 0

    def finalize(self):
        """关闭数据集"""
        if self._dataset is not None:
            try:
                self._dataset.finalize()
                logger.info("[RealDataCollector] Dataset finalized")
            except Exception as e:
                logger.info(f"[RealDataCollector] Finalize error: {e}")
            # 补全 rerun 需要的字段
            self._patch_info_json()
            self._dataset = None

    def save_progress(self):
        """保存收集进度（公共接口）"""
        save_progress(self._data_root, self._episode_count)

    def load_progress(self) -> int:
        self._episode_count = load_progress(self._data_root)
        return self._episode_count
